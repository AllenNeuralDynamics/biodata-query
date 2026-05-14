"""High-level helpers for finding and filtering assets."""

from __future__ import annotations

import logging

import pandas as pd
from zombie_squirrel import asset_basics, qc, raw_to_derived

from biodata_query.query import retrieve_records

logger = logging.getLogger(__name__)


def find_assets(
    modalities: list[str],
    query: dict | None = None,
    raw_asset_names: list[str] | None = None,
    metric_names: list[str] | None = None,
    latest: bool = True,
) -> pd.DataFrame:
    """Find derived assets for raw assets and optionally enrich with QC status.

    Parameters
    ----------
    modalities:
        Required list of modality abbreviations (e.g. ``["behavior"]``) to
        look for when resolving derived assets via ``raw_to_derived``.
    query:
        MongoDB-style filter dict routed through the cache-aware query module.
        Mutually exclusive with *raw_asset_names*.
    raw_asset_names:
        Explicit list of raw asset names.  Mutually exclusive with *query*.
    metric_names:
        Optional list of QC metric names (e.g. ``["Running Velocity",
        "General Performance"]``) that must be passing.  When provided, QC
        columns are added to the result.
    latest:
        Passed to ``raw_to_derived``; when ``True`` only the most recent
        derived asset per pipeline is returned.

    Returns
    -------
    pd.DataFrame
        Always contains:

        - ``raw_asset_name``
        - ``derived_asset_name``
        - ``modality``

        When *metric_names* is provided, the following columns are added:

        - ``{modality}_pass`` — all QC rows for this modality/asset pass
        - ``{metric_name}_pass`` — one column per requested metric
        - ``all_metrics_pass`` — every requested metric passes
        - ``qc_pass`` — ``all_metrics_pass`` AND the modality-level pass

    Raises
    ------
    ValueError
        If neither or both of *query* and *raw_asset_names* are supplied.
    """
    if (query is None) == (raw_asset_names is None):
        raise ValueError("Provide exactly one of `query` or `raw_asset_names`.")

    # ------------------------------------------------------------------ #
    # Step 1: resolve raw asset names                                      #
    # ------------------------------------------------------------------ #
    if raw_asset_names is not None:
        raw_names: list[str] = list(raw_asset_names)
    else:
        result = retrieve_records(query, names_only=True)
        raw_names = result.asset_names

    if not raw_names:
        return pd.DataFrame(columns=["raw_asset_name", "derived_asset_name", "modality"])

    # ------------------------------------------------------------------ #
    # Step 2: resolve derived asset names                                  #
    # ------------------------------------------------------------------ #
    rows: list[dict] = []
    for raw_name in raw_names:
        for modality in modalities:
            derived = raw_to_derived(raw_name, modality=modality, latest=latest)
            for derived_name in derived:
                rows.append(
                    {
                        "raw_asset_name": raw_name,
                        "derived_asset_name": derived_name,
                        "modality": modality,
                    }
                )

    df = pd.DataFrame(rows, columns=["raw_asset_name", "derived_asset_name", "modality"])

    if not metric_names or df.empty:
        return df

    # ------------------------------------------------------------------ #
    # Step 3: QC enrichment                                                #
    # ------------------------------------------------------------------ #
    # Look up subject_id for each derived asset via the local cache
    ab = asset_basics()
    derived_meta = (
        ab[ab["name"].isin(df["derived_asset_name"])][["name", "subject_id"]]
        .rename(columns={"name": "derived_asset_name"})
    )
    df = df.merge(derived_meta, on="derived_asset_name", how="left")

    # Pre-allocate QC columns (None = no QC data found)
    for modality in modalities:
        df[modality] = None
    for metric in metric_names:
        df[metric] = None
    df["all_metrics_pass"] = False
    df["qc_pass"] = False

    # Fetch QC data once per subject and join back
    for subject_id, group in df.groupby("subject_id", dropna=True):
        asset_names_for_subject = group["derived_asset_name"].tolist()
        try:
            qc_df = qc(subject_id=str(subject_id), asset_names=asset_names_for_subject)
        except Exception:
            logger.warning("Failed to fetch QC for subject %s", subject_id)
            continue

        if qc_df.empty or "status" not in qc_df.columns:
            continue

        for idx, row in group.iterrows():
            asset_name = row["derived_asset_name"]
            modality = row["modality"]
            asset_qc = qc_df[qc_df["asset_name"] == asset_name]

            # Modality-level: aggregate status across all QC rows for this modality
            modality_rows = asset_qc[asset_qc["modality"] == modality]
            if not modality_rows.empty:
                statuses = modality_rows["status"]
                if (statuses == "Pass").all():
                    modality_status = "Pass"
                elif (statuses == "Fail").any():
                    modality_status = "Fail"
                else:
                    modality_status = "Pending"
                df.at[idx, modality] = modality_status
            else:
                modality_status = None

            # Per-metric: use the actual status value from the QC row
            all_metrics_pass = True
            for metric in metric_names:
                metric_rows = asset_qc[
                    (asset_qc["name"] == metric)
                    & (asset_qc["modality"] == modality)
                ]
                if not metric_rows.empty:
                    metric_status = metric_rows["status"].iloc[0]
                    df.at[idx, metric] = metric_status
                else:
                    metric_status = None
                if metric_status != "Pass":
                    all_metrics_pass = False

            df.at[idx, "all_metrics_pass"] = all_metrics_pass
            df.at[idx, "qc_pass"] = all_metrics_pass and modality_status == "Pass"

    df = df.drop(columns=["subject_id"])
    return df
