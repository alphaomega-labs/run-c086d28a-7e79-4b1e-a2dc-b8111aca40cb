from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PreparedRealSetting:
    name: str
    source_paths: list[str]
    target_unlabeled_path: str
    target_test_path: str
    feature_columns: list[str]
    target_column: str
    license_status: str
    schema_transform: str
    provenance: str


def _parse_dataset_name(name: str) -> tuple[str, str, str]:
    # Parse "family:src->tgt" identifiers used in UDA-4-TSC result tables.
    family, pair = name.split(":", 1)
    src_domain, tgt_domain = pair.split("->", 1)
    return family, src_domain, tgt_domain


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["dataset_name"].map(_parse_dataset_name)
    out["dataset_family"] = parsed.map(lambda t: t[0])
    out["source_domain"] = parsed.map(lambda t: t[1])
    out["target_domain"] = parsed.map(lambda t: t[2])

    out["classifier_code"] = out["classifier_name"].astype("category").cat.codes.astype(float)
    out["dataset_family_code"] = out["dataset_family"].astype("category").cat.codes.astype(float)
    out["source_domain_code"] = out["source_domain"].astype("category").cat.codes.astype(float)
    out["target_domain_code"] = out["target_domain"].astype("category").cat.codes.astype(float)
    out["domain_shift_indicator"] = (out["source_domain"] != out["target_domain"]).astype(float)
    out["source_minus_iwcv"] = out["source_accuracy"] - out["iwcv_accuracy"]
    out["source_target_gap"] = out["source_accuracy"] - out["target_accuracy"]
    out["iwcv_target_gap"] = out["iwcv_accuracy"] - out["target_accuracy"]
    return out


def _split_sources(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classifiers = sorted(df["classifier_name"].unique().tolist())
    if len(classifiers) < 2:
        return df.iloc[::2].copy(), df.iloc[1::2].copy()
    midpoint = max(1, len(classifiers) // 2)
    left = set(classifiers[:midpoint])
    src_a = df[df["classifier_name"].isin(left)].copy()
    src_b = df[~df["classifier_name"].isin(left)].copy()
    if src_a.empty or src_b.empty:
        src_a = df.iloc[::2].copy()
        src_b = df.iloc[1::2].copy()
    return src_a, src_b


def prepare_u4tsc_regression_manifests(
    *,
    u4tsc_repo: Path,
    output_root: Path,
    max_settings: int = 2,
    min_rows_per_setting: int = 50,
) -> dict[str, Any]:
    target_path = u4tsc_repo / "results" / "TargetRisk.csv"
    source_path = u4tsc_repo / "results" / "SourceRisk.csv"
    iwcv_path = u4tsc_repo / "results" / "IWCV.csv"

    target_df = pd.read_csv(target_path).rename(columns={"accuracy": "target_accuracy"})
    source_df = pd.read_csv(source_path).rename(columns={"accuracy": "source_accuracy"})
    iwcv_df = pd.read_csv(iwcv_path).rename(columns={"accuracy": "iwcv_accuracy"})

    merged = target_df.merge(source_df, on=["dataset_name", "classifier_name"], how="inner").merge(
        iwcv_df,
        on=["dataset_name", "classifier_name"],
        how="inner",
    )
    enriched = _build_features(merged)

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "real_settings_metadata.json"
    manifest_path = output_root / "real_settings.json"

    feature_columns = [
        "source_accuracy",
        "iwcv_accuracy",
        "classifier_code",
        "dataset_family_code",
        "source_domain_code",
        "target_domain_code",
        "domain_shift_indicator",
        "source_minus_iwcv",
        "source_target_gap",
        "iwcv_target_gap",
    ]
    target_column = "target_accuracy"

    settings: list[PreparedRealSetting] = []
    metadata_rows: list[dict[str, Any]] = []

    family_counts = (
        enriched.groupby("dataset_family", as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["rows", "dataset_family"], ascending=[False, True])
    )
    eligible_families = family_counts[family_counts["rows"] >= min_rows_per_setting]["dataset_family"].tolist()

    for family in eligible_families[:max_settings]:
        fam_df = enriched[enriched["dataset_family"] == family].copy()
        fam_df = fam_df.sort_values(["dataset_name", "classifier_name"]).reset_index(drop=True)
        src_a, src_b = _split_sources(fam_df)
        if src_a.empty or src_b.empty:
            continue

        family_dir = output_root / family
        family_dir.mkdir(parents=True, exist_ok=True)

        src_a_path = family_dir / "source_a.csv"
        src_b_path = family_dir / "source_b.csv"
        unlabeled_path = family_dir / "target_unlabeled.csv"
        test_path = family_dir / "target_test.csv"

        src_cols = feature_columns + [target_column]
        src_a[src_cols].to_csv(src_a_path, index=False)
        src_b[src_cols].to_csv(src_b_path, index=False)
        fam_df[feature_columns].to_csv(unlabeled_path, index=False)
        fam_df[src_cols].to_csv(test_path, index=False)

        settings.append(
            PreparedRealSetting(
                name=f"u4tsc_{family}",
                source_paths=[str(src_a_path), str(src_b_path)],
                target_unlabeled_path=str(unlabeled_path),
                target_test_path=str(test_path),
                feature_columns=feature_columns,
                target_column=target_column,
                license_status="research_only",
                schema_transform="classification_to_regression_via_accuracy_table_features",
                provenance="UDA-4-TSC results/*.csv",
            )
        )

        metadata_rows.append(
            {
                "name": f"u4tsc_{family}",
                "dataset_family": family,
                "rows": int(len(fam_df)),
                "source_a_rows": int(len(src_a)),
                "source_b_rows": int(len(src_b)),
                "license_status": "research_only",
                "schema_transform": "classification_to_regression_via_accuracy_table_features",
                "provenance": "experiments/external/UDA-4-TSC/results/*.csv",
                "paths": {
                    "source_a": str(src_a_path),
                    "source_b": str(src_b_path),
                    "target_unlabeled": str(unlabeled_path),
                    "target_test": str(test_path),
                },
            }
        )

    manifest_obj = {
        "real_settings": [s.__dict__ for s in settings],
        "count": len(settings),
    }
    metadata_obj = {
        "generator": "prepare_u4tsc_regression_manifests",
        "u4tsc_repo": str(u4tsc_repo),
        "settings": metadata_rows,
    }
    manifest_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata_obj, indent=2), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "settings_count": len(settings),
        "settings": [s.__dict__ for s in settings],
    }
