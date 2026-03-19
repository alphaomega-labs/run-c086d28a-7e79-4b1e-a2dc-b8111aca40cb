from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from iwls_validation.real_data_prep import prepare_u4tsc_regression_manifests


def _write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_prepare_u4tsc_regression_manifests(tmp_path: Path) -> None:
    repo = tmp_path / "u4tsc"
    rows = []
    for idx in range(60):
        rows.append(
            {
                "dataset_name": f"har:{idx % 6}->{(idx + 1) % 6}",
                "classifier_name": f"clf_{idx % 4}",
                "accuracy": 0.5 + 0.001 * idx,
            }
        )
    for idx in range(60):
        rows.append(
            {
                "dataset_name": f"wisdm:{idx % 5}->{(idx + 2) % 5}",
                "classifier_name": f"clf_{idx % 4}",
                "accuracy": 0.45 + 0.001 * idx,
            }
        )

    _write_results_csv(repo / "results" / "TargetRisk.csv", rows)
    _write_results_csv(
        repo / "results" / "SourceRisk.csv",
        [{**r, "accuracy": float(cast(float, r["accuracy"])) - 0.03} for r in rows],
    )
    _write_results_csv(
        repo / "results" / "IWCV.csv",
        [{**r, "accuracy": float(cast(float, r["accuracy"])) - 0.01} for r in rows],
    )

    out = prepare_u4tsc_regression_manifests(
        u4tsc_repo=repo,
        output_root=tmp_path / "prepared",
        max_settings=2,
        min_rows_per_setting=50,
    )
    assert out["settings_count"] == 2
    assert Path(out["manifest_path"]).exists()
    assert Path(out["metadata_path"]).exists()
    for item in out["settings"]:
        assert len(item["source_paths"]) >= 1
        assert Path(item["target_unlabeled_path"]).exists()
        assert Path(item["target_test_path"]).exists()
