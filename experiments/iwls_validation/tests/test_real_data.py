from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from iwls_validation.real_data import RealDataConfigError, load_real_setting


def test_load_real_setting_roundtrip(tmp_path: Path) -> None:
    feature_cols = ["f1", "f2"]
    target_col = "y"

    src_path = tmp_path / "src.csv"
    unlabeled_path = tmp_path / "target_unlabeled.csv"
    test_path = tmp_path / "target_test.csv"

    pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.5, -0.1], "y": [3.0, 1.0]}).to_csv(src_path, index=False)
    pd.DataFrame({"f1": [1.5, 2.5], "f2": [0.0, 0.2]}).to_csv(unlabeled_path, index=False)
    pd.DataFrame({"f1": [1.2, 2.2], "f2": [0.3, -0.2], "y": [2.4, 1.1]}).to_csv(test_path, index=False)

    sources, target = load_real_setting(
        {
            "name": "toy",
            "source_paths": [str(src_path)],
            "target_unlabeled_path": str(unlabeled_path),
            "target_test_path": str(test_path),
            "feature_columns": feature_cols,
            "target_column": target_col,
        }
    )
    assert len(sources) == 1
    assert target.unlabeled_x.shape == (2, 2)
    assert target.test_x.shape == (2, 2)


def test_load_real_setting_missing_column_raises(tmp_path: Path) -> None:
    src_path = tmp_path / "src.csv"
    unlabeled_path = tmp_path / "target_unlabeled.csv"
    test_path = tmp_path / "target_test.csv"

    pd.DataFrame({"f1": [1.0], "f2": [2.0], "y": [3.0]}).to_csv(src_path, index=False)
    pd.DataFrame({"f1": [1.0]}).to_csv(unlabeled_path, index=False)
    pd.DataFrame({"f1": [1.0], "f2": [2.0], "y": [3.0]}).to_csv(test_path, index=False)

    with pytest.raises(RealDataConfigError):
        load_real_setting(
            {
                "name": "toy_bad",
                "source_paths": [str(src_path)],
                "target_unlabeled_path": str(unlabeled_path),
                "target_test_path": str(test_path),
                "feature_columns": ["f1", "f2"],
                "target_column": "y",
            }
        )
