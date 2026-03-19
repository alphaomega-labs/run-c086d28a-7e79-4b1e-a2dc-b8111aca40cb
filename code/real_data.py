from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from iwls_validation.core import SourceDataset, TargetDataset


class RealDataConfigError(ValueError):
    pass


def _require_columns(df: pd.DataFrame, columns: list[str], file_path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise RealDataConfigError(f"Missing columns {missing} in {file_path}")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RealDataConfigError(f"Missing required CSV file: {path}")
    return pd.read_csv(path)


def load_real_setting(setting_cfg: dict) -> Tuple[List[SourceDataset], TargetDataset]:
    required = {
        "name",
        "source_paths",
        "target_unlabeled_path",
        "target_test_path",
        "feature_columns",
        "target_column",
    }
    missing_keys = sorted(required - set(setting_cfg.keys()))
    if missing_keys:
        raise RealDataConfigError(f"Real setting missing required keys: {missing_keys}")

    feature_columns = [str(c) for c in setting_cfg["feature_columns"]]
    target_column = str(setting_cfg["target_column"])

    source_paths = [Path(p) for p in setting_cfg["source_paths"]]
    sources: List[SourceDataset] = []
    for idx, src_path in enumerate(source_paths):
        sdf = _load_csv(src_path)
        _require_columns(sdf, feature_columns + [target_column], src_path)
        sx = sdf[feature_columns].to_numpy(dtype=float)
        sy = sdf[target_column].to_numpy(dtype=float)
        sources.append(SourceDataset(name=f"{setting_cfg['name']}_source_{idx:02d}", x=sx, y=sy))

    unlabeled_path = Path(setting_cfg["target_unlabeled_path"])
    udf = _load_csv(unlabeled_path)
    _require_columns(udf, feature_columns, unlabeled_path)

    test_path = Path(setting_cfg["target_test_path"])
    tdf = _load_csv(test_path)
    _require_columns(tdf, feature_columns + [target_column], test_path)

    target = TargetDataset(
        unlabeled_x=udf[feature_columns].to_numpy(dtype=float),
        test_x=tdf[feature_columns].to_numpy(dtype=float),
        test_y=tdf[target_column].to_numpy(dtype=float),
    )
    return sources, target
