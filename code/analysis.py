from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy import stats


def to_frame(records: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(records))


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    sem = stats.sem(arr)
    q = float(stats.t.ppf((1.0 + confidence) / 2.0, len(arr) - 1))
    half = q * sem
    return mean, mean - half, mean + half


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (setting, method), gdf in df.groupby(["setting", "method"], sort=True):
        mse_mean, mse_lo, mse_hi = mean_ci(gdf["target_mse"].to_numpy())
        rows.append(
            {
                "setting": setting,
                "method": method,
                "target_mse_mean": mse_mean,
                "target_mse_ci_low": mse_lo,
                "target_mse_ci_high": mse_hi,
                "target_mse_std": float(gdf["target_mse"].std(ddof=1) if len(gdf) > 1 else 0.0),
                "ess_mean": float(gdf["ess"].mean()),
                "ess_std": float(gdf["ess"].std(ddof=1) if len(gdf) > 1 else 0.0),
                "excess_risk_mean": float(gdf["target_excess_risk_vs_oracle"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["setting", "target_mse_mean", "method"], ascending=[True, True, True]).reset_index(drop=True)


def paired_significance(df: pd.DataFrame, baseline: str, competitor: str) -> Dict[str, float | str]:
    merged = df[df["method"].isin([baseline, competitor])].pivot_table(
        index=["setting", "seed"], columns="method", values="target_mse", aggfunc="first"
    )
    merged = merged.dropna()
    if merged.empty:
        return {"test": "none", "p_value": 1.0, "n_pairs": 0}
    diff = merged[baseline] - merged[competitor]

    shapiro_p = stats.shapiro(diff).pvalue if len(diff) >= 3 else 0.0
    if shapiro_p >= 0.05:
        p = float(stats.ttest_rel(merged[baseline], merged[competitor], alternative="greater").pvalue)
        test_name = "paired_t_test"
    else:
        p = float(stats.wilcoxon(merged[baseline], merged[competitor], alternative="greater").pvalue)
        test_name = "paired_wilcoxon"

    return {"test": test_name, "p_value": p, "n_pairs": int(len(diff))}


def holm_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(p_values, key=lambda k: p_values[k])
    m = len(keys)
    adjusted: Dict[str, float] = {}
    for rank, key in enumerate(keys, start=1):
        adjusted[key] = min(1.0, p_values[key] * (m - rank + 1))
    return adjusted
