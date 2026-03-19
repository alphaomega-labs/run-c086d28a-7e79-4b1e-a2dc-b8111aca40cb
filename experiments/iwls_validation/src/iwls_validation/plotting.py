from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox


def apply_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")


def _method_label(method: str) -> str:
    mapping = {
        "stability_aware_composite": "Stable-C1",
        "random_source_plus_IWLS": "Random+IWLS",
        "mmd_nearest_source_plus_IWLS": "MMD+IWLS",
        "wasserstein_nearest_source_plus_IWLS": "Wass+IWLS",
        "pooled_source_IWLS": "Pooled+IWLS",
        "single_source_unweighted_LS": "Unweighted LS",
        "mixed_shift_gate_plus_composite": "Gate+C1",
        "oracle_best_source_retrospective": "Oracle (retro)",
    }
    return mapping.get(method, method)


def _bbox_area(bbox: Bbox) -> float:
    return max(float(bbox.width), 0.0) * max(float(bbox.height), 0.0)


def _legend_overlap_ratio(fig: plt.Figure, ax: plt.Axes, legend: Any | None) -> float:
    if legend is None:
        return 0.0
    fig.canvas.draw()
    canvas = fig.canvas
    renderer = canvas.get_renderer() if hasattr(canvas, "get_renderer") else None
    if renderer is None:
        return 0.0
    legend_bbox = legend.get_window_extent(renderer=renderer)
    axes_bbox = ax.get_window_extent(renderer=renderer)
    inter = Bbox.intersection(legend_bbox, axes_bbox)
    if inter is None:
        return 0.0
    denom = _bbox_area(axes_bbox)
    if denom <= 0.0:
        return 0.0
    return _bbox_area(inter) / denom


def plot_multi_panel_results(summary_df: pd.DataFrame, out_pdf: Path) -> dict[str, float]:
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.2, wspace=0.28)

    panel_settings = ["A", "B", "C"]
    overlap_values: list[float] = []
    legend_handle: Line2D | None = None
    for idx, setting in enumerate(panel_settings):
        ax = axes[idx]
        sdf = summary_df[summary_df["setting"] == setting].copy()
        sdf = sdf.sort_values("target_mse_mean", ascending=True)
        if sdf.empty:
            continue
        sdf["method_short"] = sdf["method"].map(_method_label)
        x = range(len(sdf))
        y = sdf["target_mse_mean"]
        yerr_low = y - sdf["target_mse_ci_low"]
        yerr_high = sdf["target_mse_ci_high"] - y

        err = ax.errorbar(
            x,
            y,
            yerr=[yerr_low, yerr_high],
            fmt="o",
            capsize=4,
            color="#1f77b4",
            label="Mean ± 95% CI",
        )
        legend_handle = err.lines[0]
        ax.set_xticks(list(x))
        ax.set_xticklabels(sdf["method_short"], rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Target MSE (squared error units)")
        ax.set_xlabel("Method")
        ax.set_title(f"Setting {setting}")
        legend = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=10)
        overlap_values.append(_legend_overlap_ratio(fig, ax, legend))
    if legend_handle is not None:
        fig.legend(
            [legend_handle],
            ["Mean ± 95% CI"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            frameon=True,
            ncol=1,
            fontsize=11,
        )

    fig.suptitle(
        "Caption: Cross-setting comparison of IWLS source selection methods with 95% CI across five seeds",
        fontsize=12,
    )
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    max_overlap = max(overlap_values) if overlap_values else 0.0
    return {"legend_overlap_ratio": float(max_overlap)}


def plot_stability_tradeoff(results_df: pd.DataFrame, out_pdf: Path) -> dict[str, float]:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.07, right=0.79, top=0.84, bottom=0.17, wspace=0.28)
    methods = sorted(results_df["method"].unique().tolist())
    settings = sorted(results_df["setting"].unique().tolist())
    palette = sns.color_palette("colorblind", n_colors=len(methods))
    color_map = {m: palette[i] for i, m in enumerate(methods)}
    marker_cycle = ["o", "X", "s", "^", "D", "v"]
    marker_map = {s: marker_cycle[i % len(marker_cycle)] for i, s in enumerate(settings)}
    overlap_values: list[float] = []

    ax0 = axes[0]
    sns.scatterplot(
        data=results_df,
        x="ess",
        y="target_mse",
        hue="method",
        style="setting",
        palette=color_map,
        markers=marker_map,
        ax=ax0,
        legend=False,
    )
    ax0.set_xlabel("Effective sample size (ESS)")
    ax0.set_ylabel("Target MSE (squared error units)")
    ax0.set_title("ESS vs Target Error")

    ax1 = axes[1]
    sns.scatterplot(
        data=results_df,
        x="condition_number",
        y="target_mse",
        hue="method",
        style="setting",
        palette=color_map,
        markers=marker_map,
        ax=ax1,
        legend=False,
    )
    ax1.set_xlabel("Condition number κ(XᵀWX)")
    ax1.set_ylabel("Target MSE (squared error units)")
    ax1.set_title("Conditioning vs Target Error")

    method_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=color_map[m], color=color_map[m], label=_method_label(m))
        for m in methods
    ]
    setting_handles = [
        Line2D([0], [0], marker=marker_map[s], linestyle="", markersize=7, color="#333333", label=f"Setting {s}")
        for s in settings
    ]
    lg_methods = fig.legend(
        handles=method_handles,
        title="Method",
        loc="upper left",
        bbox_to_anchor=(0.805, 0.86),
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    lg_settings = fig.legend(
        handles=setting_handles,
        title="Setting",
        loc="upper left",
        bbox_to_anchor=(0.805, 0.33),
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    overlap_values.append(_legend_overlap_ratio(fig, ax0, lg_methods))
    overlap_values.append(_legend_overlap_ratio(fig, ax0, lg_settings))
    overlap_values.append(_legend_overlap_ratio(fig, ax1, lg_methods))
    overlap_values.append(_legend_overlap_ratio(fig, ax1, lg_settings))

    fig.suptitle(
        "Caption: Stability diagnostics versus target performance for all settings and baselines",
        fontsize=12,
    )
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    max_overlap = max(overlap_values) if overlap_values else 0.0
    return {"legend_overlap_ratio": float(max_overlap)}


def plot_ablation_vs_pooled(ablation_df: pd.DataFrame, out_pdf: Path) -> dict[str, float]:
    if ablation_df.empty:
        return {"legend_overlap_ratio": 0.0}

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(left=0.07, right=0.79, top=0.84, bottom=0.16, wspace=0.32)
    overlap_values: list[float] = []

    ax0 = axes[0]
    sns.scatterplot(
        data=ablation_df,
        x="pooled_gap_mean",
        y="holm_adjusted_p",
        hue="ratio_estimator",
        style="clipping",
        ax=ax0,
    )
    ax0.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax0.axhline(0.05, color="black", linestyle=":", linewidth=1.0)
    ax0.set_xlabel("Mean MSE gap: stability - pooled (lower is better)")
    ax0.set_ylabel("Holm-adjusted p-value")
    ax0.set_title("Ablation significance vs pooled IWLS")
    lg0 = ax0.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=True, fontsize=10)
    overlap_values.append(_legend_overlap_ratio(fig, ax0, lg0))

    ax1 = axes[1]
    top = ablation_df.nsmallest(8, columns=["holm_adjusted_p", "pooled_gap_mean"]).copy()
    top["label"] = top["config_id"].astype(str)
    sns.barplot(data=top, x="pooled_gap_mean", y="label", hue="adaptive_gamma", ax=ax1)
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Mean MSE gap: stability - pooled")
    ax1.set_ylabel("Top ablation configurations (config_id)")
    ax1.set_title("Best pooled-gap configurations")
    lg1 = ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=True, fontsize=10)
    overlap_values.append(_legend_overlap_ratio(fig, ax1, lg1))

    fig.suptitle("Caption: Pooled-IWLS-focused ablation over stability calibration controls", fontsize=12)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    max_overlap = max(overlap_values) if overlap_values else 0.0
    return {"legend_overlap_ratio": float(max_overlap)}


def plot_real_setting_performance(summary_df: pd.DataFrame, out_pdf: Path) -> dict[str, float]:
    apply_theme()
    real_df = summary_df[summary_df["setting"].astype(str).str.startswith("R_")].copy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.subplots_adjust(left=0.1, right=0.78, top=0.86, bottom=0.2)
    if real_df.empty:
        ax.plot([0, 1], [0, 0], label="No real settings loaded", color="#1f77b4", linewidth=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Placeholder axis")
        ax.set_ylabel("Target MSE (squared error units)")
        legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
        overlap = _legend_overlap_ratio(fig, ax, legend)
        fig.suptitle("Caption: Real-setting performance panel (no real settings available)", fontsize=12)
        fig.savefig(out_pdf, format="pdf")
        plt.close(fig)
        return {"legend_overlap_ratio": float(overlap)}

    keep_methods = [
        "stability_aware_composite",
        "pooled_source_IWLS",
        "mmd_nearest_source_plus_IWLS",
        "wasserstein_nearest_source_plus_IWLS",
    ]
    real_df = real_df[real_df["method"].isin(keep_methods)].copy()
    real_df["method_short"] = real_df["method"].map(_method_label)

    sns.barplot(
        data=real_df,
        x="setting",
        y="target_mse_mean",
        hue="method_short",
        ax=ax,
        errorbar=None,
    )
    ax.set_xlabel("Real setting")
    ax.set_ylabel("Target MSE (squared error units)")
    ax.set_title("Real-track method comparison")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=10)
    overlap = _legend_overlap_ratio(fig, ax, legend)
    fig.suptitle(
        "Caption: Setting C proxy comparison on converted UDA-4-TSC real-track manifests",
        fontsize=12,
    )
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    return {"legend_overlap_ratio": float(overlap)}


def plot_governance_quality(
    governance_df: pd.DataFrame,
    pdf_checks: list[dict[str, Any]],
    legend_overlap_ratio: float,
    out_pdf: Path,
) -> dict[str, float]:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.85, bottom=0.19, wspace=0.3)
    overlap_values: list[float] = []

    ax0 = axes[0]
    if governance_df.empty:
        governance_df = pd.DataFrame(
            {
                "setting": ["none"],
                "loaded": [0.0],
                "license_ok": [0.0],
                "schema_ok": [0.0],
            }
        )
    g_long = governance_df.melt(
        id_vars=["setting"],
        value_vars=["loaded", "license_ok", "schema_ok"],
        var_name="check",
        value_name="rate",
    )
    sns.barplot(data=g_long, x="setting", y="rate", hue="check", ax=ax0)
    ax0.set_ylim(0.0, 1.05)
    ax0.set_xlabel("Setting")
    ax0.set_ylabel("Compliance rate")
    ax0.set_title("Governance and schema checks")
    lg0 = ax0.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=10)
    overlap_values.append(_legend_overlap_ratio(fig, ax0, lg0))

    ax1 = axes[1]
    readable_rate = float(np.mean([1.0 if c.get("readable", False) else 0.0 for c in pdf_checks])) if pdf_checks else 0.0
    qa_df = pd.DataFrame(
        {
            "metric": ["pdf_readability_rate", "legend_clearance"],
            "value": [readable_rate, max(0.0, 1.0 - legend_overlap_ratio)],
        }
    )
    palette = sns.color_palette("colorblind", n_colors=len(qa_df))
    sns.barplot(data=qa_df, x="metric", y="value", hue="metric", dodge=False, ax=ax1, legend=False, palette=palette)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("Artifact-quality metric")
    ax1.set_ylabel("Score")
    ax1.set_title("Figure quality checks")
    qa_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=8, color=palette[idx], label=str(row["metric"]))
        for idx, row in qa_df.reset_index(drop=True).iterrows()
    ]
    lg1 = ax1.legend(handles=qa_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=10)
    overlap_values.append(_legend_overlap_ratio(fig, ax1, lg1))

    fig.suptitle("Caption: Governance compliance and PDF readability/legend-quality audit", fontsize=12)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    max_overlap = max(overlap_values) if overlap_values else 0.0
    return {"legend_overlap_ratio": float(max_overlap)}


def plot_expC_roc_pr(
    roc_df: pd.DataFrame,
    pr_df: pd.DataFrame,
    out_pdf: Path,
) -> dict[str, float]:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.16, wspace=0.25)
    overlap_values: list[float] = []

    ax0 = axes[0]
    if roc_df.empty:
        roc_df = pd.DataFrame({"setting": ["C"], "fpr": [0.0], "tpr": [0.0]})
    sns.lineplot(data=roc_df, x="fpr", y="tpr", hue="setting", estimator="mean", errorbar=None, ax=ax0)
    ax0.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, label="Chance")
    ax0.set_xlabel("False positive rate")
    ax0.set_ylabel("True positive rate")
    ax0.set_title("exp_C ROC")
    lg0 = ax0.legend(loc="lower right", frameon=True, fontsize=10)
    overlap_values.append(_legend_overlap_ratio(fig, ax0, lg0))

    ax1 = axes[1]
    if pr_df.empty:
        pr_df = pd.DataFrame({"setting": ["C"], "recall": [0.0], "precision": [1.0]})
    sns.lineplot(data=pr_df, x="recall", y="precision", hue="setting", estimator="mean", errorbar=None, ax=ax1)
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("exp_C Precision-Recall")
    lg1 = ax1.legend(loc="lower left", frameon=True, fontsize=10)
    overlap_values.append(_legend_overlap_ratio(fig, ax1, lg1))

    fig.suptitle("Caption: exp_C harmful-source gate diagnostics with ROC and PR curves", fontsize=12)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    return {"legend_overlap_ratio": float(max(overlap_values) if overlap_values else 0.0)}


def plot_expC_retention_error(
    retention_df: pd.DataFrame,
    out_pdf: Path,
) -> dict[str, float]:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.16, wspace=0.28)
    overlap_values: list[float] = []

    ax0 = axes[0]
    if retention_df.empty:
        retention_df = pd.DataFrame(
            {
                "tau_quantile": [0.5],
                "safe_source_retention_rate": [0.0],
                "setting": ["C"],
            }
        )
    sns.lineplot(
        data=retention_df,
        x="tau_quantile",
        y="safe_source_retention_rate",
        hue="setting",
        marker="o",
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax0,
    )
    ax0.set_xlabel("Gate threshold quantile τ")
    ax0.set_ylabel("Safe-source retention rate")
    ax0.set_title("exp_C retention curve")
    lg0 = ax0.legend(loc="best", frameon=True, fontsize=10)
    overlap_values.append(_legend_overlap_ratio(fig, ax0, lg0))

    ax1 = axes[1]
    sns.lineplot(
        data=retention_df,
        x="tau_quantile",
        y="target_mse_after_gating",
        hue="setting",
        marker="o",
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax1,
    )
    ax1.set_xlabel("Gate threshold quantile τ")
    ax1.set_ylabel("Target MSE after gating (squared error units)")
    ax1.set_title("exp_C retention vs downstream error")
    lg1 = ax1.legend(loc="best", frameon=True, fontsize=10)
    overlap_values.append(_legend_overlap_ratio(fig, ax1, lg1))

    fig.suptitle("Caption: exp_C retention-vs-error diagnostics with uncertainty bands", fontsize=12)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    return {"legend_overlap_ratio": float(max(overlap_values) if overlap_values else 0.0)}


def plot_expC_threshold_feasibility(
    threshold_df: pd.DataFrame,
    out_pdf: Path,
) -> dict[str, float]:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.16, wspace=0.30)
    overlap_values: list[float] = []

    ax0 = axes[0]
    if threshold_df.empty:
        threshold_df = pd.DataFrame({"setting": ["C"], "feasibility_gap": [0.0], "threshold_feasible": [0.0]})
    sns.boxplot(data=threshold_df, x="setting", y="feasibility_gap", ax=ax0)
    ax0.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax0.set_xlabel("Setting")
    ax0.set_ylabel("Feasibility gap g_harm - g_safe")
    ax0.set_title("exp_C threshold-feasibility boundary")

    ax1 = axes[1]
    feasibility = threshold_df.groupby("setting", as_index=False)["threshold_feasible"].mean()
    sns.barplot(data=feasibility, x="setting", y="threshold_feasible", hue="setting", dodge=False, ax=ax1, legend=False)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("Setting")
    ax1.set_ylabel("Threshold feasibility rate")
    ax1.set_title("exp_C threshold-feasibility rate")
    overlap_values.append(0.0)

    fig.suptitle("Caption: exp_C threshold-feasibility and boundary diagnostics", fontsize=12)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    return {"legend_overlap_ratio": float(max(overlap_values) if overlap_values else 0.0)}
