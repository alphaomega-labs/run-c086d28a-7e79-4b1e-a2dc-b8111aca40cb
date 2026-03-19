from __future__ import annotations

from pathlib import Path

import pandas as pd

from iwls_validation.plotting import (
    plot_ablation_vs_pooled,
    plot_governance_quality,
    plot_multi_panel_results,
    plot_real_setting_performance,
    plot_stability_tradeoff,
)


def test_plotting_returns_overlap_metrics(tmp_path: Path) -> None:
    summary_df = pd.DataFrame(
        {
            "setting": ["A", "A", "B", "B", "C", "C", "R_u4tsc_har", "R_u4tsc_har"],
            "method": [
                "stability_aware_composite",
                "pooled_source_IWLS",
                "stability_aware_composite",
                "pooled_source_IWLS",
                "stability_aware_composite",
                "pooled_source_IWLS",
                "stability_aware_composite",
                "pooled_source_IWLS",
            ],
            "target_mse_mean": [0.12, 0.15, 0.22, 0.25, 0.35, 0.41, 0.29, 0.33],
            "target_mse_ci_low": [0.1, 0.12, 0.2, 0.22, 0.3, 0.36, 0.26, 0.3],
            "target_mse_ci_high": [0.14, 0.18, 0.25, 0.28, 0.4, 0.46, 0.32, 0.36],
        }
    )
    results_df = pd.DataFrame(
        {
            "setting": ["A", "A", "B", "B"],
            "method": [
                "stability_aware_composite",
                "pooled_source_IWLS",
                "stability_aware_composite",
                "pooled_source_IWLS",
            ],
            "ess": [200, 180, 140, 130],
            "condition_number": [20, 24, 40, 45],
            "target_mse": [0.12, 0.16, 0.23, 0.28],
        }
    )
    ablation_df = pd.DataFrame(
        {
            "config_id": ["cfg_0001", "cfg_0002"],
            "pooled_gap_mean": [-0.01, 0.02],
            "holm_adjusted_p": [0.08, 0.2],
            "ratio_estimator": ["gaussian_diag_proxy", "logistic_ratio_proxy"],
            "clipping": ["p99", "p95"],
            "adaptive_gamma": [True, False],
        }
    )

    fig1 = tmp_path / "main.pdf"
    fig2 = tmp_path / "stability.pdf"
    fig3 = tmp_path / "ablation.pdf"
    fig4 = tmp_path / "real.pdf"
    fig5 = tmp_path / "qa.pdf"

    m1 = plot_multi_panel_results(summary_df, fig1)
    m2 = plot_stability_tradeoff(results_df, fig2)
    m3 = plot_ablation_vs_pooled(ablation_df, fig3)
    m4 = plot_real_setting_performance(summary_df, fig4)
    governance_df = pd.DataFrame(
        {
            "setting": ["R_u4tsc_har"],
            "loaded": [1.0],
            "license_ok": [1.0],
            "schema_ok": [1.0],
        }
    )
    m5 = plot_governance_quality(
        governance_df=governance_df,
        pdf_checks=[{"path": str(fig1), "readable": True}],
        legend_overlap_ratio=0.02,
        out_pdf=fig5,
    )

    assert fig1.exists() and fig2.exists() and fig3.exists() and fig4.exists() and fig5.exists()
    assert "legend_overlap_ratio" in m1
    assert "legend_overlap_ratio" in m2
    assert "legend_overlap_ratio" in m3
    assert "legend_overlap_ratio" in m4
    assert "legend_overlap_ratio" in m5
