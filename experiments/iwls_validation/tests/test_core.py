import numpy as np

from iwls_validation.core import apply_weight_clipping, ess, evaluate_baselines, generate_setting


def test_ess_bounds() -> None:
    w = np.array([1.0, 2.0, 3.0])
    val = ess(w)
    assert 1.0 <= val <= len(w)


def test_weight_clipping_normalizes_mean() -> None:
    w = np.array([1.0, 2.0, 20.0, 50.0])
    clipped = apply_weight_clipping(w, "p95")
    assert np.isclose(clipped.mean(), 1.0)
    assert np.all(clipped > 0.0)


def test_baseline_outputs_have_required_metrics() -> None:
    sources, target = generate_setting("A", 11)
    out = evaluate_baselines(
        sources,
        target,
        seed=11,
        alpha=0.3,
        beta=0.5,
        gamma=0.5,
        lam=1e-3,
        ratio_estimator="gaussian_diag_proxy",
        clipping="p99",
        adaptive_gamma=True,
    )
    required = {
        "target_mse",
        "ess",
        "condition_number",
        "ratio_uncertainty",
        "weight_second_moment",
        "target_excess_risk_vs_oracle",
        "gamma_effective",
    }
    assert "stability_aware_composite" in out
    for metrics in out.values():
        assert required.issubset(metrics.keys())
