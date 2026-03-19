from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SourceDataset:
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class TargetDataset:
    unlabeled_x: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_setting(
    setting: str,
    seed: int,
    n_sources: int = 8,
    n_samples: int = 1200,
    n_features: int = 8,
) -> Tuple[List[SourceDataset], TargetDataset]:
    rng = make_rng(seed)
    beta_true = rng.normal(0.0, 1.0, size=n_features)

    if setting == "A":
        shift_scale = 0.8
        noise_scale = 0.25
    elif setting == "B":
        shift_scale = 1.2
        noise_scale = 0.35
    else:
        shift_scale = 1.5
        noise_scale = 0.45

    target_mean = rng.normal(0.0, shift_scale, size=n_features)
    target_cov_diag = rng.uniform(0.8, 1.4, size=n_features)

    sources: List[SourceDataset] = []
    for idx in range(n_sources):
        mean_shift = rng.normal(0.0, shift_scale * 1.3, size=n_features)
        cov_diag = rng.uniform(0.6, 1.8, size=n_features)
        x = rng.normal(mean_shift, np.sqrt(cov_diag), size=(n_samples, n_features))
        y = x @ beta_true + rng.normal(0.0, noise_scale * (1.0 + 0.1 * idx), size=n_samples)

        # Inject harmful sources for mixed-shift diagnostics.
        if idx >= n_sources - 2:
            y = y + 0.8 * np.sin(x[:, 0]) + rng.normal(0.0, 0.4, size=n_samples)

        sources.append(SourceDataset(name=f"source_{idx:02d}", x=x, y=y))

    unlabeled_x = rng.normal(target_mean, np.sqrt(target_cov_diag), size=(n_samples, n_features))
    test_x = rng.normal(target_mean, np.sqrt(target_cov_diag), size=(n_samples, n_features))
    test_y = test_x @ beta_true + rng.normal(0.0, noise_scale, size=n_samples)

    target = TargetDataset(unlabeled_x=unlabeled_x, test_x=test_x, test_y=test_y)
    return sources, target


def gaussian_logpdf_diag(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    safe_var = np.maximum(var, 1e-6)
    z = ((x - mean) ** 2) / safe_var
    log_norm = np.sum(np.log(2.0 * np.pi * safe_var))
    return -0.5 * (np.sum(z, axis=1) + log_norm)


def apply_weight_clipping(weights: np.ndarray, rule: str) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if rule == "none":
        clipped = w
    elif rule == "p99":
        cap = float(np.percentile(w, 99))
        clipped = np.minimum(w, cap)
    elif rule == "p95":
        cap = float(np.percentile(w, 95))
        clipped = np.minimum(w, cap)
    elif rule == "cap_10":
        clipped = np.minimum(w, 10.0)
    else:
        raise ValueError(f"Unsupported clipping rule: {rule}")
    return clipped / (clipped.mean() + 1e-12)


def logistic_ratio_proxy(source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    # Lightweight logistic domain classifier to obtain p_t(x)/p_s(x) odds proxy.
    x = np.vstack([source_x, target_x])
    y = np.concatenate([np.zeros(source_x.shape[0]), np.ones(target_x.shape[0])])
    x_mu = x.mean(axis=0)
    x_std = x.std(axis=0) + 1e-6
    xn = (x - x_mu) / x_std

    w = np.zeros(xn.shape[1], dtype=float)
    b = 0.0
    lr = 0.08
    reg = 1e-3
    for _ in range(90):
        logits = xn @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        err = probs - y
        grad_w = (xn.T @ err) / len(y) + reg * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b

    srcn = (source_x - x_mu) / x_std
    logits_src = srcn @ w + b
    p_t = 1.0 / (1.0 + np.exp(-np.clip(logits_src, -20.0, 20.0)))
    p_s = 1.0 - p_t
    odds = p_t / (p_s + 1e-8)
    return odds


def estimate_density_ratio(
    source_x: np.ndarray,
    target_x: np.ndarray,
    ratio_estimator: str = "gaussian_diag_proxy",
    clipping: str = "none",
) -> Tuple[np.ndarray, float]:
    if ratio_estimator == "gaussian_diag_proxy":
        src_mean = source_x.mean(axis=0)
        src_var = source_x.var(axis=0) + 1e-6
        tgt_mean = target_x.mean(axis=0)
        tgt_var = target_x.var(axis=0) + 1e-6

        log_tgt = gaussian_logpdf_diag(source_x, tgt_mean, tgt_var)
        log_src = gaussian_logpdf_diag(source_x, src_mean, src_var)
        log_ratio = np.clip(log_tgt - log_src, -5.0, 5.0)
        weights = np.exp(log_ratio)
        uncertainty = float(np.var(log_ratio))
    elif ratio_estimator == "logistic_ratio_proxy":
        weights = logistic_ratio_proxy(source_x, target_x)
        uncertainty = float(np.var(np.log(np.clip(weights, 1e-8, None))))
    else:
        raise ValueError(f"Unsupported ratio estimator: {ratio_estimator}")

    weights = np.clip(weights, 1e-8, None)
    weights = apply_weight_clipping(weights, clipping)
    return weights, uncertainty


def ridge_wls_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray, lam: float) -> np.ndarray:
    n_features = x.shape[1]
    w = np.clip(weights, 1e-6, None)
    wx = x * w[:, None]
    xtwx = x.T @ wx
    reg = lam * np.eye(n_features)
    beta = np.linalg.solve(xtwx + reg, x.T @ (w * y))
    return beta


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def ess(weights: np.ndarray) -> float:
    num = float(np.sum(weights) ** 2)
    den = float(np.sum(weights**2) + 1e-12)
    return num / den


def mmd_mean_distance(source_x: np.ndarray, target_x: np.ndarray) -> float:
    return float(np.linalg.norm(source_x.mean(axis=0) - target_x.mean(axis=0), ord=2))


def wasserstein_diag_approx(source_x: np.ndarray, target_x: np.ndarray) -> float:
    mean_term = np.linalg.norm(source_x.mean(axis=0) - target_x.mean(axis=0), ord=2)
    var_term = np.linalg.norm(np.sqrt(source_x.var(axis=0) + 1e-6) - np.sqrt(target_x.var(axis=0) + 1e-6), ord=2)
    return float(mean_term + var_term)


def stability_penalty(x: np.ndarray, weights: np.ndarray, lam: float) -> Tuple[float, float, float]:
    w = np.clip(weights, 1e-6, None)
    weighted_x = x * np.sqrt(w[:, None])
    gram = weighted_x.T @ weighted_x + lam * np.eye(x.shape[1])
    cond = float(np.linalg.cond(gram))
    e = ess(w)
    second_moment = float(np.mean(w**2))
    penalty = float((1.0 / max(e, 1e-6)) + np.log1p(cond) + second_moment)
    return penalty, e, cond


def calibrated_gamma(
    gamma: float,
    eff_n: float,
    cond: float,
    adaptive_gamma: bool,
    ess_floor: float,
    cond_ref: float,
    gamma_ess_scale: float,
    gamma_cond_scale: float,
) -> float:
    if not adaptive_gamma:
        return float(gamma)

    ess_multiplier = max(0.0, (ess_floor / max(eff_n, 1e-6)) - 1.0)
    cond_multiplier = max(0.0, np.log1p(cond / max(cond_ref, 1e-6)))
    factor = 1.0 + gamma_ess_scale * ess_multiplier + gamma_cond_scale * cond_multiplier
    return float(gamma * factor)


def source_score(
    source: SourceDataset,
    target_unlabeled_x: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    lam: float,
    ratio_estimator: str,
    clipping: str,
    adaptive_gamma: bool,
    ess_floor: float,
    cond_ref: float,
    gamma_ess_scale: float,
    gamma_cond_scale: float,
) -> Dict[str, float]:
    weights, uncertainty = estimate_density_ratio(
        source.x,
        target_unlabeled_x,
        ratio_estimator=ratio_estimator,
        clipping=clipping,
    )
    coeff = ridge_wls_fit(source.x, source.y, weights, lam)
    pred = source.x @ coeff
    proxy_risk = mse(source.y, pred)
    discrepancy = mmd_mean_distance(source.x, target_unlabeled_x)
    stab_penalty, eff_n, cond = stability_penalty(source.x, weights, lam)
    gamma_eff = calibrated_gamma(
        gamma,
        eff_n,
        cond,
        adaptive_gamma,
        ess_floor,
        cond_ref,
        gamma_ess_scale,
        gamma_cond_scale,
    )

    score = proxy_risk + alpha * discrepancy + beta * uncertainty + gamma_eff * stab_penalty
    return {
        "score": float(score),
        "proxy_risk": float(proxy_risk),
        "discrepancy": float(discrepancy),
        "uncertainty": float(uncertainty),
        "stability_penalty": float(stab_penalty),
        "ess": float(eff_n),
        "condition_number": float(cond),
        "gamma_effective": float(gamma_eff),
    }


def gate_score_components(
    source: SourceDataset,
    target_unlabeled_x: np.ndarray,
    ratio_estimator: str,
    clipping: str,
) -> dict[str, float]:
    disc = mmd_mean_distance(source.x, target_unlabeled_x)
    wass = wasserstein_diag_approx(source.x, target_unlabeled_x)
    w, uncertainty = estimate_density_ratio(
        source.x,
        target_unlabeled_x,
        ratio_estimator=ratio_estimator,
        clipping=clipping,
    )
    tail = float(np.percentile(w, 95))
    gate = 0.4 * disc + 0.3 * abs(disc - wass) + 0.3 * tail
    return {
        "gate_score": float(gate),
        "discrepancy_component": float(disc),
        "ratio_uncertainty_component": float(uncertainty),
        "tail_component": float(tail),
    }


def evaluate_single_source(
    source: SourceDataset,
    target: TargetDataset,
    lam: float,
    use_weights: bool,
    ratio_estimator: str,
    clipping: str,
) -> Dict[str, float]:
    if use_weights:
        weights, uncertainty = estimate_density_ratio(
            source.x,
            target.unlabeled_x,
            ratio_estimator=ratio_estimator,
            clipping=clipping,
        )
    else:
        weights = np.ones(source.x.shape[0], dtype=float)
        uncertainty = 0.0
    coeff = ridge_wls_fit(source.x, source.y, weights, lam)
    test_pred = target.test_x @ coeff
    target_mse = mse(target.test_y, test_pred)
    _, eff_n, cond = stability_penalty(source.x, weights, lam)
    return {
        "target_mse": target_mse,
        "ess": eff_n,
        "condition_number": cond,
        "ratio_uncertainty": float(uncertainty),
        "weight_second_moment": float(np.mean(weights**2)),
    }


def evaluate_baselines(
    sources: List[SourceDataset],
    target: TargetDataset,
    seed: int,
    alpha: float,
    beta: float,
    gamma: float,
    lam: float,
    ratio_estimator: str = "gaussian_diag_proxy",
    clipping: str = "none",
    adaptive_gamma: bool = False,
    ess_floor: float = 100.0,
    cond_ref: float = 100.0,
    gamma_ess_scale: float = 0.75,
    gamma_cond_scale: float = 0.25,
) -> Dict[str, Dict[str, float]]:
    rng = make_rng(seed)
    scores = {
        s.name: source_score(
            s,
            target.unlabeled_x,
            alpha,
            beta,
            gamma,
            lam,
            ratio_estimator,
            clipping,
            adaptive_gamma,
            ess_floor,
            cond_ref,
            gamma_ess_scale,
            gamma_cond_scale,
        )
        for s in sources
    }

    by_mmd = min(sources, key=lambda s: mmd_mean_distance(s.x, target.unlabeled_x))
    by_wass = min(sources, key=lambda s: wasserstein_diag_approx(s.x, target.unlabeled_x))
    by_score = min(sources, key=lambda s: scores[s.name]["score"])
    random_source = sources[int(rng.integers(0, len(sources)))]

    oracle = min(
        sources,
        key=lambda s: evaluate_single_source(s, target, lam, True, ratio_estimator, clipping)["target_mse"],
    )

    pooled_x = np.vstack([s.x for s in sources])
    pooled_y = np.concatenate([s.y for s in sources])
    pooled = SourceDataset(name="pooled", x=pooled_x, y=pooled_y)

    # Mixed-shift gate for H3: keep lower-risk diagnostics.
    gate_scores = []
    for src in sources:
        disc = mmd_mean_distance(src.x, target.unlabeled_x)
        wass = wasserstein_diag_approx(src.x, target.unlabeled_x)
        w, _ = estimate_density_ratio(
            src.x,
            target.unlabeled_x,
            ratio_estimator=ratio_estimator,
            clipping=clipping,
        )
        tail = float(np.percentile(w, 95))
        gate = 0.4 * disc + 0.3 * abs(disc - wass) + 0.3 * tail
        gate_scores.append((gate, src))
    gate_scores.sort(key=lambda t: t[0])
    kept = [s for _, s in gate_scores[: max(3, len(sources) // 2)]]
    gated_best = min(kept, key=lambda s: scores[s.name]["score"])

    results: Dict[str, Dict[str, float]] = {
        "stability_aware_composite": evaluate_single_source(by_score, target, lam, True, ratio_estimator, clipping),
        "random_source_plus_IWLS": evaluate_single_source(random_source, target, lam, True, ratio_estimator, clipping),
        "mmd_nearest_source_plus_IWLS": evaluate_single_source(by_mmd, target, lam, True, ratio_estimator, clipping),
        "wasserstein_nearest_source_plus_IWLS": evaluate_single_source(by_wass, target, lam, True, ratio_estimator, clipping),
        "pooled_source_IWLS": evaluate_single_source(pooled, target, lam, True, ratio_estimator, clipping),
        "single_source_unweighted_LS": evaluate_single_source(by_mmd, target, lam, False, ratio_estimator, clipping),
        "mixed_shift_gate_plus_composite": evaluate_single_source(gated_best, target, lam, True, ratio_estimator, clipping),
        "oracle_best_source_retrospective": evaluate_single_source(oracle, target, lam, True, ratio_estimator, clipping),
    }

    oracle_mse = results["oracle_best_source_retrospective"]["target_mse"]
    gamma_effective = float(scores[by_score.name]["gamma_effective"])
    for metric in results.values():
        metric["target_excess_risk_vs_oracle"] = float(metric["target_mse"] - oracle_mse)
        metric["regret_gap_to_oracle_selected_source"] = metric["target_excess_risk_vs_oracle"]
        metric["gamma_effective"] = gamma_effective

    return results


def _binary_curve_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sort by descending risk score for threshold sweeping.
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]

    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        x = np.array([0.0, 1.0], dtype=float)
        yv = np.array([0.0, 1.0], dtype=float)
        t = np.array([np.inf, -np.inf], dtype=float)
        return x, yv, t

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    recall = tps / pos
    fpr = fps / neg
    thresholds = s
    return fpr.astype(float), recall.astype(float), thresholds.astype(float)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = _binary_curve_points(scores, labels)
    x = np.concatenate(([0.0], fpr, [1.0]))
    y = np.concatenate(([0.0], tpr, [1.0]))
    return float(np.trapezoid(y, x))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    _, recall, thresholds = _binary_curve_points(scores, labels)
    order = np.argsort(-scores)
    y = labels[order]
    tps = np.cumsum(y == 1).astype(float)
    fps = np.cumsum(y == 0).astype(float)
    precision = tps / np.maximum(tps + fps, 1.0)
    x = np.concatenate(([0.0], recall, [1.0]))
    yv = np.concatenate(([1.0], precision, [float(y.mean())]))
    return float(np.trapezoid(yv, x))


def compute_gate_diagnostics(
    *,
    settings: list[str],
    seeds: list[int],
    n_sources: int,
    n_samples: int,
    alpha: float,
    beta: float,
    gamma: float,
    lam: float,
    ratio_estimator: str,
    clipping: str,
    adaptive_gamma: bool,
    ess_floor: float,
    cond_ref: float,
    gamma_ess_scale: float,
    gamma_cond_scale: float,
    tau_quantiles: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9),
) -> dict[str, list[dict[str, Any]]]:
    source_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    roc_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []

    for setting in settings:
        for seed in seeds:
            sources, target = generate_setting(setting=setting, seed=seed, n_sources=n_sources, n_samples=n_samples)
            per_source: list[dict[str, Any]] = []
            for idx, src in enumerate(sources):
                comp = gate_score_components(src, target.unlabeled_x, ratio_estimator=ratio_estimator, clipping=clipping)
                c1_score = source_score(
                    src,
                    target.unlabeled_x,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    lam=lam,
                    ratio_estimator=ratio_estimator,
                    clipping=clipping,
                    adaptive_gamma=adaptive_gamma,
                    ess_floor=ess_floor,
                    cond_ref=cond_ref,
                    gamma_ess_scale=gamma_ess_scale,
                    gamma_cond_scale=gamma_cond_scale,
                )
                harmful = 1 if idx >= n_sources - 2 else 0
                row = {
                    "setting": setting,
                    "seed": seed,
                    "source": src.name,
                    "source_index": idx,
                    "harmful_label": harmful,
                    **comp,
                    "c1_score": float(c1_score["score"]),
                }
                per_source.append(row)
                source_rows.append(row)

            scores = np.array([float(r["gate_score"]) for r in per_source], dtype=float)
            labels = np.array([int(r["harmful_label"]) for r in per_source], dtype=int)
            fpr, tpr, thresholds = _binary_curve_points(scores, labels)

            for f, t, thr in zip(fpr, tpr, thresholds):
                roc_rows.append(
                    {
                        "setting": setting,
                        "seed": seed,
                        "fpr": float(f),
                        "tpr": float(t),
                        "threshold": float(thr),
                    }
                )

            order = np.argsort(-scores)
            y = labels[order]
            tps = np.cumsum(y == 1).astype(float)
            fps = np.cumsum(y == 0).astype(float)
            precision = tps / np.maximum(tps + fps, 1.0)
            recall = tps / max(float(y.sum()), 1.0)
            for p, r, thr in zip(precision, recall, scores[order]):
                pr_rows.append(
                    {
                        "setting": setting,
                        "seed": seed,
                        "precision": float(p),
                        "recall": float(r),
                        "threshold": float(thr),
                    }
                )

            quality_rows.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "harmful_source_auroc": _auroc(scores, labels),
                    "harmful_source_auprc": _auprc(scores, labels),
                }
            )

            safe_mask = labels == 0
            harmful_mask = labels == 1
            safe_scores = scores[safe_mask]
            harmful_scores = scores[harmful_mask]
            g_safe = float(np.max(safe_scores)) if safe_scores.size else float("nan")
            g_harm = float(np.min(harmful_scores)) if harmful_scores.size else float("nan")
            feasible = bool(np.isfinite(g_safe) and np.isfinite(g_harm) and g_safe < g_harm)
            threshold_rows.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "g_safe": g_safe,
                    "g_harm": g_harm,
                    "threshold_feasible": 1.0 if feasible else 0.0,
                    "feasibility_gap": float(g_harm - g_safe) if np.isfinite(g_safe) and np.isfinite(g_harm) else float("nan"),
                }
            )

            for q in tau_quantiles:
                tau = float(np.quantile(scores, q))
                kept_idxs = [i for i, v in enumerate(scores) if float(v) <= tau]
                if not kept_idxs:
                    continue
                kept_sources = [sources[i] for i in kept_idxs]
                kept_rows = [per_source[i] for i in kept_idxs]
                safe_retention = float(np.mean([1.0 if r["harmful_label"] == 0 else 0.0 for r in kept_rows])) if kept_rows else 0.0
                n_safe_total = float(np.sum(labels == 0))
                n_safe_rejected = float(
                    np.sum(
                        [
                            1.0
                            if (i not in kept_idxs and int(r["harmful_label"]) == 0)
                            else 0.0
                            for i, r in enumerate(per_source)
                        ]
                    )
                )
                false_reject_safe = n_safe_rejected / max(n_safe_total, 1.0)

                best = min(
                    kept_sources,
                    key=lambda s: source_score(
                        s,
                        target.unlabeled_x,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        lam=lam,
                        ratio_estimator=ratio_estimator,
                        clipping=clipping,
                        adaptive_gamma=adaptive_gamma,
                        ess_floor=ess_floor,
                        cond_ref=cond_ref,
                        gamma_ess_scale=gamma_ess_scale,
                        gamma_cond_scale=gamma_cond_scale,
                    )["score"],
                )
                mse_gate = evaluate_single_source(best, target, lam, True, ratio_estimator, clipping)["target_mse"]
                source_mses = [
                    evaluate_single_source(s, target, lam, True, ratio_estimator, clipping)["target_mse"]
                    for s in sources
                ]
                source_mses_sorted = np.sort(np.array(source_mses, dtype=float))
                worst_decile = float(np.mean(source_mses_sorted[max(0, int(0.9 * len(source_mses_sorted))):]))
                best_oracle = float(np.min(source_mses_sorted))

                retention_rows.append(
                    {
                        "setting": setting,
                        "seed": seed,
                        "tau_quantile": float(q),
                        "tau_value": tau,
                        "retained_count": int(len(kept_idxs)),
                        "safe_source_retention_rate": safe_retention,
                        "false_rejection_rate_safe_sources": false_reject_safe,
                        "target_mse_after_gating": float(mse_gate),
                        "worst_decile_target_mse_after_gating": worst_decile,
                        "selection_regret_after_gating": float(mse_gate - best_oracle),
                    }
                )

    return {
        "source_rows": source_rows,
        "quality_rows": quality_rows,
        "retention_rows": retention_rows,
        "threshold_rows": threshold_rows,
        "roc_rows": roc_rows,
        "pr_rows": pr_rows,
    }
