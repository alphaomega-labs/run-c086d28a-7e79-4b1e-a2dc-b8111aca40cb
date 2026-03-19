# Knowledge Notes: Optimal Dataset Selection for IWLS under UDA

## Page 1: Problem Framing and Objective Mapping
The core objective is source dataset/distribution selection for minimizing **target squared-loss excess risk** without target labels.

Key formal anchors:
- `S01,S02,S03,S39`: covariate-shift correction via importance ratio `w(x)=p_t(x)/p_s(x)`.
- `S05,S06,S07`: adaptation bounds decomposing target risk into source risk + discrepancy + shared-error terms.
- `S16,S17,S18`: modern analyses clarifying when weighting is required and how regularization interacts with shift.

Cross-paper synthesis:
- If `p_t(y|x)=p_s(y|x)` is plausible and support overlap is adequate, ratio-weighted surrogates are theoretically aligned (`S01,S02,S05,S16`).
- If conditional/label shifts are present (`S36,S20`), pure covariate-shift IWLS can be misaligned; discrepancy diagnostics and robustness checks are required.

## Page 2: Equations and Diagnostics for Source Selection
Core optimization template for source candidate `k`:
- `S01,S39`: `\hat\beta_k=argmin_\beta \sum_i \hat w_k(x_i)(y_i-x_i^T\beta)^2 + \lambda||\beta||_2^2`.
- `S05`: domain-bound proxy: `\epsilon_T(h) <= \epsilon_S(h) + 0.5 d_{H\Delta H}(S,T) + \lambda^*`.
- `S04,S11`: discrepancy diagnostics via MMD.

Operational diagnostics to log per source candidate:
1. Ratio stability: second moment `E[\hat w^2]` and max weight.
2. Effective sample size (ESS): `ESS=(\sum_i \hat w_i)^2 / \sum_i \hat w_i^2`.
3. Condition number of weighted design matrix `X^T W X` (`S39`).
4. Unlabeled discrepancy (MMD/Wasserstein/IPM) as secondary proxy (`S04,S06`).

Consensus across `S01,S02,S16,S39`: high-variance weights can erase theoretical gains; clipping/regularization are not optional in practice.

## Page 3: Multi-Source Selection and Mixture Construction
Settings A/B require source-distribution selection or mixture design.

Relevant evidence:
- `S07,S25,S26,S37`: multi-source adaptation gains from source weighting/attention/mixture-of-experts.
- `S19,S21,S22,S23,S24,S37`: time-series UDA confirms method ranking is protocol-sensitive; standardization is essential.

Similarity:
- Multi-source methods consistently outperform uniform pooling when negative transfer is present (`S07,S25,S26,S37`).

Difference:
- NLP/vision multi-source methods (`S25,S26`) optimize classification losses, while target task here is regression/IWLS; objective transfer is conceptual, not direct.

Actionable adaptation:
- Replace class-loss improvement criterion with unlabeled target-risk proxy combining:
  - weighted empirical surrogate,
  - discrepancy term,
  - ESS/conditioning penalty.

## Page 4: Benchmark Assets and Dataset Suitability
Benchmark and code assets:
- `S19,S30,S31`: AdaTime and UDA-4-TSC for reproducible TS adaptation.
- `S29,S32`: WILDS paper/code for broader shift stress tests.
- `S33-S36`: dataset pool references (UCR/UCIHAR/WISDM/Sleep-EDF/HHAR).

Suitability constraints vs user requirements:
- Many TS-UDA benchmarks are classification-first; for IWLS regression, feature-schema alignment and synthetic/semi-synthetic regression target construction may be needed.
- No-target-label selection protocol is feasible via DEV/SRC-style model selection (`S19`) and discrepancy-driven selection (`S04,S06`).

## Page 5: Open Gaps and High-Value Next Retrievals
Strongly covered:
- IWLS/covariate-shift foundations (`S01,S02,S03,S05,S16,S39`).
- Multi-source/source-weighting ideas (`S07,S25,S26,S37`).
- TS-UDA benchmarking infrastructure (`S19,S21,S22,S30,S31`).

Remaining gaps:
- Regression-specific multi-source UDA benchmarks are sparse.
- Dataset licensing/redistribution details are heterogeneous and need per-candidate legal checks.
- `S40` (arXiv:2602.02066) needs a successful full-text scrape for complete extraction.
