# IWLS Validation Experiments

This experiment package validates hypotheses H1-H3 for stability-aware source selection under covariate shift.

## Goals
- Reproduce synthetic A/B/C setting comparisons with no-target-label selection.
- Benchmark stability-aware ranking against discrepancy-based, pooled-IWLS, and random baselines.
- Run pooled-focused ablations over stabilization controls (alpha/beta/gamma, clipping, ridge, ratio estimator, candidate pool size).
- Generate paper-ready PDF figures, CSV tables, and SymPy validation reports.

## Setup
- Python environment: `experiments/.venv`
- Install dependencies:
  - `uv pip install --python experiments/.venv/bin/python numpy pandas matplotlib seaborn scipy sympy pytest ruff mypy pymupdf`

## Run
- Prepare real-setting manifests from cloned UDA-4-TSC result tables:
  - `PYTHONPATH=experiments/iwls_validation/src experiments/.venv/bin/python experiments/iwls_validation/prepare_real_manifests.py --u4tsc-repo experiments/external/UDA-4-TSC --output-root experiments/iwls_validation/iter_2/real_data --manifest-output experiments/iwls_validation/iter_2/output/real_manifest_build.json`
- Generate run config that merges defaults with prepared `real_settings`:
  - `PYTHONPATH=experiments/iwls_validation/src experiments/.venv/bin/python experiments/iwls_validation/build_iter_config.py --base-config experiments/iwls_validation/configs/default.json --prepared-manifest experiments/iwls_validation/iter_2/real_data/real_settings.json --output-config experiments/iwls_validation/configs/iter_2.json`
- `PYTHONPATH=experiments/iwls_validation/src experiments/.venv/bin/python experiments/iwls_validation/run_experiments.py --config experiments/iwls_validation/configs/default.json --output-dir experiments/iwls_validation/output --paper-fig-dir paper/figures --paper-table-dir paper/tables --paper-data-dir paper/data`
- Iteration-safe run (used for iteration 1):
  - `PYTHONPATH=experiments/iwls_validation/src experiments/.venv/bin/python experiments/iwls_validation/run_experiments.py --config experiments/iwls_validation/configs/default.json --output-dir experiments/iwls_validation/iter_1/output --paper-fig-dir paper/figures/iter_1 --paper-table-dir paper/tables/iter_1 --paper-data-dir paper/data/iter_1`
  - Iteration-safe run with real-track manifests (iteration 2):
  - `PYTHONPATH=experiments/iwls_validation/src experiments/.venv/bin/python experiments/iwls_validation/run_experiments.py --config experiments/iwls_validation/configs/iter_2.json --output-dir experiments/iwls_validation/iter_2/output --paper-fig-dir paper/figures/iter_2 --paper-table-dir paper/tables/iter_2 --paper-data-dir paper/data/iter_2`

## Outputs
- `experiments/iwls_validation/output/results_summary.json`
- `experiments/iwls_validation/output/sympy_validation_report.txt`
- `paper/figures/fig_main_results.pdf`
- `paper/figures/fig_stability_tradeoff.pdf`
- `paper/figures/fig_ablation_pooled_gap.pdf`
- `paper/figures/fig_real_setting_performance.pdf`
- `paper/figures/fig_artifact_quality_checks.pdf`
- `paper/tables/table_main_metrics.csv`
- `paper/tables/table_significance.csv`
- `paper/tables/table_ablation_vs_pooled.csv`
- `paper/tables/table_regime_stratified.csv`
- `paper/tables/table_governance_audit.csv`
- `paper/data/iwls_results.csv`

The run summary (`results_summary.json`) records artifact paths in dedicated keys (`figures`, `tables`, `datasets`, `sympy_report`) and includes figure QA metrics such as `plot_legend_overlap_ratio`.

## Real-data hooks
- `real_settings` in config accepts locally prepared CSVs converted from AdaTime/UDA-4-TSC-style tasks.
- Required keys per real setting: `name`, `source_paths`, `target_unlabeled_path`, `target_test_path`, `feature_columns`, `target_column`.
- Missing files or schema mismatch are reported in `results_summary.json` under `real_data_warnings`.

## Statistical method
Significance uses Shapiro-Wilk normality check, then paired t-test if normal otherwise paired Wilcoxon; Holm correction is applied across method comparisons and across pooled-focused ablation configurations.

## Safety checks
- Config schema is validated strictly and rejects wrapped recovery envelopes (`payload/artifacts/notes/...`) to prevent upstream malformed inputs from silently executing.
