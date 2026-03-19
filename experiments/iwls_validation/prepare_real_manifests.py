from __future__ import annotations

import argparse
import json
from pathlib import Path

from iwls_validation.real_data_prep import prepare_u4tsc_regression_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real-setting manifests from UDA-4-TSC results tables.")
    parser.add_argument("--u4tsc-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-settings", type=int, default=2)
    parser.add_argument("--min-rows-per-setting", type=int, default=50)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prep = prepare_u4tsc_regression_manifests(
        u4tsc_repo=args.u4tsc_repo,
        output_root=args.output_root,
        max_settings=args.max_settings,
        min_rows_per_setting=args.min_rows_per_setting,
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(prep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
