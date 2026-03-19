from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build iteration config by merging default config with prepared real settings.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    prepared = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    base["real_settings"] = prepared.get("real_settings", [])
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(base, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
