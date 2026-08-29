#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed runs for an experiment grid.")
    parser.add_argument(
        "experiment_file",
        type=Path,
        help="Path to a Python experiment definition file.",
    )
    return parser.parse_args()


def _load_file(path: Path):
    experiment_path = path.resolve()
    module_name = f"_experiment_aggregate_{experiment_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, experiment_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import experiment file: {experiment_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    module = _load_file(args.experiment_file)

    from experimenter.execution import aggregate_experiment_results

    results_root = Path(module.RESULTS_ROOT)
    for config_index, config in enumerate(module.CONFIGS):
        experiment_dir = results_root / config.problem / config.name
        df = aggregate_experiment_results(experiment_dir)
        print(f"[{config_index:03d}] {experiment_dir} -> {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
