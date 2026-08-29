#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OPENMDAO_REQUIRE_MPI", "false")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"pycycle\..*")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one task from a surrogate benchmark manifest.")
    parser.add_argument("--experiment-file", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_file(path: Path):
    experiment_path = path.resolve()
    module_name = f"_surrogate_grid_{experiment_path.stem}"
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
    builder = module.BENCHMARK_BUILDER
    configs = module.CONFIGS
    results_root = Path(module.RESULTS_ROOT)

    with args.task_manifest.open("r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise SystemExit(f"--task-index must be in [0, {len(tasks)}), got {args.task_index}")

    config_index = int(tasks[args.task_index]["config_index"])
    config = configs[config_index]

    from experimenter.surrogate_fit.benchmark import run_surrogate_fit
    from experimenter.surrogate_fit.grid import config_result_dir

    print(f"[{config_index:04d}] {module.format_config(config)}", flush=True)
    print(f"      -> {config_result_dir(results_root, config)}", flush=True)
    run_surrogate_fit(builder, config, results_root, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
