#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import importlib.util
import json
import multiprocessing as mp
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
    parser = argparse.ArgumentParser(description="Run one Slurm task from an experiment grid manifest.")
    parser.add_argument(
        "--experiment-file",
        type=Path,
        required=True,
        help="Path to a Python experiment definition file.",
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        required=True,
        help="Path to the Slurm task manifest produced by submit_experiment_grid.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        required=True,
        help="Index into the Slurm task manifest.",
    )
    return parser.parse_args()


def _load_file(path: Path):
    experiment_path = path.resolve()
    module_name = f"_experiment_grid_{experiment_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, experiment_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import experiment file: {experiment_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _result_path(results_root: Path, config) -> Path:
    return results_root / config.problem / config.name


def _print_config(index: int, config, results_root: Path, module) -> None:
    print(f"[{index:03d}] {module.format_config(config)}")
    print(f"      -> {_result_path(results_root, config)}")


def _task_group(task: dict) -> list[dict]:
    if "tasks" not in task:
        return [task]
    tasks = task["tasks"]
    if not isinstance(tasks, list) or len(tasks) == 0:
        raise SystemExit("Grouped manifest entry must contain a non-empty 'tasks' list.")
    return tasks


def _run_task(task: dict, module, builder, configs, results_root: Path, experiment_file: Path) -> None:
    config_index = int(task["config_index"])
    run_id = int(task["run_id"])

    if config_index < 0 or config_index >= len(configs):
        raise SystemExit(f"config index must be in [0, {len(configs)}), got {config_index}")

    config = configs[config_index]
    experiment_dir = _result_path(results_root, config)
    _print_config(config_index, config, results_root, module)
    print(f"      run_id={run_id}")

    from experimenter.execution import load_reference_pareto, single_experiment_run

    results_root.mkdir(parents=True, exist_ok=True)
    pareto_front, _ = load_reference_pareto(config.reference_dir)
    problem = builder.build_problem(config)
    # Matches the per-run seed single_experiment_run uses (base_seed + run_id); lets
    # seed-aware builders load the seed-matched encoder / seed the head per run.
    run_seed = int(module.BASE_SEED) + run_id
    algorithm = builder.build_algorithm(problem, config, run_seed=run_seed)
    experiment_id = str(experiment_file)
    user_metadata = builder.build_user_metadata(config, experiment_id, config_index)
    if config.problem == "mdgnc_edge_failures":
        from experimenter.problems.mdgnc_encoding import get_adsg_encoding_metadata

        user_metadata["encoding"] = get_adsg_encoding_metadata(problem)

    single_experiment_run(
        problem=problem,
        algorithm=algorithm,
        pareto_front=pareto_front,
        n_runs=module.N_RUNS,
        run_id=run_id,
        base_seed=module.BASE_SEED,
        experiment_dir=experiment_dir,
        termination=("n_evals", config.max_evals),
        verbose=True,
        save_history=False,
        user_metadata=user_metadata,
    )
    print(f"DONE {experiment_dir}")


def _run_task_from_file(task: dict, experiment_file: str) -> None:
    module = _load_file(Path(experiment_file))
    _run_task(
        task,
        module,
        module.EXPERIMENT_BUILDER,
        module.CONFIGS,
        Path(module.RESULTS_ROOT),
        Path(experiment_file),
    )


def _cleanup_after_task() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    module = _load_file(args.experiment_file)
    builder = module.EXPERIMENT_BUILDER
    configs = module.CONFIGS
    results_root = Path(module.RESULTS_ROOT)

    with args.task_manifest.open("r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise SystemExit(f"--task-index must be in [0, {len(tasks)}), got {args.task_index}")

    task_group = _task_group(tasks[args.task_index])
    task_parallelism = int(getattr(module, "SLURM_TASK_PARALLELISM", 1))
    if task_parallelism <= 0:
        raise SystemExit(f"SLURM_TASK_PARALLELISM must be positive, got {task_parallelism}")
    print(f"Slurm task {args.task_index}: {len(task_group)} run(s)", flush=True)

    if task_parallelism == 1 or len(task_group) == 1:
        for local_index, task in enumerate(task_group, start=1):
            print(f"Task item {local_index}/{len(task_group)}", flush=True)
            _run_task(task, module, builder, configs, results_root, args.experiment_file)
            _cleanup_after_task()
        return 0

    max_workers = min(task_parallelism, len(task_group))
    print(f"Running {len(task_group)} task item(s) with parallelism={max_workers}", flush=True)
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(_run_task_from_file, task, str(args.experiment_file))
            for task in task_group
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
        _cleanup_after_task()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
