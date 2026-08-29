#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GPU-local chunk from a GNN encoder manifest.")
    parser.add_argument("--experiment-file", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_file(path: Path):
    experiment_path = path.resolve()
    module_name = f"_gnn_encoder_grid_{experiment_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, experiment_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import experiment file: {experiment_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _visible_gpus() -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return []
    return [g for g in cvd.split(",") if g != ""]


def _run_one(payload):
    config_index, config, resources_root, data_root, results_root, overwrite, gpu = payload
    # Pin this worker to its assigned GPU BEFORE torch is imported (the benchmark import
    # below pulls in torch), so spawned workers spread across the node's GPUs.
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from experimenter.encoder_training.benchmark import run_encoder_fit
    from experimenter.encoder_training.grid import config_result_dir, format_config

    print(f"[{config_index:04d}] gpu={gpu} {format_config(config)}", flush=True)
    print(f"      -> {config_result_dir(results_root, config)}", flush=True)
    run_encoder_fit(config, resources_root, data_root, results_root, overwrite=overwrite)
    return config_index


def main() -> int:
    args = parse_args()
    module = _load_file(args.experiment_file)
    configs = module.CONFIGS
    resources_root = Path(module.RESOURCES_ROOT)
    data_root = Path(module.DATA_ROOT)
    results_root = Path(module.RESULTS_ROOT)
    parallel_per_gpu = int(module.PARALLEL_CONFIGS_PER_GPU)

    with args.task_manifest.open("r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise SystemExit(f"--task-index must be in [0, {len(tasks)}), got {args.task_index}")

    config_indices = [int(index) for index in tasks[args.task_index]["config_indices"]]
    # Spread workers across every GPU allocated to the task (round-robin by config), so a
    # whole-node gpu:4 job actually uses all 4 GPUs instead of piling onto cuda:0.
    gpus = _visible_gpus()
    n_gpus = max(1, len(gpus))
    max_workers = parallel_per_gpu * n_gpus
    print(
        f"GPU task {args.task_index}: {len(config_indices)} configs, "
        f"parallel_per_gpu={parallel_per_gpu}, gpus={gpus or '[cpu]'}, total_workers={max_workers}",
        flush=True,
    )

    payloads = [
        (
            index,
            configs[index],
            resources_root,
            data_root,
            results_root,
            args.overwrite,
            gpus[slot % len(gpus)] if gpus else None,
        )
        for slot, index in enumerate(config_indices)
    ]
    if max_workers <= 1 or len(payloads) <= 1:
        for payload in payloads:
            _run_one(payload)
        return 0

    import multiprocessing as mp

    # spawn: each worker is a fresh process, so the per-worker CUDA_VISIBLE_DEVICES set in
    # _run_one takes effect at torch init (and avoids CUDA-with-fork issues).
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_run_one, payload) for payload in payloads]
        for future in as_completed(futures):
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
