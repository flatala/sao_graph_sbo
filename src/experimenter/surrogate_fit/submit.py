#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a surrogate benchmark grid as one Slurm task per fit.")
    parser.add_argument("experiment_file", type=Path)
    return parser.parse_args()


def _load_file(path: Path):
    experiment_path = path.resolve()
    module_name = f"_surrogate_submit_{experiment_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, experiment_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import experiment file: {experiment_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _required(module, name: str):
    if not hasattr(module, name):
        raise SystemExit(f"Experiment file must define {name}")
    return getattr(module, name)


def _chunks(values: list[dict], chunk_size: int) -> list[list[dict]]:
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]


def main() -> int:
    args = parse_args()
    module = _load_file(args.experiment_file)

    configs = module.CONFIGS
    if len(configs) == 0:
        raise SystemExit("Experiment module has no configs.")

    cpus_per_task = int(_required(module, "SLURM_CPUS_PER_TASK"))
    mem = str(_required(module, "SLURM_MEM"))
    micromamba_env = _required(module, "MICROMAMBA_ENV")
    micromamba_module_use = _required(module, "MICROMAMBA_MODULE_USE")
    micromamba_module = _required(module, "MICROMAMBA_MODULE")
    job_name = _required(module, "SLURM_JOB_NAME")
    slurm_time = _required(module, "SLURM_TIME")
    mail_user = _required(module, "SLURM_MAIL_USER")
    mail_type = _required(module, "SLURM_MAIL_TYPE")
    partition = getattr(module, "SLURM_PARTITION", None)
    account = getattr(module, "SLURM_ACCOUNT", None)
    array_chunk_size = getattr(module, "SLURM_ARRAY_CHUNK_SIZE", None)

    if cpus_per_task <= 0:
        raise SystemExit(f"SLURM_CPUS_PER_TASK must be positive, got {cpus_per_task}")
    if mem == "":
        raise SystemExit("SLURM_MEM must not be empty")
    if len(job_name) > 10:
        raise SystemExit(f"Terrabyte recommends job names of at most 10 characters, got {job_name!r}")
    if array_chunk_size is not None:
        array_chunk_size = int(array_chunk_size)
        if array_chunk_size <= 0:
            raise SystemExit(f"SLURM_ARRAY_CHUNK_SIZE must be positive, got {array_chunk_size}")

    from experimenter.surrogate_fit.grid import config_result_dir

    results_root = Path(_required(module, "RESULTS_ROOT"))
    log_dir = Path(_required(module, "SLURM_LOG_DIR"))
    log_dir.mkdir(parents=True, exist_ok=True)

    pending_tasks = []
    for config_index, config in enumerate(configs):
        run_config_path = config_result_dir(results_root, config) / "run_config.json"
        if run_config_path.exists():
            with run_config_path.open("r", encoding="utf-8") as fh:
                run_config = json.load(fh)
            if run_config["status"] == "completed":
                continue
        pending_tasks.append({"config_index": config_index})

    if len(pending_tasks) == 0:
        print("No pending surrogate fits to submit.")
        return 0

    sbatch_script = Path(__file__).with_name("run.sbatch")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if array_chunk_size is None:
        task_chunks = [pending_tasks]
        manifest_paths = [log_dir / f"pending_surrogate_tasks_{timestamp}.json"]
    else:
        task_chunks = _chunks(pending_tasks, array_chunk_size)
        manifest_paths = [
            log_dir / f"pending_surrogate_tasks_{timestamp}_chunk_{chunk_index:03d}.json"
            for chunk_index in range(len(task_chunks))
        ]

    commands = []
    for task_chunk, manifest_path in zip(task_chunks, manifest_paths):
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(task_chunk, fh, indent=2)

        array_spec = f"0-{len(task_chunk) - 1}"
        command = [
            "sbatch",
            "-D",
            str(REPO_ROOT),
            "-J",
            job_name,
            "-o",
            str(log_dir / "%x_%A_%a.out"),
            "-e",
            str(log_dir / "%x_%A_%a.err"),
            "--export=NONE",
            f"--array={array_spec}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={mem}",
            f"--mail-user={mail_user}",
            f"--mail-type={mail_type}",
            f"--time={slurm_time}",
        ]
        if partition:
            command.append(f"--partition={partition}")
        if account:
            command.append(f"--account={account}")
        command.extend(
            [
                str(sbatch_script),
                str(args.experiment_file),
                str(manifest_path),
                str(micromamba_env),
                str(micromamba_module_use),
                str(micromamba_module),
            ]
        )
        commands.append((array_spec, manifest_path, command))

    print("Submitting surrogate benchmark grid:")
    print(f"  configs: {len(configs)}")
    print(f"  pending_tasks: {len(pending_tasks)}")
    print(f"  array_chunk_size: {array_chunk_size}")
    print(f"  arrays: {len(commands)}")
    print(f"  cpus_per_task: {cpus_per_task}")
    print(f"  mem: {mem}")
    print(f"  results_root: {results_root}")
    print(f"  log_dir: {log_dir}")
    print("  commands:")
    for array_spec, manifest_path, command in commands:
        print(f"  manifest: {manifest_path}")
        print(f"  array: {array_spec}")
        print("  " + " ".join(command))

    for _, _, command in commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
