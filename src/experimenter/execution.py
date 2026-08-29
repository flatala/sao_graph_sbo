import numpy as np
import pandas as pd
import contextlib
import json
import logging
import multiprocessing as mp
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.core.algorithm import Algorithm

from adore.optimization.bridge.problem import AdoreArchOptProblem
from sb_arch_opt.algo.pymoo_interface.api import ArchOptEvaluator

from experimenter.defaults import *
from experimenter.metrics import (
    DeltaHVMetric,
    get_kernel_diagnostic_metric,
    get_kernel_weight_metric,
    get_metrics_multi,
    get_metrics_single,
    get_surrogate_metric,
)

EXPERIMENT_CONFIG_NAME = "experiment_config.json"
EXPERIMENT_CONFIG_LOCK_NAME = "experiment_config.lock"
AGGREGATE_RESULTS_NAME = "aggregate_results.csv"
AGGREGATE_RESULTS_LOCK_NAME = "aggregate_results.lock"
RUN_CONFIG_NAME = "run_config.json"
RUN_RESULTS_NAME = "run_results.csv"
RUN_LOG_NAME = "run_log.txt"
FINAL_POPULATION_NAME = "final_population.csv"
FINAL_OPTIMUM_NAME = "final_optimum.csv"

REFERENCE_PF_METADATA_NAME = "metadata.json"
REFERENCE_PF_NAME = "pf.npy"


class MetricsMaxEvalTermination(MaximumFunctionCallTermination):
    def __init__(self, n_max_evals: int, metrics):
        super().__init__(n_max_evals=n_max_evals)
        self.metrics = metrics
        self.n_eval = []

    def update(self, algorithm):
        self.n_eval.append(algorithm.evaluator.n_eval)
        for metric in self.metrics:
            metric.calculate_step(algorithm)
        return super().update(algorithm)

    def to_frame(self) -> pd.DataFrame:
        cols = {"n_eval": self.n_eval}
        for metric in self.metrics:
            for value_name in metric.value_names:
                cols[f"{metric.name}.{value_name}"] = metric.values[value_name]
        return pd.DataFrame(cols)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _run_paths(experiment_dir: Path, run_idx: int) -> dict[str, Path]:
    run_dir = experiment_dir / f"run_{run_idx:03d}"
    return {
        "run_dir": run_dir,
        "run_config": run_dir / RUN_CONFIG_NAME,
        "run_results": run_dir / RUN_RESULTS_NAME,
        "run_log": run_dir / RUN_LOG_NAME,
        "final_population": run_dir / FINAL_POPULATION_NAME,
        "final_optimum": run_dir / FINAL_OPTIMUM_NAME,
    }


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# def _write_json(path: Path, data: dict):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("w", encoding="utf-8") as fh:
#         json.dump(data, fh, indent=2, sort_keys=True)

def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(data), fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


@contextlib.contextmanager
def _log_sb_arch_opt_to_file(log_path: Path):
    logger = logging.getLogger("sb_arch_opt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s %(name)s : %(message)s"))
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        handler.close()
        logger.propagate = old_propagate


def _build_experiment_config(
    problem: AdoreArchOptProblem,
    algorithm,
    n_runs: int,
    base_seed: int,
    aggregate_results_path: Path,
    user_metadata: Optional[dict],
) -> dict:
    infill = getattr(algorithm, "infill_obj", None)
    surrogate = getattr(infill, "surrogate_model", None)
    if surrogate is not None and hasattr(surrogate, "get_config"):
        surrogate_config = _to_jsonable(surrogate.get_config())
    elif surrogate is not None:
        surrogate_config = {"class": type(surrogate).__name__}
    else:
        surrogate_config = None

    return {
        "created_at": _now_iso(),
        "runner": {
            "n_runs": int(n_runs),
            "base_seed": int(base_seed),
            "aggregate_results_path": str(aggregate_results_path.name),
            "copy_algorithm": True,
        },
        "problem": {
            "class": type(problem).__name__,
        },
        "algorithm": {
            "class": type(algorithm).__name__,
        },
        "surrogate": surrogate_config,
        "user_metadata": _to_jsonable(user_metadata or {}),
    }


def _ensure_experiment_config(
    config_path: Path,
    problem: AdoreArchOptProblem,
    algorithm,
    n_runs: int,
    base_seed: int,
    aggregate_results_path: Path,
    user_metadata: Optional[dict],
) -> dict:
    expected = _build_experiment_config(
        problem=problem,
        algorithm=algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        aggregate_results_path=aggregate_results_path,
        user_metadata=user_metadata,
    )
    lock_dir = config_path.parent / EXPERIMENT_CONFIG_LOCK_NAME
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            time.sleep(1.0)

    try:
        if not config_path.exists() or config_path.stat().st_size == 0:
            _write_json(config_path, expected)
            return expected

        existing = _read_json(config_path)
        for section in ["problem", "algorithm", "surrogate"]:
            if existing.get(section) != expected.get(section):
                raise ValueError(
                    f"Experiment config mismatch for '{section}' in {config_path}: "
                    f"{existing.get(section)!r} != {expected.get(section)!r}"
                )

        existing_metadata = dict(existing["user_metadata"])
        expected_metadata = dict(expected["user_metadata"])
        existing_metadata.pop("config_index", None)
        expected_metadata.pop("config_index", None)
        if existing_metadata != expected_metadata:
            raise ValueError(
                f"Experiment config mismatch for 'user_metadata' in {config_path}: "
                f"{existing.get('user_metadata')!r} != {expected.get('user_metadata')!r}"
            )
        if existing["user_metadata"].get("config_index") != expected["user_metadata"].get("config_index"):
            existing["user_metadata"]["config_index"] = expected["user_metadata"]["config_index"]
            _write_json(config_path, existing)

        existing_runner = existing["runner"]
        expected_runner = expected["runner"]
        for key in ["base_seed", "aggregate_results_path", "copy_algorithm"]:
            if existing_runner.get(key) != expected_runner.get(key):
                raise ValueError(
                    f"Experiment config mismatch for 'runner.{key}' in {config_path}: "
                    f"{existing_runner.get(key)!r} != {expected_runner.get(key)!r}"
                )

        if existing_runner["n_runs"] > expected_runner["n_runs"]:
            raise ValueError(
                f"Experiment config cannot reduce 'runner.n_runs' in {config_path}: "
                f"{existing_runner['n_runs']!r} > {expected_runner['n_runs']!r}"
            )
        if existing_runner["n_runs"] < expected_runner["n_runs"]:
            existing["runner"]["n_runs"] = expected_runner["n_runs"]
            _write_json(config_path, existing)
        return existing
    finally:
        lock_dir.rmdir()


def _collect_completed_run_frames(experiment_dir: Path) -> list[pd.DataFrame]:
    frames = []
    for run_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir() and p.name.startswith("run_")):
        run_config_path = run_dir / RUN_CONFIG_NAME
        run_results_path = run_dir / RUN_RESULTS_NAME
        if not run_config_path.exists():
            continue

        run_config = _read_json(run_config_path)
        if run_config.get("status") != "completed":
            continue

        df_run = pd.read_csv(run_results_path)
        if "run" not in df_run.columns:
            df_run["run"] = int(run_dir.name.split("_")[1])
        if "seed" not in df_run.columns:
            df_run["seed"] = run_config.get("seed")
        if "runtime_sec" not in df_run.columns:
            df_run["runtime_sec"] = run_config.get("runtime_sec")
        frames.append(df_run)
    return frames


def _write_aggregate_results(experiment_dir: Path, aggregate_results_path: Path) -> pd.DataFrame:
    lock_dir = experiment_dir / AGGREGATE_RESULTS_LOCK_NAME
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            time.sleep(1.0)

    try:
        frames = _collect_completed_run_frames(experiment_dir)
        if len(frames) == 0:
            aggregate_results_df = pd.DataFrame()
        else:
            aggregate_results_df = pd.concat(frames, ignore_index=True)

        aggregate_results_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_results_df.to_csv(aggregate_results_path, index=False)
        return aggregate_results_df
    finally:
        lock_dir.rmdir()


def _completed_run_matches_seed(paths: dict[str, Path], run_idx: int, run_seed: int, n_runs: int) -> bool:
    if not paths["run_config"].exists():
        return False

    run_config = _read_json(paths["run_config"])
    if run_config.get("status") != "completed":
        return False
    if run_config.get("seed") != int(run_seed):
        raise ValueError(
            f"Completed run {run_idx + 1}/{n_runs} has seed {run_config.get('seed')!r}, "
            f"expected {int(run_seed)!r}."
        )
    return True


def load_results(results_path: Union[str, Path]) -> pd.DataFrame:
    in_file = Path(results_path)
    if in_file.is_dir():
        in_file = in_file / AGGREGATE_RESULTS_NAME

    if not in_file.exists():
        raise FileNotFoundError(f"Results file does not exist: {in_file}")

    return pd.read_csv(in_file)


def aggregate_experiment_results(experiment_dir: Union[str, Path]) -> pd.DataFrame:
    experiment_dir_path = Path(experiment_dir)
    return _write_aggregate_results(experiment_dir_path, experiment_dir_path / AGGREGATE_RESULTS_NAME)


def _single_experiment_run(
    problem: AdoreArchOptProblem,
    algorithm: Algorithm,
    pareto_front,
    experiment_dir_path: Path,
    n_runs: int,
    run_idx: int,
    run_seed: int,
    termination_spec: tuple,
    minimize_kwargs: dict,
) -> None:
    paths = _run_paths(experiment_dir_path, run_idx)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)

    if _completed_run_matches_seed(paths, run_idx, run_seed, n_runs):
        print(f"Skipping completed run {run_idx + 1}/{n_runs}")
        return

    print(f"Starting run {run_idx + 1}/{n_runs}")

    # Reset state and cache
    problem.evaluator.project.architectures = []
    problem.evaluator._arch_adore_graph_map.clear()

    # Prepare metrics
    metrics = get_metrics_single(pareto_front) if problem.n_obj == 1 else get_metrics_multi(pareto_front)
    kernel_weight_metric = get_kernel_weight_metric(algorithm)
    if kernel_weight_metric is not None:
        metrics.append(kernel_weight_metric)
    kernel_diagnostic_metric = get_kernel_diagnostic_metric(algorithm)
    if kernel_diagnostic_metric is not None:
        metrics.append(kernel_diagnostic_metric)
    surrogate_metric = get_surrogate_metric(algorithm)
    if surrogate_metric is not None:
        metrics.append(surrogate_metric)
    run_termination = MetricsMaxEvalTermination(termination_spec[1], metrics)

    # Set log file paths for surrogate
    infill = getattr(algorithm, "infill_obj", None)
    if infill is not None:
        if hasattr(infill._surrogate_model_base, "set_log_path"):
            infill._surrogate_model_base.set_log_path(paths["run_log"])
        if infill._surrogate_model is not None and hasattr(infill._surrogate_model, "set_log_path"):
            infill._surrogate_model.set_log_path(paths["run_log"])

    run_meta = {
        "seed": int(run_seed),
        "status": "running",
        "runtime_sec": None,
    }
    _write_json(paths["run_config"], run_meta)
    paths["run_log"].touch(exist_ok=True)

    # Run optimisation loop
    try:
        # Ensure that logs go to the log file
        with _log_sb_arch_opt_to_file(paths["run_log"]):
            res = minimize(
                problem,
                algorithm,
                copy_algorithm=True,
                copy_termination=False,
                seed=run_seed,
                termination=run_termination,
                **minimize_kwargs,
            )

        # in case of success, write a result CSV
        df_run = run_termination.to_frame()
        df_run["run"] = run_idx
        df_run["seed"] = run_seed
        df_run["runtime_sec"] = getattr(res, "exec_time", None)
        df_run.to_csv(paths["run_results"], index=False)

        # write final population
        df_final_population = ArchOptEvaluator.get_pop_as_df(res.pop)
        df_final_population["run"] = run_idx
        df_final_population["seed"] = run_seed
        df_final_population.to_csv(paths["final_population"], index=False)

        # write final pareto front
        df_final_optimum = ArchOptEvaluator.get_pop_as_df(res.opt)
        df_final_optimum["run"] = run_idx
        df_final_optimum["seed"] = run_seed
        df_final_optimum.to_csv(paths["final_optimum"], index=False)

        # Write success metadata
        run_meta["status"] = "completed"
        run_meta["runtime_sec"] = getattr(res, "exec_time", None)
        print(f"Completed run {run_idx + 1}/{n_runs}")

    # Log error in case of failure
    except Exception as exc:
        run_meta["status"] = "failed"
        run_meta["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise

    # Write run metadata
    finally:
        _write_json(paths["run_config"], run_meta)


def _run_experiment_parallel_worker(
    problem: AdoreArchOptProblem,
    algorithm: Algorithm,
    pareto_front,
    experiment_dir_path: Path,
    n_runs: int,
    run_idx: int,
    run_seed: int,
    termination_spec: tuple,
    minimize_kwargs: dict,
) -> None:
    _single_experiment_run(
        problem=problem,
        algorithm=algorithm,
        pareto_front=pareto_front,
        experiment_dir_path=experiment_dir_path,
        n_runs=n_runs,
        run_idx=run_idx,
        run_seed=run_seed,
        termination_spec=termination_spec,
        minimize_kwargs=minimize_kwargs,
    )


def run_experiment(
    problem: AdoreArchOptProblem,
    algorithm: Algorithm,
    pareto_front,
    experiment_dir: Union[str, Path],
    n_runs: int,
    base_seed: int = 42,
    user_metadata: Optional[dict] = None,
    **minimize_kwargs,
) -> pd.DataFrame:
    minimize_kwargs.pop("copy_algorithm", None)
    termination_spec = minimize_kwargs.pop("termination", None)
    if not isinstance(termination_spec, tuple) or len(termination_spec) != 2 or termination_spec[0] not in {"n_eval", "n_evals"}:
        raise ValueError("run_experiment expects termination=('n_evals', <max_evals>)")

    # Prepare seeds for each run
    run_seeds = [base_seed + i for i in range(n_runs)]

    # Prepare dirs
    experiment_dir_path = Path(experiment_dir)
    aggregate_results_path = experiment_dir_path / AGGREGATE_RESULTS_NAME
    experiment_dir_path.mkdir(parents=True, exist_ok=True)

    # Create new or verify config matches
    _ensure_experiment_config(
        config_path=experiment_dir_path / EXPERIMENT_CONFIG_NAME,
        problem=problem,
        algorithm=algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        aggregate_results_path=aggregate_results_path,
        user_metadata=user_metadata,
    )

    # Iterate over runs
    for run_idx in range(n_runs):
        _single_experiment_run(
            problem=problem,
            algorithm=algorithm,
            pareto_front=pareto_front,
            experiment_dir_path=experiment_dir_path,
            n_runs=n_runs,
            run_idx=run_idx,
            run_seed=run_seeds[run_idx],
            termination_spec=termination_spec,
            minimize_kwargs=minimize_kwargs,
        )

    return _write_aggregate_results(experiment_dir_path, aggregate_results_path)


def single_experiment_run(
    problem: AdoreArchOptProblem,
    algorithm: Algorithm,
    pareto_front,
    experiment_dir: Union[str, Path],
    n_runs: int,
    run_id: int,
    base_seed: int = 42,
    user_metadata: Optional[dict] = None,
    **minimize_kwargs,
) -> pd.DataFrame:
    minimize_kwargs.pop("copy_algorithm", None)
    termination_spec = minimize_kwargs.pop("termination", None)
    if not isinstance(termination_spec, tuple) or len(termination_spec) != 2 or termination_spec[0] not in {"n_eval", "n_evals"}:
        raise ValueError("single_experiment_run expects termination=('n_evals', <max_evals>)")
    if run_id < 0 or run_id >= n_runs:
        raise ValueError(f"run_id must be in [0, {n_runs}), got {run_id}")

    experiment_dir_path = Path(experiment_dir)
    aggregate_results_path = experiment_dir_path / AGGREGATE_RESULTS_NAME
    experiment_dir_path.mkdir(parents=True, exist_ok=True)

    _ensure_experiment_config(
        config_path=experiment_dir_path / EXPERIMENT_CONFIG_NAME,
        problem=problem,
        algorithm=algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        aggregate_results_path=aggregate_results_path,
        user_metadata=user_metadata,
    )

    _single_experiment_run(
        problem=problem,
        algorithm=algorithm,
        pareto_front=pareto_front,
        experiment_dir_path=experiment_dir_path,
        n_runs=n_runs,
        run_idx=run_id,
        run_seed=base_seed + run_id,
        termination_spec=termination_spec,
        minimize_kwargs=minimize_kwargs,
    )

    return pd.DataFrame()


def run_experiment_parallel(
    problem: AdoreArchOptProblem,
    algorithm: Algorithm,
    pareto_front,
    experiment_dir: Union[str, Path],
    n_runs: int,
    base_seed: int = 42,
    n_workers: int = 2,
    run_ids: Optional[list[int]] = None,
    user_metadata: Optional[dict] = None,
    **minimize_kwargs,
) -> pd.DataFrame:
    minimize_kwargs.pop("copy_algorithm", None)
    termination_spec = minimize_kwargs.pop("termination", None)
    if not isinstance(termination_spec, tuple) or len(termination_spec) != 2 or termination_spec[0] not in {"n_eval", "n_evals"}:
        raise ValueError("run_experiment_parallel expects termination=('n_evals', <max_evals>)")
    if n_workers <= 0:
        raise ValueError(f"n_workers must be positive, got {n_workers}")

    run_seeds = [base_seed + i for i in range(n_runs)]

    experiment_dir_path = Path(experiment_dir)
    aggregate_results_path = experiment_dir_path / AGGREGATE_RESULTS_NAME
    experiment_dir_path.mkdir(parents=True, exist_ok=True)

    _ensure_experiment_config(
        config_path=experiment_dir_path / EXPERIMENT_CONFIG_NAME,
        problem=problem,
        algorithm=algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        aggregate_results_path=aggregate_results_path,
        user_metadata=user_metadata,
    )

    if run_ids is None:
        selected_run_ids = list(range(n_runs))
    else:
        selected_run_ids = [int(run_idx) for run_idx in run_ids]
        invalid = [run_idx for run_idx in selected_run_ids if run_idx < 0 or run_idx >= n_runs]
        if len(invalid) > 0:
            raise ValueError(f"run_ids must be in [0, {n_runs}), got {invalid}")

    pending_run_ids = []
    for run_idx in selected_run_ids:
        paths = _run_paths(experiment_dir_path, run_idx)
        if _completed_run_matches_seed(paths, run_idx, run_seeds[run_idx], n_runs):
            print(f"Skipping completed run {run_idx + 1}/{n_runs}")
        else:
            pending_run_ids.append(run_idx)

    if len(pending_run_ids) > 0:
        ctx = mp.get_context("fork")
        active: list[tuple[int, mp.Process]] = []
        queue = list(pending_run_ids)

        while len(queue) > 0 or len(active) > 0:
            while len(queue) > 0 and len(active) < n_workers:
                run_idx = queue.pop(0)
                proc = ctx.Process(
                    target=_run_experiment_parallel_worker,
                    args=(
                        problem,
                        algorithm,
                        pareto_front,
                        experiment_dir_path,
                        n_runs,
                        run_idx,
                        run_seeds[run_idx],
                        termination_spec,
                        minimize_kwargs,
                    ),
                )
                proc.start()
                active.append((run_idx, proc))

            still_active = []
            for run_idx, proc in active:
                proc.join(timeout=1.0)
                if proc.is_alive():
                    still_active.append((run_idx, proc))
                elif proc.exitcode != 0:
                    for _, other_proc in active:
                        if other_proc is not proc and other_proc.is_alive():
                            other_proc.terminate()
                    for _, other_proc in active:
                        if other_proc is not proc:
                            other_proc.join()
                    raise RuntimeError(f"Parallel experiment run {run_idx} failed with exit code {proc.exitcode}")
            active = still_active

    return _write_aggregate_results(experiment_dir_path, aggregate_results_path)


### ======================= PARETO FRONT HANDLERS ======================= ###
    
def _reference_pf_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dir": output_dir,
        "metadata": output_dir / REFERENCE_PF_METADATA_NAME,
        "pf": output_dir / REFERENCE_PF_NAME,
    }


def _build_reference_pf_metadata(problem, pf: np.ndarray, generation_method: str, generation_kwargs: dict) -> dict:
    """Get metadata for a persisted reference Pareto front."""
    pf = np.asarray(pf, dtype=float)

    metadata = {
        "created_at": _now_iso(),
        "problem": {
            "class": type(problem).__name__,
            "repr": repr(problem),
            "n_var": int(problem.n_var),
            "n_obj": int(problem.n_obj),
        },
        "generation": {
            "method": generation_method,
            "kwargs": _to_jsonable(generation_kwargs),
        },
        "reference_front": {
            "n_points": int(pf.shape[0]),
            "n_obj": int(pf.shape[1]),
            "pf_shape": list(pf.shape),
            "ideal": _to_jsonable(np.min(pf, axis=0)),
            "nadir": _to_jsonable(np.max(pf, axis=0)),
        },
    }

    if problem.n_obj > 1:
        metadata["reference_front"]["hv_true"] = float(DeltaHVMetric(pf).hv_true)
    else:
        metadata["reference_front"]["hv_true"] = None

    return metadata


def generate_and_save_reference_pareto(
    problem,
    output_dir: Union[str, Path],
    **pareto_front_kwargs,
) -> tuple[np.ndarray, dict]:
    """Generate a reference Pareto front once and persist it for reuse across machines."""

    pf = problem.calc_pareto_front(**pareto_front_kwargs)
    return save_reference_pareto(problem, output_dir, pf, "problem.calc_pareto_front", pareto_front_kwargs)


def save_reference_pareto(
    problem,
    output_dir: Union[str, Path],
    pf: np.ndarray,
    generation_method: str,
    generation_kwargs: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Persist an already available reference Pareto front for reuse across machines."""

    output_dir = Path(output_dir)
    paths = _reference_pf_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pf = np.asarray(pf, dtype=float)
    if pf.ndim == 1:
        pf = pf[:, None]

    np.save(paths["pf"], pf)
    metadata = _build_reference_pf_metadata(problem, pf, generation_method, generation_kwargs or {})
    _write_json(paths["metadata"], metadata)
    return pf, metadata


def load_reference_pareto(
    input_dir: Union[str, Path],
) -> tuple[np.ndarray, dict]:
    """Load a persisted reference Pareto front and its metadata."""

    input_dir = Path(input_dir)
    paths = _reference_pf_paths(input_dir)

    if not paths["pf"].exists():
        raise FileNotFoundError(f"Reference Pareto front file does not exist: {paths['pf']}")
    if not paths["metadata"].exists():
        raise FileNotFoundError(f"Reference Pareto metadata file does not exist: {paths['metadata']}")

    pf = np.load(paths["pf"])
    metadata = _read_json(paths["metadata"])
    return pf, metadata
