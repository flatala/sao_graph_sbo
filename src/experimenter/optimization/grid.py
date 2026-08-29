from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from experimenter.optimization.models import (
    build_bo,
    build_bo_encoded_krg,
    build_bo_ignore_hierarchy,
    build_ga,
    build_random,
    build_surrogate_bo,
)


@dataclass(frozen=True)
class ExperimentConfig:
    problem: str
    method: str
    params: dict[str, Any]
    problem_config: dict[str, Any]
    reference_dir: Path
    max_evals: int
    name: str

    @property
    def algorithm_kind(self) -> str:
        return str(self.params["algorithm_kind"])

    @property
    def n_init(self) -> int | None:
        return self.params.get("n_init")

    @property
    def batch_size(self) -> int | None:
        return self.params.get("batch_size")

    @property
    def kpls_n_dim(self) -> int | None:
        return self.params.get("kpls_n_dim")

    @property
    def surrogate_name(self) -> str | None:
        return self.params.get("surrogate_name")

    @property
    def surrogate_params(self) -> dict[str, Any]:
        return self.params.get("surrogate_params", {})


MethodBuilder = Callable[[Any, ExperimentConfig], Any]
SurrogateBuilder = Callable[[Any, ExperimentConfig], tuple[Any, Any]]


class ExperimentBuilder:
    def __init__(self, resources_root: Path, evaluator_parallelism: int):
        self.resources_root = resources_root
        self.evaluator_parallelism = evaluator_parallelism
        self.algorithm_builders = {
            "bo_kpls": build_bo,
            "bo_no_kpls": build_bo,
            "bo_ignore_hierarchy": build_bo_ignore_hierarchy,
            "bo_encoded_krg": build_bo_encoded_krg,
            "ga": build_ga,
            "random": build_random,
            "surrogate_bo": self._build_surrogate_bo,
        }
        self.surrogate_builders: dict[str, SurrogateBuilder] = {}

    def register_algorithm(self, algorithm_kind: str, builder: MethodBuilder) -> None:
        self.algorithm_builders[algorithm_kind] = builder

    def register_surrogate(self, surrogate_name: str, builder: SurrogateBuilder) -> None:
        self.surrogate_builders[surrogate_name] = builder

    def format_config(self, config: ExperimentConfig) -> str:
        return format_config(config)

    def build_problem(self, config: ExperimentConfig):
        evaluator_parallelism = int(config.problem_config.get("evaluator_parallelism", self.evaluator_parallelism))
        return build_adore_problem(config, self.resources_root, evaluator_parallelism)

    def build_algorithm(self, problem, config: ExperimentConfig, run_seed: int | None = None):
        if config.algorithm_kind not in self.algorithm_builders:
            raise ValueError(f"Unknown algorithm kind: {config.algorithm_kind!r}")
        builder = self.algorithm_builders[config.algorithm_kind]
        # Seed-aware builders (e.g. the seed-matched GNN encoder) opt in by declaring a
        # run_seed parameter; legacy builders keep the (problem, config) signature.
        if "run_seed" in inspect.signature(builder).parameters:
            return builder(problem, config, run_seed=run_seed)
        return builder(problem, config)

    def build_user_metadata(
        self,
        config: ExperimentConfig,
        experiment_module: str,
        config_index: int,
    ) -> dict[str, Any]:
        evaluator_parallelism = int(config.problem_config.get("evaluator_parallelism", self.evaluator_parallelism))
        return build_user_metadata(config, experiment_module, config_index, evaluator_parallelism)

    def _build_surrogate_bo(self, problem, config: ExperimentConfig):
        if config.surrogate_name not in self.surrogate_builders:
            raise ValueError(f"Unknown surrogate: {config.surrogate_name!r}")
        surrogate_model, normalization = self.surrogate_builders[config.surrogate_name](problem, config)
        return build_surrogate_bo(problem, config, surrogate_model, normalization)


def format_config(config: ExperimentConfig) -> str:
    details = [
        f"problem={config.problem}",
        f"method={config.method}",
        f"algorithm={config.algorithm_kind}",
        f"max_evals={config.max_evals}",
    ]
    if config.n_init is not None:
        details.append(f"n_init={config.n_init}")
    if config.batch_size is not None:
        details.append(f"batch_size={config.batch_size}")
    if config.algorithm_kind in {"bo_kpls", "bo_no_kpls"}:
        details.append(f"kpls_n_dim={config.kpls_n_dim}")
    if config.algorithm_kind == "bo_ignore_hierarchy":
        details.append("ignore_hierarchy=True")
    if config.algorithm_kind == "bo_encoded_krg":
        details.append("encoded_vector=True")
    if config.surrogate_name is not None:
        details.append(f"surrogate={config.surrogate_name}")
        details.append(f"surrogate_params={config.surrogate_params}")
    return ", ".join(details)


def make_experiment_configs(
    problem_configs: Mapping[str, Mapping[str, Any]],
    method_grids: Mapping[str, list[Mapping[str, Any]]],
) -> tuple[ExperimentConfig, ...]:
    configs: list[ExperimentConfig] = []
    for problem, raw_problem_config in problem_configs.items():
        problem_config = dict(raw_problem_config)
        reference_dir = Path(problem_config["reference_dir"])
        max_evals = int(problem_config["max_evals"])

        for method, grid in method_grids.items():
            for raw_params in grid:
                selected_problems = raw_params.get("problems")
                if selected_problems is not None and problem not in selected_problems:
                    continue
                params = _resolve_method_params(problem_config, raw_params)
                name = _default_experiment_name(problem, method, params, max_evals)
                configs.append(
                    ExperimentConfig(
                        problem=problem,
                        method=method,
                        params=params,
                        problem_config=problem_config,
                        reference_dir=reference_dir,
                        max_evals=max_evals,
                        name=name,
                    )
                )
    return tuple(configs)


def _resolve_method_params(problem_config: dict[str, Any], raw_params: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(raw_params)
    params.pop("problems", None)
    if params.pop("use_bo_settings", False):
        params["n_init"] = problem_config["bo_n_init"]
        params["batch_size"] = problem_config["bo_batch_size"]
    return params


def _default_experiment_name(problem: str, method: str, params: dict[str, Any], max_evals: int) -> str:
    if params["algorithm_kind"] in {"bo_kpls", "bo_no_kpls", "bo_ignore_hierarchy", "bo_encoded_krg"}:
        return (
            f"{method}_{problem}"
            f"_init_{params['n_init']}"
            f"_infill_{params['batch_size']}"
            f"_{max_evals}_evals"
        )
    if params["algorithm_kind"] == "surrogate_bo":
        surrogate_name = params["surrogate_name"]
        return (
            f"{surrogate_name}_{problem}"
            f"_init_{params['n_init']}"
            f"_infill_{params['batch_size']}"
            f"_{max_evals}_evals"
        )
    if params.get("n_init") is not None:
        return (
            f"{method}_{problem}"
            f"_init_{params['n_init']}"
            f"_pop_{params['batch_size']}"
            f"_{max_evals}_evals"
        )
    return f"{method}_{problem}_pop_{params['batch_size']}_{max_evals}_evals"


def build_adore_problem(config: ExperimentConfig, resources_root: Path, evaluator_parallelism: int = 1):
    from experimenter.problems.factory import build_problem

    return build_problem(config.problem, resources_root, evaluator_parallelism)


def build_user_metadata(
    config: ExperimentConfig,
    experiment_module: str,
    config_index: int,
    evaluator_parallelism: int,
) -> dict[str, Any]:
    return {
        "experiment_module": experiment_module,
        "config_index": config_index,
        "config_name": config.name,
        "problem": config.problem,
        "method": config.method,
        "algorithm_kind": config.algorithm_kind,
        "max_evals": config.max_evals,
        "n_init": config.n_init,
        "batch_size": config.batch_size,
        "kpls_n_dim": config.kpls_n_dim,
        "surrogate_name": config.surrogate_name,
        "surrogate_params": config.surrogate_params or {},
        "params": config.params,
        "problem_config": config.problem_config,
        "evaluator_parallelism": evaluator_parallelism,
    }
