from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experimenter.surrogate_fit.models import (
    build_blr_activeness,
    build_esp,
    build_md_kriging,
    build_mg_ld_wloa,
)


@dataclass(frozen=True)
class SurrogateFitConfig:
    experiment_alias: str | None
    problem: str
    surrogate: str
    params: dict[str, Any]
    problem_config: dict[str, Any]
    n_samples: int
    n_folds: int
    budget: int
    fold: int

    @property
    def surrogate_kind(self) -> str:
        return str(self.params["surrogate_kind"])


class SurrogateBenchmarkBuilder:
    def __init__(self, resources_root: Path, data_root: Path, evaluator_parallelism: int):
        self.resources_root = resources_root
        self.data_root = data_root
        self.evaluator_parallelism = evaluator_parallelism

    def build_problem(self, config: SurrogateFitConfig):
        evaluator_parallelism = int(config.problem_config.get("evaluator_parallelism", self.evaluator_parallelism))
        return build_adore_problem(config.problem, self.resources_root, evaluator_parallelism)

    def build_surrogate(self, problem, config: SurrogateFitConfig):
        params = config.params
        if config.surrogate_kind == "md_kriging":
            return build_md_kriging(problem, kpls_n_comp=params["kpls_n_comp"])
        if config.surrogate_kind == "blr_activeness":
            return build_blr_activeness(problem)
        if config.surrogate_kind == "adsg_esp_semantic_type_imputed_interaction":
            return build_esp(problem, params=params)
        if config.surrogate_kind == "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction":
            return build_mg_ld_wloa(problem, params=params)
        raise ValueError(f"Unknown surrogate kind: {config.surrogate_kind!r}")

    def dataset_path(self, config: SurrogateFitConfig) -> Path:
        return self.data_root / config.problem / f"samples_{config.n_samples}.csv"

    def format_config(self, config: SurrogateFitConfig) -> str:
        return format_config(config)


def make_surrogate_configs(
    problem_configs: Mapping[str, Mapping[str, Any]],
    surrogate_configs: Mapping[str, Mapping[str, Any]],
    budgets: tuple[int, ...],
    n_folds: int,
    experiment_alias: str | None = None,
) -> tuple[SurrogateFitConfig, ...]:
    configs: list[SurrogateFitConfig] = []
    for problem, raw_problem_config in problem_configs.items():
        problem_config = dict(raw_problem_config)
        n_samples = int(problem_config["n_samples"])
        for surrogate, raw_params in surrogate_configs.items():
            params = dict(raw_params)
            selected_problems = params.pop("problems", None)
            if selected_problems is not None and problem not in selected_problems:
                continue
            for fold in range(n_folds):
                for budget in budgets:
                    configs.append(
                        SurrogateFitConfig(
                            experiment_alias=experiment_alias,
                            problem=problem,
                            surrogate=surrogate,
                            params=params,
                            problem_config=problem_config,
                            n_samples=n_samples,
                            n_folds=n_folds,
                            budget=int(budget),
                            fold=int(fold),
                        )
                    )
    return tuple(configs)


def format_config(config: SurrogateFitConfig) -> str:
    experiment = f"experiment={config.experiment_alias}, " if config.experiment_alias else ""
    return (
        experiment +
        f"problem={config.problem}, "
        f"surrogate={config.surrogate}, "
        f"kind={config.surrogate_kind}, "
        f"n_samples={config.n_samples}, "
        f"cv={config.n_folds}, "
        f"fold={config.fold}, "
        f"budget={config.budget}, "
        f"params={config.params}"
    )


def experiment_result_root(results_root: Path, config: SurrogateFitConfig) -> Path:
    if config.experiment_alias is None:
        return results_root
    return results_root / config.experiment_alias


def config_result_dir(results_root: Path, config: SurrogateFitConfig) -> Path:
    return (
        experiment_result_root(results_root, config)
        / config.problem
        / f"cv{config.n_folds}"
        / config.surrogate
        / f"budget_{config.budget:04d}"
        / f"fold_{config.fold:03d}"
    )


def model_result_dir(results_root: Path, config: SurrogateFitConfig) -> Path:
    return experiment_result_root(results_root, config) / config.problem / f"cv{config.n_folds}" / config.surrogate


def build_adore_problem(problem_name: str, resources_root: Path, evaluator_parallelism: int = 1):
    from experimenter.problems.factory import build_problem

    return build_problem(problem_name, resources_root, evaluator_parallelism)
