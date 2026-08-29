from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EncoderRunConfig:
    experiment_alias: str | None
    problem: str
    setup: str
    params: dict[str, Any]
    problem_config: dict[str, Any]
    n_samples: int
    n_folds: int
    val_fold: int
    test_fold: int
    seed: int

    @property
    def feature_extractor(self) -> str:
        return str(self.params["feature_extractor"])


def make_encoder_configs(
    problem_configs: Mapping[str, Mapping[str, Any]],
    encoder_configs: Mapping[str, Mapping[str, Any]],
    n_folds: int,
    seeds: Sequence[int],
    experiment_alias: str | None = None,
) -> tuple[EncoderRunConfig, ...]:
    configs: list[EncoderRunConfig] = []
    for problem, raw_problem_config in problem_configs.items():
        problem_config = dict(raw_problem_config)
        n_samples = int(problem_config["n_samples"])
        for setup, raw_params in encoder_configs.items():
            params = dict(raw_params)
            selected_problems = params.pop("problems", None)
            if selected_problems is not None and problem not in selected_problems:
                continue
            for seed in seeds:
                for test_fold in range(n_folds):
                    val_fold = (test_fold + 1) % n_folds
                    configs.append(
                        EncoderRunConfig(
                            experiment_alias=experiment_alias,
                            problem=problem,
                            setup=setup,
                            params=params,
                            problem_config=problem_config,
                            n_samples=n_samples,
                            n_folds=int(n_folds),
                            val_fold=int(val_fold),
                            test_fold=int(test_fold),
                            seed=int(seed),
                        )
                    )
    return tuple(configs)


def format_config(config: EncoderRunConfig) -> str:
    experiment = f"experiment={config.experiment_alias}, " if config.experiment_alias else ""
    return (
        experiment
        + f"problem={config.problem}, "
        + f"setup={config.setup}, "
        + f"feature_extractor={config.feature_extractor}, "
        + f"n_samples={config.n_samples}, "
        + f"cv={config.n_folds}, "
        + f"val_fold={config.val_fold}, "
        + f"test_fold={config.test_fold}, "
        + f"seed={config.seed}, "
        + f"params={config.params}"
    )


def experiment_result_root(results_root: Path, config: EncoderRunConfig) -> Path:
    if config.experiment_alias is None:
        return results_root
    return results_root / config.experiment_alias


def config_result_dir(results_root: Path, config: EncoderRunConfig) -> Path:
    # No-fold runs (n_folds == 1) have a single fold, so the cv{n}/test_fold{n}
    # nesting is vestigial - flatten to setup/seed for those.
    if config.n_folds == 1:
        return setup_result_dir(results_root, config) / f"seed_{config.seed:03d}"
    return (
        setup_result_dir(results_root, config)
        / f"seed_{config.seed:03d}"
        / f"test_fold_{config.test_fold:03d}"
    )


def setup_result_dir(results_root: Path, config: EncoderRunConfig) -> Path:
    root = experiment_result_root(results_root, config) / config.problem
    if config.n_folds == 1:
        return root / config.setup
    return root / f"cv{config.n_folds}" / config.setup


def build_graph_processor(problem_name: str, resources_root: Path):
    from experimenter.problems.factory import build_graph_processor as build_problem_graph_processor

    return build_problem_graph_processor(problem_name, resources_root)


def feature_extractor_class(name: str):
    from graph_bo.gnn import ADSGNodeFeatureExtractor, ADSGNodeTypeFeatureExtractor

    if name == "node":
        return ADSGNodeFeatureExtractor
    if name == "node_type":
        return ADSGNodeTypeFeatureExtractor
    raise ValueError(f"Unknown feature extractor: {name!r}")
