from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OPENMDAO_REQUIRE_MPI", "false")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"pycycle\..*")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experimenter.optimization.grid import (
    ExperimentBuilder,
    ExperimentConfig,
    make_experiment_configs,
)
from graph_bo.kernels.extractors import ENCODING_LEVEL_DEPTHS

EXPERIMENT_ALIAS = "final_method_surrogates"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "kernels" / EXPERIMENT_ALIAS
RESULTS_ROOT = EXPERIMENT_DIR
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"

BASE_SEED = 2000
N_RUNS = 40
EVALUATOR_PARALLELISM = 1

SLURM_CPUS_PER_TASK = 4
SLURM_MEM = "24G"
SLURM_JOB_NAME = "finrest"
SLURM_PARTITION = "<cpu-partition>"
SLURM_ACCOUNT = None
SLURM_TIME = "24:00:00"
SLURM_MAIL_USER = "<your-email>"
SLURM_MAIL_TYPE = "fail"
MICROMAMBA_ENV = "adore"
MICROMAMBA_MODULE_USE = "<path-to-modulefiles>"
MICROMAMBA_MODULE = "micromamba"

MG_GRANULARITIES = [
    (level, ENCODING_LEVEL_DEPTHS[level])
    for level in ("adsg", "semantic_type")
]

PROBLEM_CONFIGS = {
    "mdgnc": {
        "reference_dir": RESOURCES_ROOT / "reference_fronts" / "mdgnc",
        "max_evals": 200,
        "evaluator_parallelism": 1,
        "bo_n_init": 30,
        "bo_batch_size": 10,
    },
    "engine": {
        "reference_dir": RESOURCES_ROOT / "reference_fronts" / "engine",
        "max_evals": 300,
        "evaluator_parallelism": 1,
        "bo_n_init": 75,
        "bo_batch_size": 10,
    },
    "mdgnc_edge_failures": {
        "reference_dir": RESOURCES_ROOT / "reference_fronts" / "mdgnc_edge_failures",
        "max_evals": 250,
        "evaluator_parallelism": 1,
        "bo_n_init": 30,
        "bo_batch_size": 10,
    },
}

METHOD_GRIDS = {
    "bo_no_kpls": [
        {
            "algorithm_kind": "bo_no_kpls",
            "use_bo_settings": True,
            "kpls_n_dim": None,
            "problems": {"mdgnc", "engine", "mdgnc_edge_failures"},
        },
    ],
    "bo_kpls": [
        {
            "algorithm_kind": "bo_kpls",
            "use_bo_settings": True,
            "kpls_n_dim": 10,
            "problems": {"mdgnc", "engine", "mdgnc_edge_failures"},
        },
    ],
    "blr_activeness": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": "blr_activeness",
            "surrogate_params": {},
            "problems": {"mdgnc", "engine", "mdgnc_edge_failures"},
        },
    ],
    "adsg_esp_semantic_type_imputed_interaction": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": "adsg_esp_semantic_type_imputed_interaction",
            "surrogate_params": {
                "sigma0": 1.0,
                "directed": True,
                "encoding_level": "semantic_type",
                "composition": "additive_interaction",
                "sizing_kernel_mode": "imputed",
                "lambda_l2": 0.0,
                "pow_exp_power": 1.9,
            },
            "problems": {"mdgnc", "engine"},
        },
    ],
    "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction",
            "surrogate_params": {
                "cutoff": 8,
                "composition": "additive_interaction",
                "sizing_kernel_mode": "imputed",
                "lambda_l2": 0.0,
                "pow_exp_power": 1.9,
            },
            "problems": {"mdgnc", "engine"},
        },
    ],
    "adsg_esp_semantic_type_edge_weight_imputed_interaction": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": "adsg_esp_semantic_type_edge_weight_imputed_interaction",
            "surrogate_params": {
                "sigma0": 1.0,
                "directed": True,
                "encoding_level": "semantic_type",
                "composition": "additive_interaction",
                "sizing_kernel_mode": "imputed",
                "lambda_l2": 0.0,
                "pow_exp_power": 1.9,
            },
            "problems": {"mdgnc_edge_failures"},
        },
    ],
    "adsg_mg_ld_wloa_adsg_semantic_type_d8_edge_weight_imputed_interaction": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": "adsg_mg_ld_wloa_adsg_semantic_type_d8_edge_weight_imputed_interaction",
            "surrogate_params": {
                "cutoff": 8,
                "composition": "additive_interaction",
                "sizing_kernel_mode": "imputed",
                "lambda_l2": 0.0,
                "pow_exp_power": 1.9,
            },
            "problems": {"mdgnc_edge_failures"},
        },
    ],
}


def build_blr_surrogate(problem, config: ExperimentConfig):
    from graph_bo.surrogates.bayes_linear import BayesianLinearSurrogate
    from sb_arch_opt.algo.arch_sbo.models import ModelFactory

    factory = ModelFactory(problem)
    normalization = factory.get_md_normalization()
    evaluator = problem.evaluator
    output_names = (
        [obj.ref if obj.ref is not None else obj.name for obj in evaluator.objectives]
        + [con.ref if con.ref is not None else con.name for con in evaluator.constraints]
    )
    return BayesianLinearSurrogate(output_names=output_names), normalization


def _build_graph_surrogate(problem, config: ExperimentConfig, graph_kernel, name: str, edge_weight: bool):
    from graph_bo.kernels import EdgeWeightKernel
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from graph_bo.surrogates.surrogates import GraphKernelHandler

    structure_kernels = [GraphKernelHandler(kernel=graph_kernel, name=name)]
    if edge_weight:
        structure_kernels.append(
            GraphKernelHandler(kernel=EdgeWeightKernel(gamma0=1.0), name="edge_weight")
        )
    return GraphModelFactory(problem).get_md_adsg_kriging_model(
        structure_kernels=structure_kernels,
        sizing_kernel=True,
        sizing_kernel_mode=config.surrogate_params["sizing_kernel_mode"],
        composition=config.surrogate_params["composition"],
        lambda_l2=float(config.surrogate_params["lambda_l2"]),
        multi=True,
        ignore_hierarchy=False,
        enable_timing=False,
        use_branch_theta0=True,
        poly="constant",
        corr="squar_exp",
        pow_exp_power=float(config.surrogate_params["pow_exp_power"]),
    )


def build_mg_ld_wloa_surrogate(problem, config: ExperimentConfig, edge_weight: bool = False):
    from graph_bo.kernels import MultiGranularityLdWloa

    kernel = MultiGranularityLdWloa(
        cutoff=int(config.surrogate_params["cutoff"]),
        granularities=MG_GRANULARITIES,
    )
    return _build_graph_surrogate(
        problem,
        config,
        kernel,
        "mg_ld_wloa_adsg_semantic_type",
        edge_weight,
    )


def build_esp_surrogate(problem, config: ExperimentConfig, edge_weight: bool = False):
    from graph_bo.kernels import ShortestPathKernel

    encoding_level = config.surrogate_params["encoding_level"]
    kernel = ShortestPathKernel(
        normalize=True,
        exponential=True,
        sigma0=float(config.surrogate_params["sigma0"]),
        directed=bool(config.surrogate_params["directed"]),
        depth_by_family=ENCODING_LEVEL_DEPTHS[encoding_level],
    )
    return _build_graph_surrogate(
        problem,
        config,
        kernel,
        f"esp_{encoding_level}",
        edge_weight,
    )


EXPERIMENT_BUILDER = ExperimentBuilder(
    resources_root=RESOURCES_ROOT,
    evaluator_parallelism=EVALUATOR_PARALLELISM,
)
EXPERIMENT_BUILDER.register_surrogate("blr_activeness", build_blr_surrogate)
EXPERIMENT_BUILDER.register_surrogate(
    "adsg_esp_semantic_type_imputed_interaction",
    build_esp_surrogate,
)
EXPERIMENT_BUILDER.register_surrogate(
    "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction",
    build_mg_ld_wloa_surrogate,
)
EXPERIMENT_BUILDER.register_surrogate(
    "adsg_esp_semantic_type_edge_weight_imputed_interaction",
    lambda problem, config: build_esp_surrogate(problem, config, edge_weight=True),
)
EXPERIMENT_BUILDER.register_surrogate(
    "adsg_mg_ld_wloa_adsg_semantic_type_d8_edge_weight_imputed_interaction",
    lambda problem, config: build_mg_ld_wloa_surrogate(problem, config, edge_weight=True),
)

CONFIGS = make_experiment_configs(PROBLEM_CONFIGS, METHOD_GRIDS)
format_config = EXPERIMENT_BUILDER.format_config
