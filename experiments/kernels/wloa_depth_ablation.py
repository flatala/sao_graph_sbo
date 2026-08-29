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
from graph_bo.kernels import Wloa
from graph_bo.kernels.extractors import ENCODING_LEVEL_DEPTHS

EXPERIMENT_ALIAS = "wloa_depth_ablation"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "kernels" / EXPERIMENT_ALIAS
RESULTS_ROOT = EXPERIMENT_DIR
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"

BASE_SEED = 52
N_RUNS = 20
EVALUATOR_PARALLELISM = 1

SLURM_CPUS_PER_TASK = 4
SLURM_MEM = "24G"
SLURM_JOB_NAME = "semwldepth"
SLURM_PARTITION = "<cpu-partition>"
SLURM_ACCOUNT = None
SLURM_TIME = "24:00:00"
SLURM_MAIL_USER = "<your-email>"
SLURM_MAIL_TYPE = "fail"
MICROMAMBA_ENV = "adore"
MICROMAMBA_MODULE_USE = "<path-to-modulefiles>"
MICROMAMBA_MODULE = "micromamba"

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
}

FIXED_DEPTHS = tuple(range(9))
SEMANTIC_TYPE_DEPTHS = ENCODING_LEVEL_DEPTHS["semantic_type"]


COMMON_SURROGATE_PARAMS = {
    "composition": "additive_interaction",
    "sizing_kernel_mode": "imputed",
    "lambda_l2": 0.0,
    "pow_exp_power": 1.9,
}

METHOD_GRIDS = {
    "fixed_depth_wloa": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": f"adsg_wloa_semantic_type_d{depth}_imputed_interaction",
            "surrogate_params": {
                **COMMON_SURROGATE_PARAMS,
                "depth": depth,
            },
            "problems": {"mdgnc", "engine"},
        }
        for depth in FIXED_DEPTHS
    ],
}


def _build_wloa_surrogate(
    problem,
    config: ExperimentConfig,
    kernel,
    branch_name: str,
):
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from graph_bo.surrogates.surrogates import GraphKernelHandler

    return GraphModelFactory(problem).get_md_adsg_kriging_model(
        structure_kernels=[
            GraphKernelHandler(
                kernel=kernel,
                name=branch_name,
            )
        ],
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


def build_fixed_depth_wloa_surrogate(problem, config: ExperimentConfig):
    kernel = Wloa(
        depth=int(config.surrogate_params["depth"]),
        depth_by_family=SEMANTIC_TYPE_DEPTHS,
    )
    return _build_wloa_surrogate(
        problem,
        config,
        kernel,
        "wloa_semantic_type",
    )


EXPERIMENT_BUILDER = ExperimentBuilder(
    resources_root=RESOURCES_ROOT,
    evaluator_parallelism=EVALUATOR_PARALLELISM,
)
for depth in FIXED_DEPTHS:
    EXPERIMENT_BUILDER.register_surrogate(
        f"adsg_wloa_semantic_type_d{depth}_imputed_interaction",
        build_fixed_depth_wloa_surrogate,
    )
CONFIGS = make_experiment_configs(PROBLEM_CONFIGS, METHOD_GRIDS)
format_config = EXPERIMENT_BUILDER.format_config
