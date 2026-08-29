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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experimenter.optimization.models import (  # noqa: E402
    build_esp_semantic_type_surrogate,
    build_mg_ld_wloa_adsg_semantic_type_surrogate,
)
from experiments.kernels.final_method_surrogates import build_blr_surrogate  # noqa: E402
from experimenter.optimization.grid import (  # noqa: E402
    ExperimentBuilder,
    make_experiment_configs,
)

EXPERIMENT_ALIAS = "final_method_rocket"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "kernels" / EXPERIMENT_ALIAS
RESULTS_ROOT = EXPERIMENT_DIR
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"

BASE_SEED = 2000
N_RUNS = 40
EVALUATOR_PARALLELISM = 1

SLURM_CPUS_PER_TASK = 4
SLURM_MEM = "48G"
SLURM_JOB_NAME = "finrktlcb"
SLURM_PARTITION = "<cpu-partition>"
SLURM_ACCOUNT = None
SLURM_ARRAY_CHUNK_SIZE = 500
SLURM_TIME = "72:00:00"
SLURM_MAIL_USER = "<your-email>"
SLURM_MAIL_TYPE = "fail"
MICROMAMBA_ENV = "adore"
MICROMAMBA_MODULE_USE = "<path-to-modulefiles>"
MICROMAMBA_MODULE = "micromamba"

PROBLEM_CONFIGS = {
    "rocket": {
        "reference_dir": RESOURCES_ROOT / "reference_fronts" / "rocket",
        "max_evals": 750,
        "evaluator_parallelism": 1,
        "bo_n_init": 100,
        "bo_batch_size": 10,
    },
}

METHOD_GRIDS = {
    "random_hierarchical": [
        {
            "algorithm_kind": "random",
            "use_bo_settings": True,
        },
    ],
    "ga": [
        {
            "algorithm_kind": "ga",
            "use_bo_settings": True,
        },
    ],
    "bo_no_kpls_mean_lcb": [
        {
            "algorithm_kind": "bo_no_kpls",
            "use_bo_settings": True,
            "kpls_n_dim": None,
            "infill_kind": "mean_lcb",
        },
    ],
    "bo_kpls_mean_lcb": [
        {
            "algorithm_kind": "bo_kpls",
            "use_bo_settings": True,
            "kpls_n_dim": 10,
            "infill_kind": "mean_lcb",
        },
    ],
    "blr_activeness_mean_lcb": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": "blr_activeness_mean_lcb",
            "surrogate_params": {},
            "infill_kind": "mean_lcb",
        },
    ],
    "adsg_esp_semantic_type_imputed_interaction_mean_lcb": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": "adsg_esp_semantic_type_imputed_interaction_mean_lcb",
            "surrogate_params": {
                "sigma0": 1.0,
                "directed": True,
                "encoding_level": "semantic_type",
                "composition": "additive_interaction",
                "sizing_kernel_mode": "imputed",
                "lambda_l2": 0.0,
                "pow_exp_power": 1.9,
            },
            "infill_kind": "mean_lcb",
        },
    ],
    "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction_mean_lcb": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": (
                "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction_mean_lcb"
            ),
            "surrogate_params": {
                "cutoff": 8,
                "composition": "additive_interaction",
                "sizing_kernel_mode": "imputed",
                "lambda_l2": 0.0,
                "pow_exp_power": 1.9,
            },
            "infill_kind": "mean_lcb",
        },
    ],
}

EXPERIMENT_BUILDER = ExperimentBuilder(
    resources_root=RESOURCES_ROOT,
    evaluator_parallelism=EVALUATOR_PARALLELISM,
)
EXPERIMENT_BUILDER.register_surrogate(
    "blr_activeness_mean_lcb",
    build_blr_surrogate,
)
EXPERIMENT_BUILDER.register_surrogate(
    "adsg_esp_semantic_type_imputed_interaction_mean_lcb",
    build_esp_semantic_type_surrogate,
)
EXPERIMENT_BUILDER.register_surrogate(
    "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction_mean_lcb",
    build_mg_ld_wloa_adsg_semantic_type_surrogate,
)

CONFIGS = make_experiment_configs(PROBLEM_CONFIGS, METHOD_GRIDS)
format_config = EXPERIMENT_BUILDER.format_config
