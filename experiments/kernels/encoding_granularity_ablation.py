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

EXPERIMENT_ALIAS = "encoding_granularity_ablation"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "kernels" / EXPERIMENT_ALIAS
RESULTS_ROOT = EXPERIMENT_DIR
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"

BASE_SEED = 52
N_RUNS = 20
EVALUATOR_PARALLELISM = 1

SLURM_CPUS_PER_TASK = 2
SLURM_MEM = "16G"
SLURM_JOB_NAME = "encinter"
SLURM_PARTITION = "<cpu-partition>"
SLURM_ACCOUNT = None
SLURM_TIME = "12:00:00"
SLURM_MAIL_USER = "<your-email>"
SLURM_MAIL_TYPE = "fail"
MICROMAMBA_ENV = "adore"
MICROMAMBA_MODULE_USE = "<path-to-modulefiles>"
MICROMAMBA_MODULE = "micromamba"

CUTOFF = 8
ENCODING_LEVELS = ("dsg", "adsg", "semantic_type")

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

METHOD_GRIDS = {
    f"adsg_ld_wloa_{encoding_level}_d{CUTOFF}_imputed_interaction": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": (
                f"adsg_ld_wloa_{encoding_level}_d{CUTOFF}_imputed_interaction"
            ),
            "surrogate_params": {
                "cutoff": CUTOFF,
                "encoding_level": encoding_level,
                "composition": "additive_interaction",
                "sizing_kernel_mode": "imputed",
                "lambda_l2": 0.0,
                "pow_exp_power": 1.9,
            },
            "problems": {"mdgnc", "engine"},
        }
    ]
    for encoding_level in ENCODING_LEVELS
}
def build_ld_wloa_encoding_surrogate(problem, config: ExperimentConfig):
    from graph_bo.kernels import LdWloa
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from graph_bo.surrogates.surrogates import GraphKernelHandler

    encoding_level = config.surrogate_params["encoding_level"]
    kernel = LdWloa(
        cutoff=int(config.surrogate_params["cutoff"]),
        depth_by_family=ENCODING_LEVEL_DEPTHS[encoding_level],
    )
    return GraphModelFactory(problem).get_md_adsg_kriging_model(
        structure_kernels=[
            GraphKernelHandler(kernel=kernel, name=f"ld_wloa_{encoding_level}")
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


EXPERIMENT_BUILDER = ExperimentBuilder(
    resources_root=RESOURCES_ROOT,
    evaluator_parallelism=EVALUATOR_PARALLELISM,
)
for encoding_level in ENCODING_LEVELS:
    surrogate_name = f"adsg_ld_wloa_{encoding_level}_d{CUTOFF}_imputed_interaction"
    EXPERIMENT_BUILDER.register_surrogate(
        surrogate_name,
        build_ld_wloa_encoding_surrogate,
    )
CONFIGS = make_experiment_configs(PROBLEM_CONFIGS, METHOD_GRIDS)
format_config = EXPERIMENT_BUILDER.format_config
