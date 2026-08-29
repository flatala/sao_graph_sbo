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

from experimenter.surrogate_fit.grid import (
    SurrogateBenchmarkBuilder,
    make_surrogate_configs,
)

EXPERIMENT_ALIAS = "inductive_bias_ablation"
RESULTS_ROOT = REPO_ROOT / "experiments" / "kernels"
EXPERIMENT_DIR = RESULTS_ROOT / EXPERIMENT_ALIAS
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"
DATA_ROOT = REPO_ROOT / "resources" / "datasets"

N_FOLDS = 10
BUDGETS = tuple(range(25, 301, 25))
EVALUATOR_PARALLELISM = 1

SLURM_CPUS_PER_TASK = 1
SLURM_MEM = "8G"
SLURM_ARRAY_CHUNK_SIZE = 500
SLURM_JOB_NAME = "mdgind"
SLURM_PARTITION = "<cpu-partition>"
SLURM_ACCOUNT = None
SLURM_TIME = "00:30:00"
SLURM_MAIL_USER = "<your-email>"
SLURM_MAIL_TYPE = "fail"
MICROMAMBA_ENV = "adore"
MICROMAMBA_MODULE_USE = "<path-to-modulefiles>"
MICROMAMBA_MODULE = "micromamba"

PROBLEM_CONFIGS = {
    "mdgnc": {
        "n_samples": 5000,
        "evaluator_parallelism": 1,
        "target_columns": ("f0", "f1"),
        "train_selection": "unique_discrete_coverage_v1",
    },
    "mdgnc_edge_failures": {
        "n_samples": 5000,
        "evaluator_parallelism": 1,
        "target_columns": ("f0", "f1"),
        "train_selection": "unique_discrete_coverage_v1",
    },
}

SURROGATE_CONFIGS = {
    "Hierarchical": {
        "surrogate_kind": "md_kriging",
        "kpls_n_comp": None,
    },
    "BLR": {
        "surrogate_kind": "blr_activeness",
    },
    "KPLS": {
        "surrogate_kind": "md_kriging",
        "kpls_n_comp": 10,
    },
    "ESP": {
        "surrogate_kind": "adsg_esp_semantic_type_imputed_interaction",
        "sigma0": 1.0,
        "directed": True,
        "encoding_level": "semantic_type",
        "composition": "additive_interaction",
        "sizing_kernel_mode": "imputed",
        "lambda_l2": 0.0,
        "pow_exp_power": 1.9,
        "edge_weight": False,
        "problems": {"mdgnc"},
    },
    "MG LdWLOA": {
        "surrogate_kind": "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction",
        "cutoff": 8,
        "encoding_levels": ("adsg", "semantic_type"),
        "composition": "additive_interaction",
        "sizing_kernel_mode": "imputed",
        "lambda_l2": 0.0,
        "pow_exp_power": 1.9,
        "edge_weight": False,
        "problems": {"mdgnc"},
    },
    "ESP + EW": {
        "surrogate_kind": "adsg_esp_semantic_type_imputed_interaction",
        "sigma0": 1.0,
        "directed": True,
        "encoding_level": "semantic_type",
        "composition": "additive_interaction",
        "sizing_kernel_mode": "imputed",
        "lambda_l2": 0.0,
        "pow_exp_power": 1.9,
        "edge_weight": True,
        "problems": {"mdgnc_edge_failures"},
    },
    "MG LdWLOA + EW": {
        "surrogate_kind": "adsg_mg_ld_wloa_adsg_semantic_type_d8_imputed_interaction",
        "cutoff": 8,
        "encoding_levels": ("adsg", "semantic_type"),
        "composition": "additive_interaction",
        "sizing_kernel_mode": "imputed",
        "lambda_l2": 0.0,
        "pow_exp_power": 1.9,
        "edge_weight": True,
        "problems": {"mdgnc_edge_failures"},
    },
}

BENCHMARK_BUILDER = SurrogateBenchmarkBuilder(
    resources_root=RESOURCES_ROOT,
    data_root=DATA_ROOT,
    evaluator_parallelism=EVALUATOR_PARALLELISM,
)
CONFIGS = make_surrogate_configs(
    PROBLEM_CONFIGS,
    SURROGATE_CONFIGS,
    BUDGETS,
    N_FOLDS,
    experiment_alias=EXPERIMENT_ALIAS,
)
format_config = BENCHMARK_BUILDER.format_config
