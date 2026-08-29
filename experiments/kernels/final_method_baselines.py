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
    make_experiment_configs,
)

EXPERIMENT_ALIAS = "final_method_baselines"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "kernels" / EXPERIMENT_ALIAS
RESULTS_ROOT = EXPERIMENT_DIR
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"

BASE_SEED = 2000
N_RUNS = 40
EVALUATOR_PARALLELISM = 1

SLURM_CPUS_PER_TASK = 2
SLURM_MEM = "4G"
SLURM_JOB_NAME = "finbase"
SLURM_PARTITION = "<cpu-partition>"
SLURM_ACCOUNT = None
SLURM_TIME = "01:00:00"
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
    "mdgnc_edge_failures": {
        "reference_dir": RESOURCES_ROOT / "reference_fronts" / "mdgnc_edge_failures",
        "max_evals": 250,
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
    "ga": [
        {
            "algorithm_kind": "ga",
            "use_bo_settings": True,
            "problems": {"mdgnc", "engine", "mdgnc_edge_failures"},
        },
    ],
    "random": [
        {
            "algorithm_kind": "random",
            "use_bo_settings": True,
            "problems": {"mdgnc", "engine", "mdgnc_edge_failures"},
        },
    ],
}

EXPERIMENT_BUILDER = ExperimentBuilder(resources_root=RESOURCES_ROOT, evaluator_parallelism=EVALUATOR_PARALLELISM)
CONFIGS = make_experiment_configs(PROBLEM_CONFIGS, METHOD_GRIDS)
format_config = EXPERIMENT_BUILDER.format_config
