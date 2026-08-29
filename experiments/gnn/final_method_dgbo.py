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

from experimenter.optimization.grid import (  # noqa: E402
    ExperimentBuilder,
    make_experiment_configs,
)
from experimenter.optimization.models import (  # noqa: E402
    DGBO_PARAMS,
    build_dgbo,
)

EXPERIMENT_ALIAS = "final_method_dgbo"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "gnn" / EXPERIMENT_ALIAS
RESULTS_ROOT = EXPERIMENT_DIR
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"

# Seeds match the final_method runs: run_seed = BASE_SEED + run_id -> {2000..2039}. The
# DGBO net (weight init + minibatch shuffle) is seeded per run with that same run_seed.
BASE_SEED = 2000
N_RUNS = 40
EVALUATOR_PARALLELISM = 1

SLURM_CPUS_PER_TASK = 8
SLURM_MEM = "96G"
SLURM_JOB_NAME = "findgbo"
SLURM_CLUSTERS = "<slurm-cluster>"
SLURM_PARTITION = "<gpu-partition>"
SLURM_GRES = "gpu:1"
SLURM_ACCOUNT = None
SLURM_RUNS_PER_TASK = 3
SLURM_TASK_PARALLELISM = 3
SLURM_ARRAY_CHUNK_SIZE = 500
SLURM_TIME = "10:00:00"
SLURM_MAIL_USER = "<your-email>"
SLURM_MAIL_TYPE = "fail"
MICROMAMBA_ENV = "adore-cuda"
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
    "mdgnc_edge_failures": {
        "reference_dir": RESOURCES_ROOT / "reference_fronts" / "mdgnc_edge_failures",
        "max_evals": 250,
        "evaluator_parallelism": 1,
        "bo_n_init": 30,
        "bo_batch_size": 10,
    },
}

METHOD_GRIDS = {
    "dgbo": [
        {
            "algorithm_kind": "bo_dgbo",
            "use_bo_settings": True,
            "surrogate_params": DGBO_PARAMS,
            "problems": {"mdgnc", "engine", "mdgnc_edge_failures"},
        },
    ],
}


EXPERIMENT_BUILDER = ExperimentBuilder(resources_root=RESOURCES_ROOT, evaluator_parallelism=EVALUATOR_PARALLELISM)
EXPERIMENT_BUILDER.register_algorithm("bo_dgbo", build_dgbo)

CONFIGS = make_experiment_configs(PROBLEM_CONFIGS, METHOD_GRIDS)
format_config = EXPERIMENT_BUILDER.format_config
