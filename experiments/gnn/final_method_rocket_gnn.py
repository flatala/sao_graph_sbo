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

from experiments.gnn.final_method_arch2vec_bo import (  # noqa: E402
    BETA,
    DEPTH,
    ENCODER_ADJ_CUTOFF,
    ENCODER_ADJ_DECODER,
    ENCODER_BATCH_SIZE,
    ENCODER_HIDDEN_DIM,
    ENCODER_LR,
    ENCODER_NORM,
    ENCODER_RESULTS_ROOT,
    ENCODER_SELECTION_WINDOW,
    ENCODER_VAL_FRACTION,
    HEAD_BATCH_SIZE,
    HEAD_CALIBRATE_FOLDS,
    HEAD_HIDDEN_UNITS,
    HEAD_LR,
    HEAD_N_EPOCHS,
    LATENT_DIM,
    RECONSTRUCTION_ADJ_CUTOFF,
    SETUP,
)
from experimenter.optimization.models import (  # noqa: E402
    DGBO_PARAMS,
    build_arch2vec_dngo,
    build_dgbo,
)
from experimenter.optimization.grid import (  # noqa: E402
    ExperimentBuilder,
    make_experiment_configs,
)

EXPERIMENT_ALIAS = "final_method_rocket_gnn"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "gnn" / EXPERIMENT_ALIAS
RESULTS_ROOT = EXPERIMENT_DIR
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"

BASE_SEED = 2000
N_RUNS = 40
EVALUATOR_PARALLELISM = 1

SLURM_CPUS_PER_TASK = 6
SLURM_MEM = "96G"
SLURM_JOB_NAME = "finrktgpu"
SLURM_CLUSTERS = "<slurm-cluster>"
SLURM_PARTITION = "<gpu-partition>"
SLURM_GRES = "gpu:1"
SLURM_ACCOUNT = None
SLURM_RUNS_PER_TASK = 1
SLURM_TASK_PARALLELISM = 1
SLURM_ARRAY_CHUNK_SIZE = 500
SLURM_TIME = "72:00:00"
SLURM_MAIL_USER = "<your-email>"
SLURM_MAIL_TYPE = "fail"
MICROMAMBA_ENV = "adore-cuda"
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
    "dgbo_mean_lcb": [
        {
            "algorithm_kind": "bo_dgbo",
            "use_bo_settings": True,
            "surrogate_params": dict(DGBO_PARAMS),
            "infill_kind": "mean_lcb",
        },
    ],
    f"gnn_dngo_{SETUP}_e050_mean_lcb": [
        {
            "algorithm_kind": "bo_gnn_dngo",
            "use_bo_settings": True,
            "encoder_results_root": str(ENCODER_RESULTS_ROOT),
            "encoder_setup": SETUP,
            "encoder_epoch": 50,
            "encoder_depth": DEPTH,
            "encoder_beta": BETA,
            "encoder_latent_dim": LATENT_DIM,
            "encoder_hidden_dim": ENCODER_HIDDEN_DIM,
            "encoder_batch_size": ENCODER_BATCH_SIZE,
            "encoder_lr": ENCODER_LR,
            "encoder_adj_cutoff": ENCODER_ADJ_CUTOFF,
            "encoder_selection_window": ENCODER_SELECTION_WINDOW,
            "encoder_norm": ENCODER_NORM,
            "encoder_adj_decoder": ENCODER_ADJ_DECODER,
            "encoder_val_fraction": ENCODER_VAL_FRACTION,
            "reconstruction_adj_cutoff": RECONSTRUCTION_ADJ_CUTOFF,
            "head_hidden_units": HEAD_HIDDEN_UNITS,
            "head_n_epochs": HEAD_N_EPOCHS,
            "head_batch_size": HEAD_BATCH_SIZE,
            "head_lr": HEAD_LR,
            "head_calibrate_folds": HEAD_CALIBRATE_FOLDS,
            "infill_kind": "mean_lcb",
        },
    ],
}

EXPERIMENT_BUILDER = ExperimentBuilder(
    resources_root=RESOURCES_ROOT,
    evaluator_parallelism=EVALUATOR_PARALLELISM,
)
EXPERIMENT_BUILDER.register_algorithm("bo_dgbo", build_dgbo)
EXPERIMENT_BUILDER.register_algorithm("bo_gnn_dngo", build_arch2vec_dngo)

CONFIGS = make_experiment_configs(PROBLEM_CONFIGS, METHOD_GRIDS)
format_config = EXPERIMENT_BUILDER.format_config
