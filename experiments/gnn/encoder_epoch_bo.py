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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimenter.optimization.grid import ExperimentBuilder, make_experiment_configs  # noqa: E402
from experimenter.optimization.models import build_arch2vec_dngo  # noqa: E402


EXPERIMENT_ALIAS = "encoder_epoch_bo"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "gnn" / EXPERIMENT_ALIAS
RESULTS_ROOT = EXPERIMENT_DIR
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"

# Seed-matched to encoder_epoch_training.py: run_id 0 loads seed_052, etc.
ENCODER_RESULTS_ROOT = REPO_ROOT / "experiments" / "gnn" / "encoder_epoch_training"
BASE_SEED = 52
N_RUNS = 20
EVALUATOR_PARALLELISM = 1

DEPTHS = (3, 4, 5)
BETAS = (0.0, 0.0025, 0.01)
EPOCHS = (0, 10, 25, 50, 100, 150)
LATENT_DIM = 32

ENCODER_HIDDEN_DIM = 64
ENCODER_BATCH_SIZE = 64
ENCODER_LR = 1e-3
ENCODER_ADJ_CUTOFF = 0.3
ENCODER_SELECTION_WINDOW = 5
ENCODER_NORM = "batch"
ENCODER_ADJ_DECODER = "mlp"
ENCODER_VAL_FRACTION = 0.1
RECONSTRUCTION_ADJ_CUTOFF = 0.3

HEAD_HIDDEN_UNITS = 128
HEAD_N_EPOCHS = 10
HEAD_BATCH_SIZE = 10
HEAD_LR = 1e-2
HEAD_CALIBRATE_FOLDS = 0

SLURM_CPUS_PER_TASK = 3
SLURM_MEM = "16G"
SLURM_JOB_NAME = "a2v2bo"
SLURM_PARTITION = "<cpu-partition>"
SLURM_ACCOUNT = None
SLURM_ARRAY_CHUNK_SIZE = 500
SLURM_TIME = "2:00:00"
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
}


def _beta_tag(beta: float) -> str:
    return f"{beta:g}".replace(".", "p")


def _setup(depth: int, beta: float) -> str:
    return f"node_d{depth}_b{_beta_tag(beta)}_z{LATENT_DIM}"


def _method_grids() -> dict[str, list[dict]]:
    grids: dict[str, list[dict]] = {}
    for depth in DEPTHS:
        for beta in BETAS:
            setup = _setup(depth, beta)
            for epoch in EPOCHS:
                method = f"gnn_dngo_{setup}_e{int(epoch):03d}"
                grids[method] = [
                    {
                        "algorithm_kind": "bo_gnn_dngo",
                        "use_bo_settings": True,
                        "problems": {"mdgnc", "engine"},
                        "encoder_results_root": str(ENCODER_RESULTS_ROOT),
                        "encoder_setup": setup,
                        "encoder_epoch": int(epoch),
                        "encoder_depth": int(depth),
                        "encoder_beta": float(beta),
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
                    }
                ]
    return grids


METHOD_GRIDS = _method_grids()

EXPERIMENT_BUILDER = ExperimentBuilder(resources_root=RESOURCES_ROOT, evaluator_parallelism=EVALUATOR_PARALLELISM)
EXPERIMENT_BUILDER.register_algorithm("bo_gnn_dngo", build_arch2vec_dngo)

CONFIGS = make_experiment_configs(PROBLEM_CONFIGS, METHOD_GRIDS)
format_config = EXPERIMENT_BUILDER.format_config
