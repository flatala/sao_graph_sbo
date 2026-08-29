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

from experimenter.encoder_training.grid import make_encoder_configs, format_config  # noqa: F401,E402


EXPERIMENT_ALIAS = "final_method_arch2vec_encoders"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "gnn" / EXPERIMENT_ALIAS
RESULTS_ROOT = REPO_ROOT / "experiments" / "gnn"
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"
DATA_ROOT = REPO_ROOT / "resources" / "datasets"

# One Slurm array task gets one GPU. Inside that task we run this many configs
# concurrently on the same visible GPU.
CONFIGS_PER_TASK = 10
PARALLEL_CONFIGS_PER_GPU = 10

N_FOLDS = 1
BASE_SEED = 2000
N_RERUNS = 40
SEEDS = tuple(BASE_SEED + i for i in range(N_RERUNS))

# Thesis Arch2Vec baseline: selected from the reconstruction/BO trade-off sweep.
FEATURE_EXTRACTOR = "node"
HIDDEN_DIM = 64
LATENT_DIM = 32
N_GIN_LAYERS = 5
BATCH_SIZE = 64
N_EPOCHS = 50
SNAPSHOT_EPOCHS = [0, 10, 50]
LR = 1e-3
BETA = 0.0025
ADJ_CUTOFF = 0.3
SELECTION_WINDOW = 5
VAL_FRACTION = 0.1
NORM = "batch"
ADJ_DECODER = "mlp"
N_SAMPLES = 5000

SLURM_CPUS_PER_TASK = 18
SLURM_MEM = "220G"
SLURM_JOB_NAME = "a2vbase"
SLURM_CLUSTERS = "<slurm-cluster>"
SLURM_PARTITION = "<gpu-partition>"
SLURM_GRES = "gpu:1"
SLURM_ACCOUNT = None
SLURM_TIME = "12:00:00"
SLURM_ARRAY_CONCURRENCY = None
SLURM_MAIL_USER = "<your-email>"
SLURM_MAIL_TYPE = "fail"
MICROMAMBA_ENV = "adore-cuda"
MICROMAMBA_MODULE_USE = "<path-to-modulefiles>"
MICROMAMBA_MODULE = "micromamba"


ENABLED_PROBLEMS = (
    "engine",
    "rocket",
    "mdgnc",
    "mdgnc_edge_failures",
)

PROBLEM_CONFIGS = {problem: {"n_samples": N_SAMPLES} for problem in ENABLED_PROBLEMS}


def _beta_tag(beta: float) -> str:
    return f"{beta:g}".replace(".", "p")


SETUP = f"{FEATURE_EXTRACTOR}_d{N_GIN_LAYERS}_b{_beta_tag(BETA)}_z{LATENT_DIM}"

ENCODER_CONFIGS = {
    SETUP: {
        "feature_extractor": FEATURE_EXTRACTOR,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "n_gin_layers": N_GIN_LAYERS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "beta": BETA,
        "adj_cutoff": ADJ_CUTOFF,
        "selection_window": SELECTION_WINDOW,
        "norm": NORM,
        "adj_decoder": ADJ_DECODER,
        "val_fraction": VAL_FRACTION,
        "source": "resample",
        "snapshot_epochs": list(SNAPSHOT_EPOCHS),
        "torch_threads": 1,
    }
}


CONFIGS = make_encoder_configs(
    problem_configs=PROBLEM_CONFIGS,
    encoder_configs=ENCODER_CONFIGS,
    n_folds=N_FOLDS,
    seeds=SEEDS,
    experiment_alias=EXPERIMENT_ALIAS,
)
