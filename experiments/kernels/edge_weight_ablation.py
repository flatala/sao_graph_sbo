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
from experimenter.optimization.models import build_graph_kernel_surrogate
from graph_bo.kernels.extractors import EncodingDepth

EXPERIMENT_ALIAS = "edge_weight_ablation"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "kernels" / EXPERIMENT_ALIAS
RESULTS_ROOT = EXPERIMENT_DIR
SLURM_LOG_DIR = EXPERIMENT_DIR / "logs"
RESOURCES_ROOT = REPO_ROOT / "resources"

BASE_SEED = 52
N_RUNS = 20
EVALUATOR_PARALLELISM = 1

SLURM_CPUS_PER_TASK = 2
SLURM_MEM = "8G"
SLURM_JOB_NAME = "edgeabl"
SLURM_PARTITION = "<cpu-partition>"
SLURM_ACCOUNT = None
SLURM_TIME = "12:00:00"
SLURM_MAIL_USER = "<your-email>"
SLURM_MAIL_TYPE = "fail"
MICROMAMBA_ENV = "adore"
MICROMAMBA_MODULE_USE = "<path-to-modulefiles>"
MICROMAMBA_MODULE = "micromamba"

# Explicitly preserve the node encoding used to produce the retained ablation runs.
EDGE_WEIGHT_ABLATION_DEPTHS = {
    "external": EncodingDepth.FAMILY,
    "function": EncodingDepth.FAMILY,
    "function_decomposition": EncodingDepth.FAMILY,
    "component": EncodingDepth.SEMANTIC,
    "component_instance": EncodingDepth.SEMANTIC,
    "component_instance_group": EncodingDepth.FAMILY,
    "provided_port": EncodingDepth.FAMILY,
    "needed_port": EncodingDepth.FAMILY,
    "attribute": EncodingDepth.SEMANTIC,
    "attribute_value": EncodingDepth.SEMANTIC,
    "metric": EncodingDepth.FAMILY,
    "design_variable": EncodingDepth.SEMANTIC,
    "group": EncodingDepth.FAMILY,
    "connector_group": EncodingDepth.FAMILY,
}

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
}

METHOD_GRIDS = {
    "bo_no_kpls": [
        {
            "algorithm_kind": "bo_no_kpls",
            "use_bo_settings": True,
            "kpls_n_dim": None,
            "problems": {"mdgnc", "mdgnc_edge_failures"},
        },
    ],
    "adsg_ld_wloa": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": "adsg_ld_wloa_cutoff8",
            "surrogate_params": {"cutoff": 8},
            "problems": {"mdgnc", "mdgnc_edge_failures"},
        },
    ],
    "adsg_ld_wloa_edge_weight": [
        {
            "algorithm_kind": "surrogate_bo",
            "use_bo_settings": True,
            "surrogate_name": "adsg_ld_wloa_edge_weight_cutoff8",
            "surrogate_params": {"cutoff": 8},
            "problems": {"mdgnc_edge_failures"},
        },
    ],
}


def build_ld_wloa_surrogate(problem, config: ExperimentConfig):
    from graph_bo.kernels import LdWloa
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from graph_bo.surrogates.surrogates import GraphKernelHandler

    kernel = LdWloa(
        cutoff=int(config.surrogate_params["cutoff"]),
        depth_by_family=EDGE_WEIGHT_ABLATION_DEPTHS,
    )
    return GraphModelFactory(problem).get_md_adsg_kriging_model(
        structure_kernels=[GraphKernelHandler(kernel=kernel, name="ld_wloa")],
        sizing_kernel=True,
        composition="additive",
        multi=True,
        ignore_hierarchy=False,
        enable_timing=False,
        use_branch_theta0=True,
        poly="constant",
        corr="squar_exp",
    )


def build_ld_wloa_edge_weight_surrogate(problem, config: ExperimentConfig):
    from graph_bo.kernels import LdWloa, EdgeWeightKernel
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from graph_bo.surrogates.surrogates import GraphKernelHandler

    ld_wloa = LdWloa(
        cutoff=int(config.surrogate_params["cutoff"]),
        depth_by_family=EDGE_WEIGHT_ABLATION_DEPTHS,
    )
    edge_weight = EdgeWeightKernel(gamma0=1.0)
    return GraphModelFactory(problem).get_md_adsg_kriging_model(
        structure_kernels=[
            GraphKernelHandler(kernel=ld_wloa, name="ld_wloa"),
            GraphKernelHandler(kernel=edge_weight, name="edge_weight"),
        ],
        sizing_kernel=True,
        composition="additive",
        multi=True,
        ignore_hierarchy=False,
        enable_timing=False,
        use_branch_theta0=True,
        poly="constant",
        corr="squar_exp",
    )


EXPERIMENT_BUILDER = ExperimentBuilder(
    resources_root=RESOURCES_ROOT,
    evaluator_parallelism=EVALUATOR_PARALLELISM,
)
EXPERIMENT_BUILDER.register_surrogate("adsg_ld_wloa_cutoff8", build_ld_wloa_surrogate)
EXPERIMENT_BUILDER.register_surrogate("adsg_ld_wloa_edge_weight_cutoff8", build_ld_wloa_edge_weight_surrogate)

CONFIGS = make_experiment_configs(PROBLEM_CONFIGS, METHOD_GRIDS)
format_config = EXPERIMENT_BUILDER.format_config
