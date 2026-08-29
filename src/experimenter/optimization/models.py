from __future__ import annotations

from pathlib import Path


# DGBO paper architecture with the problem-tuned learning rate and marginalized
# Bayesian-linear-regression hyperparameters used by the retained experiments.
DGBO_DEVICE = "cuda"
DGBO_PARAMS = {
    "graph_hidden_dim": 48,
    "graph_layers": 5,
    "pool_hidden_dim": 50,
    "fc_layers": 5,
    "feature_dim": 45,
    "dropout": 0.0,
    "n_epochs": 200,
    "batch_size": 10,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "blr_hypers": "marginalize",
    "blr_normalize_x": True,
    "blr_normalize_y": True,
    "include_self_loops": True,
}


def build_bo(problem, config):
    from sb_arch_opt.algo.arch_sbo.models import check_dependencies
    from sb_arch_opt.algo.arch_sbo.models import ModelFactory

    check_dependencies()
    kpls_n_comp = config.kpls_n_dim if config.kpls_n_dim is not None and problem.n_var > config.kpls_n_dim else None
    surrogate_model, normalization = ModelFactory(problem).get_md_kriging_model(kpls_n_comp=kpls_n_comp)
    return build_sbo_algorithm(problem, config, surrogate_model, normalization)


def build_bo_ignore_hierarchy(problem, config):
    from sb_arch_opt.algo.arch_sbo.models import check_dependencies
    from sb_arch_opt.algo.arch_sbo.models import ModelFactory

    check_dependencies()
    surrogate_model, normalization = ModelFactory(problem).get_md_kriging_model(
        kpls_n_comp=None,
        ignore_hierarchy=True,
    )
    return build_sbo_algorithm(problem, config, surrogate_model, normalization)


def build_bo_encoded_krg(problem, config):
    from sb_arch_opt.algo.arch_sbo.models import check_dependencies
    from sb_arch_opt.algo.arch_sbo.models import ModelFactory

    check_dependencies()
    factory = ModelFactory(problem)
    normalization = factory.get_md_normalization()
    surrogate_model = ModelFactory.get_kriging_model(
        multi=True,
        kpls_n_comp=None,
        corr="squar_exp",
        theta0=[1e-2] * problem.n_var,
        hyper_opt="Cobyla",
    )
    surrogate_model.supports["x_hierarchy"] = False
    return build_sbo_algorithm(problem, config, surrogate_model, normalization)


def build_ga(problem, config):
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from sb_arch_opt.algo.pymoo_interface.api import get_nsga2, provision_pymoo

    pop_size = config.n_init if config.n_init is not None else config.batch_size
    n_offsprings = config.batch_size

    if problem.n_obj == 1:
        algorithm = GA(pop_size=pop_size, n_offsprings=n_offsprings)
        provision_pymoo(algorithm)
        return algorithm
    return get_nsga2(pop_size=pop_size, n_offsprings=n_offsprings)


def build_random(problem, config):
    from graph_bo.algorithms.random_search import RandomSearchAlgorithm
    from sb_arch_opt.algo.pymoo_interface.api import provision_pymoo

    algorithm = RandomSearchAlgorithm(
        pop_size=config.batch_size,
        init_size=config.n_init if config.n_init is not None else config.batch_size,
    )
    return provision_pymoo(algorithm)


def build_graph_kernel_surrogate(problem, structure_handlers):
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from sb_arch_opt.algo.arch_sbo.models import check_dependencies

    check_dependencies()
    gm_factory = GraphModelFactory(problem)
    return gm_factory.get_md_adsg_kriging_model(
        structure_kernels=structure_handlers,
        sizing_kernel=True,
        composition="additive",
        multi=True,
        ignore_hierarchy=False,
        enable_timing=False,
        poly="constant",
        corr="squar_exp",
    )


def build_mg_ld_wloa_adsg_semantic_type_surrogate(problem, config):
    from graph_bo.kernels import MultiGranularityLdWloa
    from graph_bo.kernels.extractors import ENCODING_LEVEL_DEPTHS
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from graph_bo.surrogates.surrogates import GraphKernelHandler

    granularities = [
        (level, ENCODING_LEVEL_DEPTHS[level])
        for level in ("adsg", "semantic_type")
    ]
    kernel = MultiGranularityLdWloa(
        cutoff=int(config.surrogate_params["cutoff"]),
        granularities=granularities,
    )
    return GraphModelFactory(problem).get_md_adsg_kriging_model(
        structure_kernels=[
            GraphKernelHandler(
                kernel=kernel,
                name="mg_ld_wloa_adsg_semantic_type",
            )
        ],
        sizing_kernel=True,
        sizing_kernel_mode=config.surrogate_params.get(
            "sizing_kernel_mode",
            "hierarchical",
        ),
        composition=config.surrogate_params.get("composition", "additive"),
        lambda_l2=float(config.surrogate_params["lambda_l2"]),
        multi=True,
        ignore_hierarchy=False,
        enable_timing=False,
        use_branch_theta0=True,
        poly="constant",
        corr="squar_exp",
        pow_exp_power=float(config.surrogate_params.get("pow_exp_power", 1.9)),
    )


def build_esp_semantic_type_surrogate(problem, config):
    from graph_bo.kernels import ShortestPathKernel
    from graph_bo.kernels.extractors import ENCODING_LEVEL_DEPTHS
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from graph_bo.surrogates.surrogates import GraphKernelHandler

    kernel = ShortestPathKernel(
        normalize=True,
        exponential=True,
        sigma0=float(config.surrogate_params["sigma0"]),
        directed=bool(config.surrogate_params["directed"]),
        depth_by_family=ENCODING_LEVEL_DEPTHS["semantic_type"],
    )
    return GraphModelFactory(problem).get_md_adsg_kriging_model(
        structure_kernels=[
            GraphKernelHandler(kernel=kernel, name="esp_semantic_type")
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


def _encoder_checkpoint(root: Path, problem: str, setup: str, seed: int, epoch: int) -> Path:
    filename = "last.pt" if int(epoch) >= 200 else f"epoch_{int(epoch):03d}.pt"
    return root / problem / setup / f"seed_{seed:03d}" / filename


def _encoder_checkpoint_pattern(root: Path, problem: str, setup: str, epoch: int) -> str:
    filename = "last.pt" if int(epoch) >= 200 else f"epoch_{int(epoch):03d}.pt"
    return str(root / problem / setup / "seed_<run_seed>" / filename)


def _make_run_seed_stable_config(surrogate_model, checkpoint_pattern: str):
    get_config = surrogate_model.get_config

    def stable_get_config():
        config = get_config()
        config["embedder"]["model_path"] = checkpoint_pattern
        config["embedder"]["seed"] = "<run_seed>"
        config["downstream"]["seed"] = "<run_seed>"
        if config.get("head") is not None:
            config["head"]["seed"] = "<run_seed>"
        return config

    return stable_get_config


def build_arch2vec_dngo(problem, config, run_seed: int):
    from graph_bo.gnn import Arch2VecEmbedder, DNGO
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from sb_arch_opt.algo.arch_sbo.models import check_dependencies

    if run_seed is None:
        raise ValueError("bo_gnn_dngo requires a run_seed (the seed-matched encoder/head seed).")
    check_dependencies()

    params = config.params
    checkpoint = _encoder_checkpoint(
        Path(params["encoder_results_root"]),
        config.problem,
        params["encoder_setup"],
        int(run_seed),
        int(params["encoder_epoch"]),
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found (seed {run_seed}): {checkpoint}")

    embedder = Arch2VecEmbedder(
        latent_dim=int(params["encoder_latent_dim"]),
        hidden_dim=int(params["encoder_hidden_dim"]),
        n_gin_layers=int(params["encoder_depth"]),
        norm=params["encoder_norm"],
        n_epochs=int(params["encoder_epoch"]),
        batch_size=int(params["encoder_batch_size"]),
        lr=float(params["encoder_lr"]),
        beta=float(params["encoder_beta"]),
        adj_cutoff=float(params["encoder_adj_cutoff"]),
        selection_window=int(params["encoder_selection_window"]),
        val_fraction=float(params["encoder_val_fraction"]),
        adj_decoder=params["encoder_adj_decoder"],
        model_path=checkpoint,
        embedding_batch_size=256,
        seed=int(run_seed),
    )
    downstream = DNGO(
        hidden_units=int(params["head_hidden_units"]),
        n_epochs=int(params["head_n_epochs"]),
        batch_size=int(params["head_batch_size"]),
        lr=float(params["head_lr"]),
        normalize_x=True,
        normalize_y=False,
        include_bias=False,
        hypers="optimize",
        calibrate_folds=int(params["head_calibrate_folds"]),
        seed=int(run_seed),
    )
    surrogate_model, normalization = GraphModelFactory(problem).get_md_adsg_gnn_model(
        embedder=embedder,
        downstream=downstream,
        reconstruction_adj_cutoff=float(params["reconstruction_adj_cutoff"]),
        multi=True,
        ignore_hierarchy=False,
    )
    surrogate_model.get_config = _make_run_seed_stable_config(
        surrogate_model,
        checkpoint_pattern=_encoder_checkpoint_pattern(
            Path(params["encoder_results_root"]),
            config.problem,
            params["encoder_setup"],
            int(params["encoder_epoch"]),
        ),
    )
    return build_sbo_algorithm(problem, config, surrogate_model, normalization)


def _make_seed_stable_config(surrogate_model):
    get_config = surrogate_model.get_config

    def stable_get_config():
        config = get_config()
        config["seed"] = "<run_seed>"
        return config

    return stable_get_config


def build_dgbo(problem, config, run_seed: int):
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from sb_arch_opt.algo.arch_sbo.models import check_dependencies

    if run_seed is None:
        raise ValueError("bo_dgbo requires a run_seed (the per-run net seed).")
    check_dependencies()

    surrogate_model, normalization = GraphModelFactory(problem).get_md_adsg_dgbo_model(
        multi=True,
        ignore_hierarchy=False,
        device=DGBO_DEVICE,
        seed=int(run_seed),
        **dict(config.surrogate_params),
    )
    surrogate_model.get_config = _make_seed_stable_config(surrogate_model)
    return build_sbo_algorithm(problem, config, surrogate_model, normalization)


def build_sbo_algorithm(problem, config, surrogate_model, normalization):
    from sb_arch_opt.algo.arch_sbo import get_sbo
    from sb_arch_opt.algo.arch_sbo.infill import (
        EnsembleInfill,
        FunctionEstimateConstrainedInfill,
        LowerConfidenceBoundInfill,
        MeanConstraintPrediction,
        MinVariancePFInfill,
        get_default_infill,
    )
    from sb_arch_opt.algo.arch_sbo.models import check_dependencies
    from sb_arch_opt.algo.arch_sbo.hc_strategy import get_hc_strategy

    check_dependencies()

    infill_kind = config.params.get("infill_kind", "mpoi")
    if infill_kind == "mpoi":
        infill, _ = get_default_infill(
            problem,
            n_parallel=config.batch_size,
            min_pof=config.params.get("min_pof"),
            g_aggregation=None,
        )
    elif infill_kind == "mean_lcb":
        infill = EnsembleInfill(
            infills=[
                FunctionEstimateConstrainedInfill(),
                LowerConfidenceBoundInfill(alpha=2.0),
            ],
            constraint_strategy=MeanConstraintPrediction(),
        )
    elif infill_kind == "mvpf":
        infill = MinVariancePFInfill(
            constraint_strategy=MeanConstraintPrediction(),
        )
    else:
        raise ValueError(f"Unknown infill kind: {infill_kind!r}")

    # For engine problem we want to handle hidden constraints
    hc_strategy = get_hc_strategy(kpls_n_dim=config.kpls_n_dim) if config.problem == "engine" else None

    return get_sbo(
        surrogate_model=surrogate_model,
        infill=infill,
        infill_size=config.batch_size,
        init_size=config.n_init,
        normalization=normalization,
        hc_strategy=hc_strategy,
    )


def build_surrogate_bo(problem, config, surrogate_model, normalization):
    return build_sbo_algorithm(problem, config, surrogate_model, normalization)
