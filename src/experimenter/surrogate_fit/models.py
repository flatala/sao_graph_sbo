from __future__ import annotations


def build_md_kriging(problem, kpls_n_comp: int | None):
    from sb_arch_opt.algo.arch_sbo.models import ModelFactory, check_dependencies

    check_dependencies()
    if kpls_n_comp is not None and problem.n_var <= kpls_n_comp:
        kpls_n_comp = None
    return ModelFactory(problem).get_md_kriging_model(kpls_n_comp=kpls_n_comp)


def build_blr_activeness(problem):
    from sb_arch_opt.algo.arch_sbo.models import ModelFactory
    from graph_bo.surrogates.bayes_linear import BayesianLinearSurrogate

    factory = ModelFactory(problem)
    normalization = factory.get_md_normalization()
    evaluator = problem.evaluator
    output_names = (
        [obj.ref if obj.ref is not None else obj.name for obj in evaluator.objectives]
        + [con.ref if con.ref is not None else con.name for con in evaluator.constraints]
    )
    surrogate_model = BayesianLinearSurrogate(output_names=output_names)
    return surrogate_model, normalization


def build_esp(
    problem,
    *,
    params: dict,
):
    from graph_bo.kernels import EdgeWeightKernel, ShortestPathKernel
    from graph_bo.kernels.extractors import ENCODING_LEVEL_DEPTHS
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from graph_bo.surrogates.surrogates import GraphKernelHandler
    from sb_arch_opt.algo.arch_sbo.models import check_dependencies

    check_dependencies()
    structure_kernels = [
        GraphKernelHandler(
            kernel=ShortestPathKernel(
                normalize=True,
                exponential=True,
                sigma0=float(params["sigma0"]),
                directed=bool(params["directed"]),
                depth_by_family=ENCODING_LEVEL_DEPTHS[str(params["encoding_level"])],
            ),
            name=f"esp_{params['encoding_level']}",
        )
    ]
    if bool(params["edge_weight"]):
        structure_kernels.append(
            GraphKernelHandler(
                kernel=EdgeWeightKernel(gamma0=1.0),
                name="edge_weight",
            )
        )
    return GraphModelFactory(problem).get_md_adsg_kriging_model(
        structure_kernels=structure_kernels,
        sizing_kernel=True,
        sizing_kernel_mode=str(params["sizing_kernel_mode"]),
        composition=str(params["composition"]),
        lambda_l2=float(params["lambda_l2"]),
        multi=True,
        ignore_hierarchy=False,
        enable_timing=True,
        use_branch_theta0=True,
        poly="constant",
        corr="squar_exp",
        pow_exp_power=float(params["pow_exp_power"]),
    )


def build_mg_ld_wloa(
    problem,
    *,
    params: dict,
):
    from graph_bo.kernels import EdgeWeightKernel, MultiGranularityLdWloa
    from graph_bo.kernels.extractors import ENCODING_LEVEL_DEPTHS
    from graph_bo.surrogates.model_factory import GraphModelFactory
    from graph_bo.surrogates.surrogates import GraphKernelHandler
    from sb_arch_opt.algo.arch_sbo.models import check_dependencies

    check_dependencies()
    granularities = [
        (level, ENCODING_LEVEL_DEPTHS[level])
        for level in params["encoding_levels"]
    ]
    structure_kernels = [
        GraphKernelHandler(
            kernel=MultiGranularityLdWloa(
                cutoff=int(params["cutoff"]),
                granularities=granularities,
            ),
            name="mg_ld_wloa_adsg_semantic_type",
        )
    ]
    if bool(params["edge_weight"]):
        structure_kernels.append(
            GraphKernelHandler(
                kernel=EdgeWeightKernel(gamma0=1.0),
                name="edge_weight",
            )
        )
    return GraphModelFactory(problem).get_md_adsg_kriging_model(
        structure_kernels=structure_kernels,
        sizing_kernel=True,
        sizing_kernel_mode=str(params["sizing_kernel_mode"]),
        composition=str(params["composition"]),
        lambda_l2=float(params["lambda_l2"]),
        multi=True,
        ignore_hierarchy=False,
        enable_timing=True,
        use_branch_theta0=True,
        poly="constant",
        corr="squar_exp",
        pow_exp_power=float(params["pow_exp_power"]),
    )
