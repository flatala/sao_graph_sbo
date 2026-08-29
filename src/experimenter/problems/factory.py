from pathlib import Path


def build_evaluator(problem_name: str, resources_root: Path):
    if problem_name == "mdgnc":
        from experimenter.problems.mdgnc import get_evaluator

        return get_evaluator(resources_root / "problems" / "mdgnc.adore")

    if problem_name == "mdgnc_edge_failures":
        from experimenter.problems.mdgnc_edge_failures import get_evaluator
        from experimenter.problems.mdgnc_encoding import force_mdgnc_encoding

        with force_mdgnc_encoding():
            return get_evaluator(resources_root / "problems" / "mdgnc_edge_failures.adore")

    if problem_name == "engine":
        from experimenter.problems.engine import SimpleJetEngineSurrogateEvaluator

        evaluator = SimpleJetEngineSurrogateEvaluator.from_file(
            str(resources_root / "problems" / "engine.adore")
        )
        evaluator.update_external_databases()
        evaluator.max_iter = 30
        evaluator.verbose = False
        evaluator.save_to_project = False
        return evaluator

    if problem_name == "rocket":
        from experimenter.problems.rocket import AdoreRocketEvaluator

        return AdoreRocketEvaluator.from_file(
            str(resources_root / "problems" / "rocket.adore"), save_to_project=True
        )

    raise ValueError(f"Unknown problem: {problem_name!r}")


def build_problem(problem_name: str, resources_root: Path, evaluator_parallelism: int = 1):
    from adore.optimization.bridge.problem import AdoreArchOptProblem

    problem = AdoreArchOptProblem(build_evaluator(problem_name, resources_root))
    problem.n_parallel = evaluator_parallelism
    return problem


def build_graph_processor(problem_name: str, resources_root: Path):
    return build_evaluator(problem_name, resources_root).translator.graph_processor
