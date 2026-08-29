from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from adsg_core.optimization.graph_processor import GraphProcessor


MDGNC_PROBLEMS = {"mdgnc", "mdgnc_edge_failures"}
MDGNC_FORCED_CONNECTION_ENCODERS = {
    "port_COMMAND": "element_grouped",
    "port_DATA": "lazy_conn_idx",
}


class MDGNCGraphProcessorFactory:
    def __call__(self, adore_graph, encoding_timeout=None, encoder_type=None):
        from adore.graph.api.design_problem import SelChoiceEncoderType

        return MDGNCGraphProcessor(
            adore_graph,
            encoding_timeout=encoding_timeout,
            encoder_type=SelChoiceEncoderType.COMPLETE,
        )


@contextmanager
def force_mdgnc_encoding():
    from adore.graph.api.design_problem import DesignProblemTranslator
    from adsg_core.optimization.assign_enc.matrix import AggregateAssignmentMatrixGenerator

    old_factory = DesignProblemTranslator.gp_factory
    old_write_to_cache = AggregateAssignmentMatrixGenerator._write_to_cache
    DesignProblemTranslator.gp_factory = MDGNCGraphProcessorFactory()
    AggregateAssignmentMatrixGenerator._write_to_cache = lambda self, cache_path, value: None
    try:
        yield
    finally:
        DesignProblemTranslator.gp_factory = old_factory
        AggregateAssignmentMatrixGenerator._write_to_cache = old_write_to_cache


def get_adsg_encoding_metadata(problem) -> dict[str, Any]:
    gp = problem.evaluator._translator.graph_processor
    _ = gp.all_des_vars
    connection_encoders = {}
    for node, data in gp._conn_choice_data_map.items():
        assignment_manager = data[0]
        connection_encoders[str(node)] = {
            "assignment_manager": type(assignment_manager).__name__,
            "encoder": type(assignment_manager.encoder).__name__,
            "design_variable_options": [int(des_var.n_opts) for des_var in assignment_manager.design_vars],
        }

    return {
        "n_var": int(problem.n_var),
        "encoder_type": str(gp.encoder_type),
        "design_variables": [str(des_var) for des_var in gp.all_des_vars],
        "connection_encoders": connection_encoders,
    }


class MDGNCGraphProcessorMixin:
    def _encode_connection_choice(self, connection_choice_node):
        if connection_choice_node.decision_id not in MDGNC_FORCED_CONNECTION_ENCODERS:
            return super()._encode_connection_choice(connection_choice_node)

        from adsg_core.optimization.dv_output_defs import DesVar

        settings, node_map, existence_map, all_conn_nodes = connection_choice_node.get_assignment_encoding_args(
            self.graph,
            hierarchy_analyzer=self._hierarchy_analyzer,
        )
        assignment_manager = _build_assignment_manager(connection_choice_node.decision_id, settings)

        n_matrix_map = assignment_manager.get_n_matrices_by_existence(cache=True)
        for i_exist, existence in enumerate(assignment_manager.matrix_gen.existence_patterns.patterns):
            if n_matrix_map[existence] == 0:
                if isinstance(existence_map, dict):
                    existence_map = {
                        exist_mask: -1 if i_pattern == i_exist else i_pattern
                        for exist_mask, i_pattern in existence_map.items()
                    }
                else:
                    existence_map[existence_map == i_exist] = -1

        base_name = connection_choice_node.decision_id
        des_vars = []
        for i, des_var in enumerate(assignment_manager.design_vars):
            options = list(range(des_var.n_opts))
            name = f"{base_name}_{i}"
            des_vars.append(
                DesVar.from_choice_node(
                    connection_choice_node,
                    options=options,
                    name=name,
                    conditionally_active=des_var.conditionally_active,
                )
            )

        return assignment_manager, des_vars, node_map, existence_map, all_conn_nodes


def _build_assignment_manager(decision_id: str, settings):
    from adsg_core.optimization.assign_enc.assignment_manager import AssignmentManager, LazyAssignmentManager
    from adsg_core.optimization.assign_enc.eager.encodings.group_element import ElementGroupedEncoder
    from adsg_core.optimization.assign_enc.encoder_registry import DEFAULT_EAGER_IMPUTER, DEFAULT_LAZY_IMPUTER
    from adsg_core.optimization.assign_enc.lazy.encodings.conn_idx import (
        FlatConnCombsEncoder,
        LazyConnIdxMatrixEncoder,
    )

    if decision_id == "port_COMMAND":
        return AssignmentManager(settings, ElementGroupedEncoder(DEFAULT_EAGER_IMPUTER()), cache=False)

    if decision_id == "port_DATA":
        return LazyAssignmentManager(
            settings,
            LazyConnIdxMatrixEncoder(DEFAULT_LAZY_IMPUTER(), FlatConnCombsEncoder(), by_src=False),
        )

    raise ValueError(f"Unsupported forced MDGNC connection decision: {decision_id!r}")


class MDGNCGraphProcessor(MDGNCGraphProcessorMixin, GraphProcessor):
    pass
