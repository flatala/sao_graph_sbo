from __future__ import annotations

from typing import Any

import numpy as np
from adore.graph.adore_nodes import AttributeNode, AttributeValueNode, ComponentInstanceNode
from adsg_core.graph.adsg_nodes import ConnectionChoiceNode, ConnectorDegreeGroupingNode
from adsg_core.graph.graph_edges import EdgeType, get_edge_type, iter_in_edges, iter_out_edges
from adsg_core.optimization.graph_processor import GraphProcessor
from pymoo.util.normalization import Normalization

class ArchInstanceResolver:
    """ Shared converter and cache for the expensive common stuff pertaining to graph instances coming from DV's. """

    def __init__(self, graph_processor: GraphProcessor, normalization: Normalization):
        self.gp = graph_processor
        self.norm = normalization

        self._raw_to_corr: dict[tuple, tuple] = {}
        self._raw_to_is_active: dict[tuple, tuple[bool, ...]] = {}
        self._corr_to_adsg: dict[tuple, Any] = {}
        self._corr_to_attribute_choices: dict[tuple, dict[tuple[str, int, str], Any]] = {}

    def denormalize(self, x_row_norm) -> np.ndarray:
        """ Denormalize the dv to a raw dv. """
        x2d = np.asarray(x_row_norm, dtype=float).ravel()[None, :]
        raw2d = self.norm.backward(x2d)
        return raw2d[0]

    def correct_raw_dv(self, dv_raw) -> tuple[tuple, tuple[bool, ...]]:
        """ Get the corrected dv and is_active. Cache to avoid recomputation if encountered again. """
        raw_key = tuple(np.asarray(dv_raw, dtype=float).ravel())

        # Look in cache, otherwise compute
        if (dv_corr := self._raw_to_corr.get(raw_key)) is None or (is_active := self._raw_to_is_active.get(raw_key)) is None:
            _, dv_corr_arr, is_active_arr = self.gp.get_graph(np.asarray(dv_raw, dtype=float), create=False)
            dv_corr = tuple(dv_corr_arr)
            is_active = tuple(bool(v) for v in is_active_arr)
            self._raw_to_corr[raw_key] = dv_corr
            self._raw_to_is_active[raw_key] = is_active

        return dv_corr, is_active

    def correct_normalised_dv(self, dv_norm) -> tuple[tuple, tuple[bool, ...]]:
        raw_dv = self.denormalize(dv_norm)
        return self.correct_raw_dv(raw_dv)

    def adsg_from_normalised_dv(self, dv_norm):
        """ Denormalizes and corrects dv, gets the arch instance graph and caches it. """
        dv_corr, _ = self.correct_normalised_dv(dv_norm)
        return self.get_adsg_from_corrected_dv(dv_corr)

    def get_adsg_from_corrected_dv(self, dv_corr: tuple):
        """Get the arch instance graph from an already-corrected design vector."""
        adsg = self._corr_to_adsg.get(dv_corr)
        if adsg is None:
            adsg, _, _ = self.gp.get_graph(np.asarray(dv_corr, dtype=float), create=True)
            self._corr_to_adsg[dv_corr] = adsg
        return adsg

    @staticmethod
    def _get_attribute_connector_node(adsg, attr_node: AttributeNode):
        group_nodes = [
            edge[1] for edge in iter_out_edges(adsg._graph, attr_node)
            if get_edge_type(edge) == EdgeType.DERIVES and isinstance(edge[1], ConnectorDegreeGroupingNode)
        ]

        return group_nodes[0] if len(group_nodes) == 1 else attr_node

    def get_attribute_schema(self) -> list[tuple[ComponentInstanceNode, AttributeNode, ConnectionChoiceNode, tuple[Any, ...]]]:
        graph = self.gp.graph
        schema: list[tuple[ComponentInstanceNode, AttributeNode, ConnectionChoiceNode, tuple[Any, ...]]] = []
        instance_nodes = sorted(
            [node for node in graph._graph.nodes if isinstance(node, ComponentInstanceNode)],
            key=lambda node: (node.comp_name, node.idx),
        )

        for instance_node in instance_nodes:
            # noinspection PyUnresolvedReferences
            for attr_node in graph.next(instance_node):
                if not isinstance(attr_node, AttributeNode) or not attr_node.is_inst:
                    continue

                connector_node = self._get_attribute_connector_node(graph, attr_node)
                decision_nodes = [
                    edge[1] for edge in iter_out_edges(graph._graph, connector_node)
                    if get_edge_type(edge) == EdgeType.CONNECTS and isinstance(edge[1], ConnectionChoiceNode)
                ]

                option_nodes = [
                    edge[1] for edge in iter_out_edges(graph._graph, decision_nodes[0])
                    if get_edge_type(edge) == EdgeType.CONNECTS and isinstance(edge[1], AttributeValueNode)
                ]

                option_nodes = ConnectionChoiceNode.get_sorted_connector_nodes(option_nodes)
                schema.append((instance_node, attr_node, decision_nodes[0], tuple(node.value for node in option_nodes)))

        return schema

    def get_attribute_choices(self, dv_corr: tuple) -> dict[tuple[str, int, str], Any]:
        cached = self._corr_to_attribute_choices.get(dv_corr)
        if cached is not None:
            return cached

        adsg = self.get_adsg_from_corrected_dv(dv_corr)
        attribute_choices: dict[tuple[str, int, str], Any] = {}
        instance_nodes = sorted(
            [node for node in adsg._graph.nodes if isinstance(node, ComponentInstanceNode)],
            key=lambda node: (node.comp_name, node.idx),
        )

        for instance_node in instance_nodes:
            for attr_node in adsg.next(instance_node):
                if not isinstance(attr_node, AttributeNode) or not attr_node.is_inst:
                    continue

                connector_node = self._get_attribute_connector_node(adsg, attr_node)
                if isinstance(connector_node, ConnectorDegreeGroupingNode):
                    source_attr_nodes = [
                        edge[0]
                        for edge in iter_in_edges(adsg._graph, connector_node)
                        if get_edge_type(edge) == EdgeType.DERIVES and isinstance(edge[0], AttributeNode)
                    ]
                    target_value_nodes = [
                        edge[1]
                        for edge in iter_out_edges(adsg._graph, connector_node)
                        if get_edge_type(edge) == EdgeType.CONNECTS and isinstance(edge[1], AttributeValueNode)
                    ]
                    source_attr_nodes = ConnectionChoiceNode.get_sorted_connector_nodes(source_attr_nodes)
                    target_value_nodes = ConnectionChoiceNode.get_sorted_connector_nodes(target_value_nodes)
                    chosen_value_nodes = [target_value_nodes[source_attr_nodes.index(attr_node)]]
                else:
                    chosen_value_nodes = [
                        edge[1]
                        for edge in iter_out_edges(adsg._graph, attr_node)
                        if get_edge_type(edge) == EdgeType.CONNECTS and isinstance(edge[1], AttributeValueNode)
                    ]
                    chosen_value_nodes = ConnectionChoiceNode.get_sorted_connector_nodes(chosen_value_nodes)

                if len(chosen_value_nodes) == 0:
                    continue

                attribute_choices[(instance_node.comp_name, instance_node.idx, attr_node.key)] = chosen_value_nodes[0].value

        self._corr_to_attribute_choices[dv_corr] = attribute_choices
        return attribute_choices
