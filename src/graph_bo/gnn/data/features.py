from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
from adore.graph.adore_nodes import (
    AttributeNode,
    AttributeValueNode,
    ComponentInstanceGroupNode,
    ComponentInstanceNode,
    ComponentNode,
    DesignVariableNode,
)
from adsg_core.graph.adsg_nodes import ConnectionChoiceNode, ConnectorDegreeGroupingNode
from adsg_core.graph.graph_edges import EdgeType, get_edge_type, iter_in_edges, iter_out_edges
from adsg_core.optimization.graph_processor import GraphProcessor


_BASE_NODE_TYPES = (
    "attribute",
    "attribute_value",
    "component",
    "component_instance",
    "component_instance_group",
    "connector_group",
    "design_variable",
    "external",
    "function",
    "function_decomposition",
    "group",
    "metric",
    "needed_port",
    "provided_port",
)


def node_family(node) -> str:
    class_name = type(node).__name__
    if class_name in {"ExternalConnectionNode", "ExternalOutConnectionNode"}:
        return "external"
    if class_name == "FunctionDecompositionNode":
        return "function_decomposition"
    if class_name == "FunctionNode":
        return "function"
    if class_name == "ComponentInstanceNode":
        return "component_instance"
    if class_name == "ComponentInstanceGroupNode":
        return "component_instance_group"
    if class_name == "ComponentNode":
        return "component"
    if class_name == "ProvidedPortNode":
        return "provided_port"
    if class_name == "NeededPortNode":
        return "needed_port"
    if class_name == "AttributeValueNode":
        return "attribute_value"
    if class_name == "AttributeNode":
        return "attribute"
    if class_name == "MetricNode":
        return "metric"
    if class_name == "DesignVariableNode":
        return "design_variable"
    if class_name == "ConnectorDegreeGroupingNode":
        return "connector_group"
    if class_name == "GroupNode":
        return "group"
    return class_name


@dataclass(frozen=True)
class ValueFeatureSpec:
    dv_index: int
    node: DesignVariableNode | AttributeNode
    row_index: int
    value_key: tuple
    value_index: int
    kind: str
    bounds: tuple[float, float] | None
    n_options: int | None
    source: str = "design_variable"
    attribute_comp_name: str | None = None
    attribute_instance_idx: int | None = None
    attribute_key: str | None = None
    attribute_option_value: object | None = None
    attribute_option_values: tuple | None = None
    dv_option_index: int | None = None


class ADSGFeatureExtractor(ABC):
    feature_dim: int
    node_labels: np.ndarray
    X_parent: np.ndarray
    binary_feature_indices: tuple[int, ...]
    continuous_feature_indices: tuple[int, ...]
    n_component_types: int
    n_value_features: int
    dv_specs: tuple[ValueFeatureSpec, ...]
    attribute_specs: tuple[ValueFeatureSpec, ...]

    def __init__(self, graph_processor: GraphProcessor, nodes: Sequence, parent_nx: nx.MultiDiGraph):
        self.graph_processor = graph_processor
        self.nodes = list(nodes)
        self.parent_nx = parent_nx

    @property
    @abstractmethod
    def inactive_token(self) -> np.ndarray:
        ...

    @abstractmethod
    def encode_instance(
        self,
        adsg_instance,
        active_mask: np.ndarray,
        corrected_vector: Sequence[float | int],
        is_active: Sequence[bool],
    ) -> np.ndarray:
        ...

    def summary(self) -> dict[str, int]:
        return {
            "feature_dim": self.feature_dim,
            "n_binary_features": len(self.binary_feature_indices),
            "n_continuous_features": len(self.continuous_feature_indices),
            "n_component_types": self.n_component_types,
            "n_value_features": self.n_value_features,
            "n_dv_value_specs": len(self.dv_specs),
            "n_attribute_value_specs": len(self.attribute_specs),
        }

    def print_summary(self) -> None:
        for key, value in self.summary().items():
            print(f"{key} = {value}")


class ADSGNodeFeatureExtractor(ADSGFeatureExtractor):
    def __init__(self, graph_processor: GraphProcessor, nodes: Sequence, parent_nx: nx.MultiDiGraph):
        super().__init__(graph_processor, nodes, parent_nx)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        self._families = tuple(sorted(set(_BASE_NODE_TYPES) | {node_family(node) for node in self.nodes}))
        self._family_to_idx = {family: i for i, family in enumerate(self._families)}

        component_types = sorted(
            {
                comp_type
                for node in self.nodes
                for comp_type in self._component_types_for_node(node)
            }
        )
        self._component_types = tuple(component_types)
        self._component_type_to_idx = {comp_type: i for i, comp_type in enumerate(self._component_types)}

        raw_value_specs = self._build_raw_dv_value_specs()
        raw_value_specs.extend(self._build_raw_attribute_value_specs())
        self._value_keys = tuple(
            sorted(
                {spec["value_key"] for spec in raw_value_specs},
                key=lambda value_key: tuple(str(part) for part in value_key),
            )
        )

        self.family_slice = slice(0, len(self._families))
        self.component_type_slice = slice(self.family_slice.stop, self.family_slice.stop + len(self._component_types))
        self.value_slice = slice(self.component_type_slice.stop, self.component_type_slice.stop + len(self._value_keys))
        self.feature_dim = self.value_slice.stop

        value_key_to_index = {
            value_key: self.value_slice.start + i
            for i, value_key in enumerate(self._value_keys)
        }
        self.value_specs = tuple(
            ValueFeatureSpec(
                dv_index=spec["dv_index"],
                node=spec["node"],
                row_index=spec["row_index"],
                value_key=spec["value_key"],
                value_index=value_key_to_index[spec["value_key"]],
                kind=spec["kind"],
                bounds=spec["bounds"],
                n_options=spec["n_options"],
                source=spec["source"],
                attribute_comp_name=spec.get("attribute_comp_name"),
                attribute_instance_idx=spec.get("attribute_instance_idx"),
                attribute_key=spec.get("attribute_key"),
                attribute_option_value=spec.get("attribute_option_value"),
                attribute_option_values=spec.get("attribute_option_values"),
                dv_option_index=spec.get("dv_option_index"),
            )
            for spec in raw_value_specs
        )
        self.dv_specs = tuple(spec for spec in self.value_specs if spec.source == "design_variable")
        self.attribute_specs = tuple(spec for spec in self.value_specs if spec.source == "attribute")
        self.binary_feature_indices = tuple(
            range(self.family_slice.start, self.component_type_slice.stop)
        ) + tuple(
            sorted({spec.value_index for spec in self.value_specs if spec.kind in {"categorical", "ordinal"}})
        )
        self.continuous_feature_indices = tuple(
            sorted({spec.value_index for spec in self.value_specs if spec.kind == "continuous"})
        )

        self.node_labels = np.array([self._family_to_idx[node_family(node)] for node in self.nodes], dtype=np.int64)
        self.X_parent = self._encode_parent()

    @property
    def n_semantic_labels(self) -> int:
        return 0

    @property
    def n_component_types(self) -> int:
        return len(self._component_types)

    @property
    def n_value_features(self) -> int:
        return len(self._value_keys)

    @property
    def value_feature_names(self) -> tuple[str, ...]:
        return tuple(".".join(str(part) for part in value_key[1:]) for value_key in self._value_keys)

    @property
    def inactive_token(self) -> np.ndarray:
        return np.zeros((self.feature_dim,), dtype=np.float32)

    def encode_instance(
        self,
        adsg_instance,
        active_mask: np.ndarray,
        corrected_vector: Sequence[float | int],
        is_active: Sequence[bool],
    ) -> np.ndarray:
        X = self.X_parent.copy()
        X[active_mask == 0.0] = self.inactive_token

        for spec in self.dv_specs:
            if not is_active[spec.dv_index]:
                continue
            if active_mask[spec.row_index] == 0.0:
                raise ValueError(f"Active design variable node is not present in the ADSG: {spec.node!r}")

            if spec.kind == "ordinal":
                value = int(corrected_vector[spec.dv_index])
                if value > spec.dv_option_index:
                    X[spec.row_index, spec.value_index] = 1.0
            elif spec.kind == "categorical":
                if int(corrected_vector[spec.dv_index]) == spec.dv_option_index:
                    X[spec.row_index, spec.value_index] = 1.0
            else:
                X[spec.row_index, spec.value_index] = self._normalize_value(corrected_vector[spec.dv_index], spec)

        attribute_choices = self._attribute_choices(adsg_instance)
        for spec in self.attribute_specs:
            if active_mask[spec.row_index] == 0.0:
                continue
            choice_key = (spec.attribute_comp_name, spec.attribute_instance_idx, spec.attribute_key)
            if choice_key not in attribute_choices:
                continue
            choice_value = attribute_choices.get(choice_key)
            if spec.kind == "ordinal":
                option_index = spec.attribute_option_values.index(choice_value)
                if option_index > spec.dv_option_index:
                    X[spec.row_index, spec.value_index] = 1.0
            elif choice_value == spec.attribute_option_value:
                X[spec.row_index, spec.value_index] = 1.0

        return X

    def _build_raw_dv_value_specs(self) -> list[dict]:
        specs: list[dict] = []
        for dv_index, des_var in enumerate(self.graph_processor.des_vars):
            node = des_var.node
            if not isinstance(node, DesignVariableNode):
                continue
            if node not in self.node_to_idx:
                raise ValueError(f"Design variable node is not in the parent ADSG node set: {node!r}")

            if des_var.is_discrete:
                kind = "ordinal" if des_var.is_ordinal else "categorical"
                bounds = None
                n_options = des_var.n_opts
            else:
                kind = "continuous"
                lower, upper = des_var.bounds
                bounds = (float(lower), float(upper))
                n_options = None

            if kind in {"categorical", "ordinal"}:
                n_encoded_options = n_options if kind == "categorical" else max(n_options - 1, 1)
                for option_index in range(n_encoded_options):
                    specs.append(
                        {
                            "dv_index": dv_index,
                            "node": node,
                            "row_index": self.node_to_idx[node],
                            "value_key": (*self._value_key(node), option_index),
                            "kind": kind,
                            "bounds": bounds,
                            "n_options": n_options,
                            "source": "design_variable",
                            "dv_option_index": option_index,
                        }
                    )
            else:
                specs.append(
                    {
                        "dv_index": dv_index,
                        "node": node,
                        "row_index": self.node_to_idx[node],
                        "value_key": self._value_key(node),
                        "kind": kind,
                        "bounds": bounds,
                        "n_options": n_options,
                        "source": "design_variable",
                    }
                )
        return specs

    def _build_raw_attribute_value_specs(self) -> list[dict]:
        specs: list[dict] = []
        graph = self.graph_processor.graph
        instance_nodes = sorted(
            [node for node in graph._graph.nodes if isinstance(node, ComponentInstanceNode)],
            key=lambda node: (node.comp_name, node.idx),
        )

        for instance_node in instance_nodes:
            for attr_node in graph.next(instance_node):
                if not isinstance(attr_node, AttributeNode) or not attr_node.is_inst:
                    continue
                if attr_node not in self.node_to_idx:
                    continue
                comp_name = self._canonical_component_type(instance_node.comp_name)

                connector_node = self._get_attribute_connector_node(graph._graph, attr_node)
                decision_nodes = [
                    edge[1]
                    for edge in iter_out_edges(graph._graph, connector_node)
                    if get_edge_type(edge) == EdgeType.CONNECTS and isinstance(edge[1], ConnectionChoiceNode)
                ]
                option_nodes = [
                    edge[1]
                    for edge in iter_out_edges(graph._graph, decision_nodes[0])
                    if get_edge_type(edge) == EdgeType.CONNECTS and isinstance(edge[1], AttributeValueNode)
                ]
                option_nodes = ConnectionChoiceNode.get_sorted_connector_nodes(option_nodes)

                kind = "ordinal" if decision_nodes[0].is_ordinal else "categorical"
                n_encoded_options = len(option_nodes) if kind == "categorical" else max(len(option_nodes) - 1, 1)
                for option_index, option_node in enumerate(option_nodes[:n_encoded_options]):
                    value_key_suffix = option_index if kind == "ordinal" else option_node.value
                    specs.append(
                        {
                            "dv_index": -1,
                            "node": attr_node,
                            "row_index": self.node_to_idx[attr_node],
                            "value_key": ("attribute", comp_name, attr_node.key, value_key_suffix),
                            "kind": kind,
                            "bounds": None,
                            "n_options": len(option_nodes),
                            "source": "attribute",
                            "attribute_comp_name": comp_name,
                            "attribute_instance_idx": instance_node.idx,
                            "attribute_key": attr_node.key,
                            "attribute_option_value": option_node.value,
                            "attribute_option_values": tuple(node.value for node in option_nodes),
                            "dv_option_index": option_index,
                        }
                    )
        return specs

    def _encode_parent(self) -> np.ndarray:
        X = np.zeros((len(self.nodes), self.feature_dim), dtype=np.float32)

        for i, node in enumerate(self.nodes):
            family = node_family(node)
            X[i, self.family_slice.start + self._family_to_idx[family]] = 1.0

            for comp_type in self._component_types_for_node(node):
                X[i, self.component_type_slice.start + self._component_type_to_idx[comp_type]] = 1.0

        return X

    def _value_key(self, node: DesignVariableNode) -> tuple[str, str, str]:
        component_types = self._component_types_for_node(node)
        if len(component_types) != 1:
            raise ValueError(f"Design variable node must belong to exactly one component type: {node!r}")
        return "design_variable", component_types[0], node.name

    @staticmethod
    def _get_attribute_connector_node(graph: nx.MultiDiGraph, attr_node: AttributeNode):
        group_nodes = [
            edge[1]
            for edge in iter_out_edges(graph, attr_node)
            if get_edge_type(edge) == EdgeType.DERIVES and isinstance(edge[1], ConnectorDegreeGroupingNode)
        ]
        return group_nodes[0] if len(group_nodes) == 1 else attr_node

    def _attribute_choices(self, adsg_instance) -> dict[tuple[str, int, str], object]:
        if len(self.attribute_specs) == 0:
            return {}

        choices: dict[tuple[str, int, str], object] = {}
        graph = adsg_instance._graph
        instance_nodes = sorted(
            [node for node in graph.nodes if isinstance(node, ComponentInstanceNode)],
            key=lambda node: (node.comp_name, node.idx),
        )

        for instance_node in instance_nodes:
            for attr_node in adsg_instance.next(instance_node):
                if not isinstance(attr_node, AttributeNode) or not attr_node.is_inst:
                    continue

                connector_node = self._get_attribute_connector_node(graph, attr_node)
                if isinstance(connector_node, ConnectorDegreeGroupingNode):
                    source_attr_nodes = [
                        edge[0]
                        for edge in iter_in_edges(graph, connector_node)
                        if get_edge_type(edge) == EdgeType.DERIVES and isinstance(edge[0], AttributeNode)
                    ]
                    target_value_nodes = [
                        edge[1]
                        for edge in iter_out_edges(graph, connector_node)
                        if get_edge_type(edge) == EdgeType.CONNECTS and isinstance(edge[1], AttributeValueNode)
                    ]
                    source_attr_nodes = ConnectionChoiceNode.get_sorted_connector_nodes(source_attr_nodes)
                    target_value_nodes = ConnectionChoiceNode.get_sorted_connector_nodes(target_value_nodes)
                    chosen_value_nodes = [target_value_nodes[source_attr_nodes.index(attr_node)]]
                else:
                    chosen_value_nodes = [
                        edge[1]
                        for edge in iter_out_edges(graph, attr_node)
                        if get_edge_type(edge) == EdgeType.CONNECTS and isinstance(edge[1], AttributeValueNode)
                    ]
                    chosen_value_nodes = ConnectionChoiceNode.get_sorted_connector_nodes(chosen_value_nodes)

                if len(chosen_value_nodes) == 0:
                    continue
                comp_name = self._canonical_component_type(instance_node.comp_name)
                choices[(comp_name, instance_node.idx, attr_node.key)] = chosen_value_nodes[0].value

        return choices

    def _component_types_for_node(self, node) -> tuple[str, ...]:
        component_types: set[str] = set()
        if isinstance(node, ComponentInstanceNode):
            component_types.add(self._canonical_component_type(node.comp_name))
        if isinstance(node, ComponentInstanceGroupNode):
            component_types.add(self._canonical_component_type(node.comp_name))
        if isinstance(node, ComponentNode):
            component_types.add(self._canonical_component_type(node.name))

        visited = {node}
        frontier = [node]
        while frontier:
            current = frontier.pop(0)
            for pred in self.parent_nx.predecessors(current):
                if pred in visited:
                    continue
                visited.add(pred)
                if isinstance(pred, ComponentInstanceNode):
                    component_types.add(self._canonical_component_type(pred.comp_name))
                    continue
                if isinstance(pred, ComponentInstanceGroupNode):
                    component_types.add(self._canonical_component_type(pred.comp_name))
                    continue
                if isinstance(pred, ComponentNode):
                    component_types.add(self._canonical_component_type(pred.name))
                    continue
                frontier.append(pred)

        return tuple(sorted(component_types))

    @staticmethod
    def _canonical_component_type(comp_name: str) -> str:
        return re.sub(r"(_\d+)+$", "", comp_name)

    @staticmethod
    def _normalize_value(value: float | int, spec: ValueFeatureSpec) -> float:
        if spec.bounds is None:
            raise ValueError(f"Continuous design variable has no bounds: {spec.node!r}")
        lower, upper = spec.bounds
        scale = upper - lower
        if scale <= 0.0:
            raise ValueError(f"Invalid design variable bounds: {spec.node!r}")
        return (float(value) - lower) / scale


class ADSGNodeTypeFeatureExtractor(ADSGFeatureExtractor):
    def __init__(self, graph_processor: GraphProcessor, nodes: Sequence, parent_nx: nx.MultiDiGraph):
        super().__init__(graph_processor, nodes, parent_nx)

        self._families = tuple(sorted(set(_BASE_NODE_TYPES) | {node_family(node) for node in self.nodes}))
        self._family_to_idx = {family: i for i, family in enumerate(self._families)}
        self.family_slice = slice(0, len(self._families))
        self.component_type_slice = slice(self.family_slice.stop, self.family_slice.stop)
        self.value_slice = slice(self.component_type_slice.stop, self.component_type_slice.stop)
        self.feature_dim = self.family_slice.stop

        self.value_specs = ()
        self.dv_specs = ()
        self.attribute_specs = ()
        self.binary_feature_indices = tuple(range(self.feature_dim))
        self.continuous_feature_indices = ()

        self.node_labels = np.array([self._family_to_idx[node_family(node)] for node in self.nodes], dtype=np.int64)
        self.X_parent = self._encode_parent()

    @property
    def n_semantic_labels(self) -> int:
        return 0

    @property
    def n_component_types(self) -> int:
        return 0

    @property
    def n_value_features(self) -> int:
        return 0

    @property
    def value_feature_names(self) -> tuple[str, ...]:
        return ()

    @property
    def inactive_token(self) -> np.ndarray:
        return np.zeros((self.feature_dim,), dtype=np.float32)

    def encode_instance(
        self,
        adsg_instance,
        active_mask: np.ndarray,
        corrected_vector: Sequence[float | int],
        is_active: Sequence[bool],
    ) -> np.ndarray:
        X = self.X_parent.copy()
        X[active_mask == 0.0] = self.inactive_token
        return X

    def _encode_parent(self) -> np.ndarray:
        X = np.zeros((len(self.nodes), self.feature_dim), dtype=np.float32)
        for i, node in enumerate(self.nodes):
            X[i, self._family_to_idx[node_family(node)]] = 1.0
        return X


DVFeatureSpec = ValueFeatureSpec
NodeFeatureExtractor = ADSGNodeFeatureExtractor
