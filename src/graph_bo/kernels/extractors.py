from collections import defaultdict
from enum import Enum

import networkx as nx

from adore.graph.adore_nodes import *
from adsg_core.graph.adsg_nodes import *


class EncodingDepth(str, Enum):
    CONSTANT = "constant"  # all nodes -> same label
    DSG = "dsg"            # generic DSG role
    ADSG = "adsg"          # architectural role
    TYPE = "type"          # semantic type, without instance identity
    FAMILY = "family"      # legacy architectural family
    SEMANTIC = "semantic"  # legacy semantic label

CONSTANT_DEPTH_BY_FAMILY = {"*": EncodingDepth.CONSTANT}
DSG_DEPTH_BY_FAMILY = {"*": EncodingDepth.DSG}
ADSG_DEPTH_BY_FAMILY = {"*": EncodingDepth.ADSG}
TYPE_DEPTH_BY_FAMILY = {"*": EncodingDepth.TYPE}
ENCODING_LEVEL_DEPTHS = {
    "constant": CONSTANT_DEPTH_BY_FAMILY,
    "dsg": DSG_DEPTH_BY_FAMILY,
    "adsg": ADSG_DEPTH_BY_FAMILY,
    "semantic_type": TYPE_DEPTH_BY_FAMILY,
}

def node_family(node) -> str:
    if isinstance(node, (ExternalConnectionNode, ExternalOutConnectionNode)):
        return "external"
    if isinstance(node, FunctionDecompositionNode):
        return "function_decomposition"
    if isinstance(node, FunctionNode):
        return "function"
    if isinstance(node, ComponentInstanceNode):
        return "component_instance"
    if isinstance(node, ComponentInstanceGroupNode):
        return "component_instance_group"
    if isinstance(node, ComponentNode):
        return "component"
    if isinstance(node, ProvidedPortNode):
        return "provided_port"
    if isinstance(node, NeededPortNode):
        return "needed_port"
    if isinstance(node, AttributeValueNode):
        return "attribute_value"
    if isinstance(node, AttributeNode):
        return "attribute"
    if isinstance(node, MetricNode):
        return "metric"
    if isinstance(node, DesignVariableNode):
        return "design_variable"
    if isinstance(node, ConnectorDegreeGroupingNode):
        return "connector_group"
    if isinstance(node, GroupNode):
        return "group"
    return type(node).__name__

def family_node_key(node) -> tuple[str]:
    return (node_family(node),)

def constant_node_key(_node) -> tuple[str]:
    return ("node",)

def dsg_node_key(node) -> tuple[str]:
    """Encode the most specific role available in the generic DSG model."""
    if isinstance(node, SelectionChoiceNode):
        return ("selection_choice",)
    if isinstance(node, ConnectionChoiceNode):
        return ("connection_choice",)
    if isinstance(node, ChoiceNode):
        return ("choice",)
    if isinstance(node, DesignVariableNode):
        return ("design_variable",)
    if isinstance(node, MetricNode):
        return ("metric",)
    if isinstance(node, ConnectorDegreeGroupingNode):
        return ("connector_group",)
    if isinstance(node, ConnectorNode):
        return ("connector",)
    if isinstance(node, CollectorNode):
        return ("collector",)
    if isinstance(node, NonSelectionNode):
        return ("non_selection",)
    if isinstance(node, NamedNode):
        return ("named_node",)
    return ("node",)

def adsg_node_key(node) -> tuple[str]:
    """Encode the ADORE architectural role without semantic names or identities."""
    if isinstance(node, ExternalOutConnectionNode):
        return ("external_out",)
    if isinstance(node, ExternalConnectionNode):
        return ("external",)
    if isinstance(node, ComponentNode):
        return ("component",)
    if isinstance(node, ComponentInstanceNode):
        return ("component_instance",)
    if isinstance(node, ComponentInstanceGroupNode):
        return ("component_instance_group",)
    if isinstance(node, FunctionNode):
        return ("function",)
    if isinstance(node, FunctionDecompositionNode):
        return ("function_decomposition",)
    if isinstance(node, NonFulfillmentNode):
        return ("non_fulfillment",)
    if isinstance(node, MultiFulfillmentNode):
        return ("multi_fulfillment",)
    if isinstance(node, ConceptNode):
        return ("concept",)
    if isinstance(node, SystemNode):
        return ("system",)
    if isinstance(node, FunctionDerivationNode):
        return ("function_derivation",)
    if isinstance(node, PortGroupNode):
        return ("needed_port_group" if node.is_needed else "provided_port_group",)
    if isinstance(node, SystemGroupNode):
        return ("system_group",)
    if isinstance(node, GroupNode):
        return ("group",)
    if isinstance(node, ProvidedPortNode):
        return ("provided_port",)
    if isinstance(node, NeededPortNode):
        return ("needed_port",)
    if isinstance(node, AttributeValueNode):
        return ("attribute_value",)
    if isinstance(node, AttributeNode):
        return ("attribute",)
    if isinstance(node, InputParamNode):
        return ("input_parameter",)
    if isinstance(node, NopNode):
        return ("nop",)
    return dsg_node_key(node)

def semantic_type_node_key(node):
    """Encode reusable semantic types while excluding instance identities."""
    role = adsg_node_key(node)[0]

    if isinstance(node, ComponentNode):
        return (role, node.name)
    if isinstance(node, ComponentInstanceNode):
        return (role, node.comp_name)
    if isinstance(node, ComponentInstanceGroupNode):
        return (role, node.comp_name)
    if isinstance(node, FunctionNode):
        return (role, node.name)
    if isinstance(node, FunctionDerivationNode):
        return (role, node.name)
    if isinstance(node, PortGroupNode):
        return (role, node.name)
    if isinstance(node, (ProvidedPortNode, NeededPortNode)):
        return (role, node.name)
    if isinstance(node, AttributeValueNode):
        return (role, node.key)
    if isinstance(node, AttributeNode):
        return (role, node.key)
    if isinstance(node, MetricNode):
        return (role, node.name)
    if isinstance(node, DesignVariableNode):
        return (role, node.name)
    if isinstance(node, InputParamNode):
        return (role, node.name)
    if isinstance(node, NamedNode):
        return (role, node.name)

    return (role,)

def semantic_node_key(node):
    """Encode legacy semantic labels without instance identities."""
    if isinstance(node, ExternalOutConnectionNode):
        return ("external", "EXT_OUT")
    if isinstance(node, ExternalConnectionNode):
        return ("external", "EXT")
    if isinstance(node, FunctionNode):
        return ("function", node.name)
    if isinstance(node, FunctionDecompositionNode):
        return ("function_decomposition", node.name)
    if isinstance(node, ComponentNode):
        return ("component", node.name)
    if isinstance(node, ComponentInstanceNode):
        return ("component_instance", node.comp_name)
    if isinstance(node, ComponentInstanceGroupNode):
        return ("component_instance_group", node.comp_name)
    if isinstance(node, ProvidedPortNode):
        return ("provided_port", node.name)
    if isinstance(node, NeededPortNode):
        return ("needed_port", node.name)
    if isinstance(node, AttributeNode):
        return ("attribute", node.key)
    if isinstance(node, AttributeValueNode):
        return ("attribute_value", node.key, repr(node.value))
    if isinstance(node, MetricNode):
        return ("metric", node.name)
    if isinstance(node, DesignVariableNode):
        return ("design_variable", node.name)

    return (node_family(node), str(node))

_NODE_KEY_BY_DEPTH = {
    EncodingDepth.CONSTANT: constant_node_key,
    EncodingDepth.DSG: dsg_node_key,
    EncodingDepth.ADSG: adsg_node_key,
    EncodingDepth.TYPE: semantic_type_node_key,
    EncodingDepth.FAMILY: family_node_key,
    EncodingDepth.SEMANTIC: semantic_node_key,
}

def _node_key(node, depth_by_family: dict[str, EncodingDepth]):
    family = node_family(node)
    depth = depth_by_family.get(family, depth_by_family.get("*"))
    if depth is None:
        raise KeyError(f"No encoding depth configured for node family {family!r}")

    encoder = _NODE_KEY_BY_DEPTH.get(depth)
    if encoder is None:
        raise ValueError(f"Unsupported encoding depth: {depth!r}")
    return encoder(node)

def node_label(node, depth_by_family: dict[str, EncodingDepth]) -> str:
    return "|".join(map(str, _node_key(node, depth_by_family=depth_by_family)))

def graph_edge_labels(
    G_nx: nx.MultiDiGraph,
    nodes: list,
) -> dict[tuple[int, int], str]:
    node_index = {node: i for i, node in enumerate(nodes)}
    multiplicity_by_pair = defaultdict(int)

    for src, tgt, _ in G_nx.edges(keys=True):
        multiplicity_by_pair[(node_index[src], node_index[tgt])] += 1

    return {
        edge_idx: f"m:{multiplicity}"
        for edge_idx, multiplicity in multiplicity_by_pair.items()
    }
