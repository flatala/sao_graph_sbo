from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
from adore.graph.adore_nodes import NeededPortNode, ProvidedPortNode
from adsg_core import DSGType
from adsg_core.graph.graph_edges import EdgeType, get_edge_type

from graph_bo.surrogates.theta import ThetaParamSpec
from graph_bo.kernels.base import GraphKernel


@dataclass(frozen=True)
class EdgeWeightFeatures:
    values: dict[tuple[str, str], float]
    fingerprint: tuple[tuple[str, str, float], ...]


def _connection_endpoint_label(node: object) -> str:
    if isinstance(node, ProvidedPortNode):
        role = "provided_port"
    elif isinstance(node, NeededPortNode):
        role = "needed_port"
    else:
        raise TypeError(
            "EdgeWeightKernel CONNECTS endpoints must be provided or needed ports, "
            f"got {type(node).__name__}"
        )
    return "|".join((role, node.name, node.comp_name))


class EdgeWeightKernel(GraphKernel):

    def __init__(
        self,
        gamma0: float = 1.0,
    ):
        if gamma0 <= 0.0:
            raise ValueError("'gamma0' must be positive")
        self.gamma0 = float(gamma0)

        self._train_features: tuple[EdgeWeightFeatures, ...] | None = None
        self._graph_feature_cache: dict[tuple, EdgeWeightFeatures] = {}
        self._distance_matrix_cache: dict[tuple, np.ndarray] = {}

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            if key in {"_graph_feature_cache", "_distance_matrix_cache"}:
                setattr(copied, key, value)
            else:
                setattr(copied, key, copy.deepcopy(value, memo))
        return copied

    def get_theta_specs(self) -> list[ThetaParamSpec]:
        return [ThetaParamSpec(1e-6, 20.0, "log", self.gamma0)]

    def build_graph(self, G: DSGType) -> EdgeWeightFeatures:
        G_nx: nx.MultiDiGraph = G.graph

        # Get connection multiplicity for realized connection endpoints.
        connection_multiplicity_by_pair = defaultdict(int)
        for src, tgt, key, data in G_nx.edges(keys=True, data=True):
            if get_edge_type((src, tgt, key, data)) == EdgeType.CONNECTS:
                connection_multiplicity_by_pair[(src, tgt)] += 1

        endpoint_nodes = {
            node
            for src, tgt in connection_multiplicity_by_pair.keys()
            for node in (src, tgt)
        }
        labels = {
            node: _connection_endpoint_label(node)
            for node in endpoint_nodes
        }

        # Prepare fingerprint for caching
        graph_fingerprint = (
            tuple(
                sorted(
                    (
                        labels[src],
                        labels[tgt],
                        multiplicity,
                    )
                    for (src, tgt), multiplicity in connection_multiplicity_by_pair.items()
                )
            ),
        )
        cached = self._graph_feature_cache.get(graph_fingerprint)
        if cached is not None:
            return cached

        # Sum logs of edge weights by directed semantic endpoint context.
        values = defaultdict(float)
        for (src, tgt), multiplicity in connection_multiplicity_by_pair.items():
            values[(labels[src], labels[tgt])] += math.log1p(float(multiplicity))

        feature_values = dict(values)
        fingerprint = tuple(
            sorted(
                (src_label, tgt_label, value)
                for (src_label, tgt_label), value in feature_values.items()
            )
        )
        features = EdgeWeightFeatures(
            values=feature_values,
            fingerprint=fingerprint,
        )

        self._graph_feature_cache[graph_fingerprint] = features
        return features

    def fit_transform(
        self,
        train_graphs: Sequence[EdgeWeightFeatures],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        self._train_features = tuple(train_graphs)
        distances = self._distance_matrix(self._train_features, self._train_features)
        # e ^ -(gamma * Sum(distances))
        return np.exp(-self._gamma_from_theta(theta) * distances)

    def transform(
        self,
        test_graphs: Sequence[EdgeWeightFeatures],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._train_features is None:
            raise RuntimeError("WLEdgeWeightKernel.transform called before fit_transform")
        distances = self._distance_matrix(tuple(test_graphs), self._train_features)
        return np.exp(-self._gamma_from_theta(theta) * distances)

    def _gamma_from_theta(self, theta: np.ndarray | None) -> float:
        theta_arr = np.array(
            [s.init for s in self.get_theta_specs()] if theta is None else theta, dtype=float
        ).ravel()
        if theta_arr.size != 1:
            raise ValueError(f"WLEdgeWeightKernel expects 1 theta value, got {theta_arr.size}")
        gamma = float(theta_arr[0])
        if gamma <= 0.0:
            raise ValueError("WLEdgeWeightKernel gamma must be positive")
        return gamma

    def _distance_matrix(
        self,
        left: tuple[EdgeWeightFeatures, ...],
        right: tuple[EdgeWeightFeatures, ...],
    ) -> np.ndarray:
        left_fingerprints = tuple(features.fingerprint for features in left)
        right_fingerprints = tuple(features.fingerprint for features in right)
        cache_key = (left_fingerprints, right_fingerprints)
        cached = self._distance_matrix_cache.get(cache_key)
        if cached is not None:
            return cached

        reverse_cached = self._distance_matrix_cache.get((right_fingerprints, left_fingerprints))
        if reverse_cached is not None:
            return reverse_cached.T

        distances = np.empty((len(left), len(right)), dtype=float)
        for i, x in enumerate(left):
            for j, y in enumerate(right):
                distances[i, j] = self._l1_distance(x, y)

        self._distance_matrix_cache[cache_key] = distances
        return distances

    @staticmethod
    def _l1_distance(
        x: EdgeWeightFeatures,
        y: EdgeWeightFeatures,
    ) -> float:
        # take the summed distance
        keys = set(x.values) | set(y.values)
        # Sum_ctx_i |Sum_w_ctx_i_1(log(w)) - Sum_w_ctx_i_2(log(w))|
        return float(sum(abs(x.values.get(key, 0.0) - y.values.get(key, 0.0)) for key in keys))

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "gamma0": self.gamma0,
            "theta_count": len(self.get_theta_specs()),
            "edge_filter": "EdgeType.CONNECTS",
            "endpoint_labels": "port direction + port type + component type",
            "features": "sum log1p(connection_multiplicity) by endpoint labels",
            "kernel": "exp(-gamma * l1_distance)",
        }
