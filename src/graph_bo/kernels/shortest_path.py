from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
from adsg_core import DSGType
from scipy import sparse

from graph_bo.kernels.extractors import EncodingDepth, node_label

from graph_bo.surrogates.theta import ThetaParamSpec

from graph_bo.kernels.base import GraphKernel


@dataclass(frozen=True)
class ShortestPathGraph:
    counts: Counter
    fingerprint: tuple


class ShortestPathKernel(GraphKernel):
    """Node-labeled shortest-path kernel for this repository's graph representation."""

    def __init__(
        self,
        *,
        depth_by_family: dict[str, EncodingDepth],
        normalize: bool = True,
        exponential: bool = False,
        sigma0: float = 1.0,
        directed: bool = True,
    ):
        if sigma0 <= 0.0:
            raise ValueError("'sigma0' must be positive")

        # params
        self.normalize = bool(normalize)
        self.exponential = bool(exponential)
        self.sigma0 = float(sigma0)
        self.directed = bool(directed)
        self.depth_by_family = dict(depth_by_family)

        # cache
        self._train_graphs: tuple[ShortestPathGraph, ...] | None = None
        self._train_fingerprints: tuple | None = None
        self._graph_feature_cache: dict[tuple, ShortestPathGraph] = {}
        self._raw_kernel_cache: dict[tuple, np.ndarray] = {}

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            if key in {"_graph_feature_cache", "_raw_kernel_cache"}:
                setattr(copied, key, value)
            else:
                setattr(copied, key, copy.deepcopy(value, memo))
        return copied

    def get_theta_specs(self) -> list[ThetaParamSpec]:
        if not self.exponential:
            return []
        return [ThetaParamSpec(1e-4, 20.0, "log", self.sigma0)]

    def build_graph(self, G: DSGType) -> ShortestPathGraph:
        G_nx: nx.MultiDiGraph = G.graph
        nodes = list(G_nx.nodes())
        adjacency = nx.to_numpy_array(G_nx, nodelist=nodes, weight="weight")
        labels = tuple(
            node_label(node, depth_by_family=self.depth_by_family)
            for node in nodes
        )
        edge_fingerprint = tuple(
            sorted(
                (i, j, float(adjacency[i, j]))
                for i in range(adjacency.shape[0])
                for j in range(adjacency.shape[1])
                if adjacency[i, j] != 0.0
            )
        )
        graph_fingerprint = (labels, edge_fingerprint, self.directed)
        cached = self._graph_feature_cache.get(graph_fingerprint)
        if cached is not None:
            return cached

        # Compute shortest paths between any two nodes
        distances = sparse.csgraph.shortest_path(
            sparse.csr_matrix(adjacency),
            directed=self.directed,
            unweighted=False,
        )

        counts = Counter()
        for u in range(distances.shape[0]):
            for v in range(distances.shape[1]):
                distance = distances[u, v]
                if u == v or not np.isfinite(distance):
                    continue
                # accumulate distances between nodes u, v of each specfic length
                counts[(labels[u], labels[v], float(distance))] += 1

        fingerprint = tuple(sorted((repr(key), count) for key, count in counts.items()))
        graph = ShortestPathGraph(counts=counts, fingerprint=fingerprint)
        self._graph_feature_cache[graph_fingerprint] = graph
        return graph

    def fit_transform(
        self,
        train_graphs: Sequence[ShortestPathGraph],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        self._train_graphs = tuple(train_graphs)
        self._train_fingerprints = tuple(graph.fingerprint for graph in self._train_graphs)
        raw = self._raw_kernel_matrix(self._train_graphs, self._train_graphs)
        return self._finalize_kernel(raw, theta, self._train_graphs, self._train_graphs)

    def transform(
        self,
        test_graphs: Sequence[ShortestPathGraph],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._train_graphs is None:
            raise RuntimeError("ShortestPathKernel.transform called before fit_transform")
        test_graphs = tuple(test_graphs)
        raw = self._raw_kernel_matrix(test_graphs, self._train_graphs)
        return self._finalize_kernel(raw, theta, test_graphs, self._train_graphs)

    def _sigma_from_theta(self, theta: np.ndarray | None) -> float:
        theta_arr = np.array(
            [s.init for s in self.get_theta_specs()] if theta is None else theta, dtype=float
        ).ravel()
        if theta_arr.size != 1:
            raise ValueError(f"ShortestPathKernel expects 1 theta value, got {theta_arr.size}")
        sigma = float(theta_arr[0])
        if sigma <= 0.0:
            raise ValueError("ShortestPathKernel sigma must be positive")
        return sigma

    def _finalize_kernel(
            self,
            raw: np.ndarray,
            theta: np.ndarray | None,
            left: tuple[ShortestPathGraph, ...],
            right: tuple[ShortestPathGraph, ...],
    ) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)

        if not np.all(np.isfinite(raw)):
            raise ValueError("ShortestPathKernel raw kernel contains NaN/Inf")

        if self.exponential:
            sigma = self._sigma_from_theta(theta)
            sigma2 = sigma * sigma

            if self.normalize:
                diag_left = np.array(
                    [self._self_kernel_value(graph) for graph in left],
                    dtype=float,
                )
                diag_right = np.array(
                    [self._self_kernel_value(graph) for graph in right],
                    dtype=float,
                )

                if not np.all(np.isfinite(diag_left)):
                    raise ValueError("ShortestPathKernel left diagonal contains NaN/Inf")
                if not np.all(np.isfinite(diag_right)):
                    raise ValueError("ShortestPathKernel right diagonal contains NaN/Inf")

                dist2 = diag_left[:, None] + diag_right[None, :] - 2.0 * raw

                # Cleanup tiny numerical negatives like -1e-12
                dist2 = np.maximum(dist2, 0.0)

                exponent = -dist2 / (2.0 * sigma2)

                # Prevent underflow weirdness; upper bound should already be 0
                exponent = np.clip(exponent, -745.0, 0.0)

                K = np.exp(exponent)

            else:
                # Original non-normalized exponential mode.
                # This can still overflow naturally, so clip it.
                exponent = raw / sigma2
                exponent = np.clip(exponent, -745.0, 700.0)
                K = np.exp(exponent)

        else:
            theta_arr = np.asarray([] if theta is None else theta, dtype=float).ravel()
            if theta_arr.size != 0:
                raise ValueError(f"ShortestPathKernel does not expose theta parameters, got {theta_arr.size}")

            K = raw.copy()
            if self.normalize:
                diag_left = np.array(
                    [self._self_kernel_value(graph) for graph in left],
                    dtype=float,
                )
                diag_right = np.array(
                    [self._self_kernel_value(graph) for graph in right],
                    dtype=float,
                )
                denom = np.sqrt(np.outer(diag_left, diag_right))
                K = np.divide(K, denom, out=np.zeros_like(K), where=denom > 0.0)

        if not np.all(np.isfinite(K)):
            raise ValueError(
                "ShortestPathKernel produced NaN/Inf "
                f"(min={np.nanmin(K)}, max={np.nanmax(K)})"
            )
        return K

    def _raw_kernel_matrix(
        self,
        left: tuple[ShortestPathGraph, ...],
        right: tuple[ShortestPathGraph, ...],
    ) -> np.ndarray:
        left_fingerprints = tuple(graph.fingerprint for graph in left)
        right_fingerprints = tuple(graph.fingerprint for graph in right)
        cache_key = (left_fingerprints, right_fingerprints)
        cached = self._raw_kernel_cache.get(cache_key)
        if cached is not None:
            return cached

        reverse_cached = self._raw_kernel_cache.get((right_fingerprints, left_fingerprints))
        if reverse_cached is not None:
            return reverse_cached.T

        K = self._feature_dot_matrix(left, right)
        self._raw_kernel_cache[cache_key] = K
        return K

    @staticmethod
    def _feature_dot_matrix(
        left: tuple[ShortestPathGraph, ...],
        right: tuple[ShortestPathGraph, ...],
    ) -> np.ndarray:
        keys = sorted(
            set().union(*(graph.counts.keys() for graph in left), *(graph.counts.keys() for graph in right)),
            key=repr,
        )
        key_index = {key: i for i, key in enumerate(keys)}
        left_mat = ShortestPathKernel._counts_to_csr(left, key_index)
        right_mat = ShortestPathKernel._counts_to_csr(right, key_index)
        return (left_mat @ right_mat.T).toarray().astype(float)

    @staticmethod
    def _counts_to_csr(graphs: tuple[ShortestPathGraph, ...], key_index) -> sparse.csr_matrix:
        rows = []
        cols = []
        values = []
        for row, graph in enumerate(graphs):
            for key, count in graph.counts.items():
                rows.append(row)
                cols.append(key_index[key])
                values.append(float(count))
        return sparse.csr_matrix((values, (rows, cols)), shape=(len(graphs), len(key_index)))

    @staticmethod
    def _self_kernel_value(graph: ShortestPathGraph) -> float:
        return float(sum(count * count for count in graph.counts.values()))

    def report_diagnostics(self, theta: np.ndarray) -> dict[str, float]:
        if not self.exponential:
            return {}
        return {"sigma": self._sigma_from_theta(theta)}

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "normalize": self.normalize,
            "exponential": self.exponential,
            "sigma0": self.sigma0,
            "theta_count": len(self.get_theta_specs()),
            "directed": self.directed,
            "depth_by_family": dict(self.depth_by_family),
            "kernel": "node_labeled_shortest_path",
        }
