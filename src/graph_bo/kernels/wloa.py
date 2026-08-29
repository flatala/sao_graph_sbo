from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
from adsg_core import DSGType
from scipy import sparse

from graph_bo.kernels.extractors import EncodingDepth, node_label
from graph_bo.surrogates.theta import ThetaParamSpec
from graph_bo.kernels.base import GraphKernel


@dataclass(frozen=True)
class WLFeatures:
    edge_dict: dict[int, dict[int, float]]
    labels: dict[int, int]
    counts_by_level: tuple[Counter, ...]
    feature_key: tuple


@dataclass(frozen=True)
class MultiGranularityWLFeatures:
    edge_dict: dict[int, dict[int, float]]
    features_by_granularity: tuple[WLFeatures, ...]
    feature_key: tuple


class LdWloa(GraphKernel):
    """
    Learned depth Weisfeiler-Lehman Optimal Assignment kernel.
    """

    def __init__(self, cutoff: int, depth_by_family: dict[str, EncodingDepth]):
        if type(cutoff) is not int or cutoff < 0:
            raise ValueError("'cutoff' must be a non-negative integer")
        self.cutoff = cutoff
        self.depth_by_family = dict(depth_by_family)

        # caches
        self._train_counts_by_level: tuple[tuple[Counter, ...], ...] | None = None
        self._train_level_kernels: tuple[np.ndarray, ...] | None = None
        self._train_level_diags: tuple[np.ndarray, ...] | None = None
        self._train_fingerprints: tuple | None = None
        self._graph_feature_cache: dict[tuple, WLFeatures] = {}
        self._label_to_id: dict[Any, int] = {}

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            if key in {"_graph_feature_cache", "_label_to_id"}:
                setattr(copied, key, value)
            else:
                setattr(copied, key, copy.deepcopy(value, memo))
        return copied

    def _relabel(self, label: Any) -> int:
        if label not in self._label_to_id:
            self._label_to_id[label] = len(self._label_to_id)
        return self._label_to_id[label]

    def get_theta_specs(self) -> list[ThetaParamSpec]:
        return [ThetaParamSpec(0.0, 1.0, "linear", 0.5)] * (self.cutoff + 1)

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "cutoff": self.cutoff,
            "theta_count": len(self.get_theta_specs()),
            "kernel": "ld_wloa_histogram_intersection",
        }

    def report_diagnostics(self, theta: np.ndarray) -> dict[str, float]:
        theta_arr = np.clip(np.asarray(theta, dtype=float).ravel(), 0.0, 1.0)
        result = {f"w_{i}": float(theta_arr[i]) for i in range(self.cutoff + 1)}
        result["w_sum"] = float(np.sum(theta_arr))
        return result

    def build_graph(self, G: DSGType) -> WLFeatures:
        # noinspection PyTypeChecker
        nx_graph: nx.MultiDiGraph = G.graph
        nodes = list(nx_graph.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}
        edge_dict = {node_index[node]: {} for node in nodes}

        for src in nodes:
            src_idx = node_index[src]
            for tgt in nx_graph.successors(src):
                edge_dict[src_idx][node_index[tgt]] = 1.0
            for tgt in nx_graph.predecessors(src):
                edge_dict[src_idx][node_index[tgt]] = 1.0

        labels = {
            node_index[node]: self._relabel(node_label(node, depth_by_family=self.depth_by_family))
            for node in nodes
        }

        structure_key = (
            tuple(labels[i] for i in range(len(nodes))),
            tuple(
                sorted(
                    (src_idx, tgt_idx)
                    for src_idx, successors in edge_dict.items()
                    for tgt_idx in successors.keys()
                )
            ),
        )

        cached = self._graph_feature_cache.get(structure_key)
        if cached is not None:
            return cached

        counts_by_level = []
        current_labels = labels
        for level in range(self.cutoff + 1):
            counts_by_level.append(Counter(current_labels.values()))
            if level == self.cutoff:
                continue
            next_labels = {}
            for node in edge_dict.keys():
                neighbor_labels = tuple(sorted(current_labels[nbr] for nbr in edge_dict.get(node, {}).keys()))
                next_labels[node] = self._relabel((current_labels[node], neighbor_labels))
            current_labels = next_labels

        feature_key = tuple(
            tuple(sorted((label, count) for label, count in level_counts.items()))
            for level_counts in counts_by_level
        )
        graph = WLFeatures(
            edge_dict=edge_dict,
            labels=labels,
            counts_by_level=tuple(counts_by_level),
            feature_key=feature_key,
        )
        self._graph_feature_cache[structure_key] = graph
        return graph

    def fit_transform(
        self,
        train_graphs: Sequence[WLFeatures],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        train_graphs = tuple(train_graphs)
        counts_by_level = self._counts_by_level_for_graphs(train_graphs)
        fingerprints = tuple(graph.feature_key for graph in train_graphs)

        same_train_fingerprints = (
            self._train_fingerprints is not None
            and len(self._train_fingerprints) == len(fingerprints)
            and all(old is new for old, new in zip(self._train_fingerprints, fingerprints))
        )

        self._train_level_kernels = self._train_level_kernels_for_counts(counts_by_level, fingerprints)
        if not same_train_fingerprints:
            self._train_level_diags = tuple(
                np.array([sum(counter.values()) for counter in level_counts], dtype=float)
                for level_counts in counts_by_level
            )
        self._train_counts_by_level = counts_by_level
        self._train_fingerprints = fingerprints

        return self._combine_level_kernels(
            self._train_level_kernels,
            theta,
            counts_left=counts_by_level,
            counts_right=counts_by_level,
        )

    def transform(
        self,
        test_graphs: Sequence[WLFeatures],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._train_counts_by_level is None:
            raise RuntimeError("LdWloa.transform called before fit_transform")

        test_graphs = tuple(test_graphs)
        test_counts_by_level = self._counts_by_level_for_graphs(test_graphs)
        level_kernels = self._level_kernels_for_counts(
            test_counts_by_level,
            self._train_counts_by_level,
        )
        return self._combine_level_kernels(
            level_kernels,
            theta,
            counts_left=test_counts_by_level,
            counts_right=self._train_counts_by_level,
        )

    def _weights_from_theta(self, theta: np.ndarray | None) -> np.ndarray:
        theta_arr = np.array(
            theta if theta is not None else [s.init for s in self.get_theta_specs()], dtype=float
        ).ravel()
        if theta_arr.size != self.cutoff + 1:
            raise ValueError(
                f"LdWloa expects {self.cutoff + 1} theta values, got {theta_arr.size}"
            )
        return np.clip(theta_arr, 0.0, 1.0)

    def _combine_level_kernels(
        self,
        level_kernels: tuple[np.ndarray, ...],
        theta: np.ndarray | None,
        *,
        counts_left,
        counts_right,
    ) -> np.ndarray:
        weights = self._weights_from_theta(theta)
        K = np.zeros_like(level_kernels[0], dtype=float)
        for weight, level_kernel in zip(weights, level_kernels):
            K = K + float(weight) * level_kernel

        diag_left = self._self_kernel_diag_for_counts(counts_left, weights)
        diag_right = self._self_kernel_diag_for_counts(counts_right, weights)
        denom = np.sqrt(np.outer(diag_left, diag_right))
        return np.divide(K, denom, out=np.zeros_like(K), where=denom > 0.0)

    def _counts_by_level_for_graphs(
        self,
        graphs: tuple[WLFeatures, ...],
    ) -> tuple[tuple[Counter, ...], ...]:
        return tuple(
            tuple(graph.counts_by_level[level] for graph in graphs)
            for level in range(self.cutoff + 1)
        )

    def _level_diags_for_counts(self, counts_by_level) -> tuple[np.ndarray, ...]:
        if counts_by_level is self._train_counts_by_level and self._train_level_diags is not None:
            return self._train_level_diags
        return tuple(
            np.array([sum(counter.values()) for counter in level_counts], dtype=float)
            for level_counts in counts_by_level
        )

    def _train_level_kernels_for_counts(
        self,
        counts_by_level: tuple[tuple[Counter, ...], ...],
        fingerprints: tuple,
    ) -> tuple[np.ndarray, ...]:
        if (
            self._train_level_kernels is not None
            and self._train_fingerprints is not None
            and len(self._train_fingerprints) == len(fingerprints)
            and all(old is new for old, new in zip(self._train_fingerprints, fingerprints))
        ):
            return self._train_level_kernels

        # Incremental growth: if old fingerprints are a prefix of new ones,
        # only compute new x old and new x new blocks; reuse old x old from the cached train kernels.
        if (
            self._train_fingerprints is not None
            and self._train_level_kernels is not None
            and len(self._train_fingerprints) < len(fingerprints)
            and all(old is new for old, new in zip(self._train_fingerprints, fingerprints))
        ):
            old_n = len(self._train_fingerprints)
            level_kernels = []
            for level, old_kernel in enumerate(self._train_level_kernels):
                new_counts = counts_by_level[level][old_n:]
                old_counts = counts_by_level[level][:old_n]
                new_old = LdWloa._kernel_matrix_for_level(new_counts, old_counts)
                new_new = LdWloa._kernel_matrix_for_level(new_counts, new_counts)
                K = np.empty((len(fingerprints), len(fingerprints)), dtype=float)
                K[:old_n, :old_n] = old_kernel
                K[old_n:, :old_n] = new_old
                K[:old_n, old_n:] = new_old.T
                K[old_n:, old_n:] = new_new
                level_kernels.append(K)
            return tuple(level_kernels)

        return self._level_kernels_for_counts(counts_by_level, counts_by_level)

    def _level_kernels_for_counts(
        self,
        left_counts_by_level: tuple[tuple[Counter, ...], ...],
        right_counts_by_level: tuple[tuple[Counter, ...], ...],
    ) -> tuple[np.ndarray, ...]:
        return tuple(
            LdWloa._kernel_matrix_for_level(left_counts, right_counts)
            for left_counts, right_counts in zip(left_counts_by_level, right_counts_by_level)
        )

    def _self_kernel_diag_for_counts(self, counts_by_level, weights: np.ndarray) -> np.ndarray:
        level_diags = self._level_diags_for_counts(counts_by_level)
        diag = np.zeros_like(level_diags[0], dtype=float)
        for weight, level_diag in zip(weights, level_diags):
            diag += float(weight) * level_diag
        return diag

    @staticmethod
    def _kernel_matrix_for_level(left_counts, right_counts) -> np.ndarray:
        """Per-level histogram-intersection Gram via sparse CSC arithmetic."""
        labels = sorted(
            set().union(
                *(counter.keys() for counter in left_counts),
                *(counter.keys() for counter in right_counts),
            )
        )
        label_index = {label: i for i, label in enumerate(labels)}
        left = LdWloa._counts_to_csc(left_counts, label_index)
        right = LdWloa._counts_to_csc(right_counts, label_index)

        K = np.zeros((len(left_counts), len(right_counts)), dtype=float)
        left_csc = left.tocsc()
        right_csc = right.tocsc()
        for col in range(len(labels)):
            left_start, left_stop = left_csc.indptr[col], left_csc.indptr[col + 1]
            right_start, right_stop = right_csc.indptr[col], right_csc.indptr[col + 1]
            if left_start == left_stop or right_start == right_stop:
                continue
            left_rows = left_csc.indices[left_start:left_stop]
            right_rows = right_csc.indices[right_start:right_stop]
            left_values = left_csc.data[left_start:left_stop]
            right_values = right_csc.data[right_start:right_stop]
            K[np.ix_(left_rows, right_rows)] += np.minimum(left_values[:, None], right_values[None, :])
        return K

    @staticmethod
    def _counts_to_csc(counts, label_index) -> sparse.csc_matrix:
        rows = []
        cols = []
        values = []
        for row, counter in enumerate(counts):
            for label, count in counter.items():
                rows.append(row)
                cols.append(label_index[label])
                values.append(float(count))
        return sparse.csc_matrix((values, (rows, cols)), shape=(len(counts), len(label_index)))


class MultiGranularityLdWloa(GraphKernel):
    """Shared-depth WLOA over a hierarchy of node-label granularities."""

    def __init__(
        self,
        cutoff: int,
        granularities: list[tuple[str, dict[str, EncodingDepth]]],
    ):
        if type(cutoff) is not int or cutoff < 0:
            raise ValueError("'cutoff' must be a non-negative integer")
        if len(granularities) < 2:
            raise ValueError("'granularities' must contain at least two entries")

        names = tuple(name for name, _ in granularities)
        if len(set(names)) != len(names):
            raise ValueError("'granularities' must have unique names")

        self.cutoff = cutoff
        self.granularities = tuple(
            (name, dict(depths))
            for name, depths in granularities
        )

        self._matrix_kernels = tuple(
            LdWloa(cutoff=cutoff, depth_by_family=depths)
            for _, depths in self.granularities
        )
        self._graph_feature_cache: dict[tuple, MultiGranularityWLFeatures] = {}
        self._label_to_id: dict[Any, int] = {}

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            if key in {"_graph_feature_cache", "_label_to_id"}:
                setattr(copied, key, value)
            else:
                setattr(copied, key, copy.deepcopy(value, memo))
        return copied

    def _relabel(self, label: Any) -> int:
        if label not in self._label_to_id:
            self._label_to_id[label] = len(self._label_to_id)
        return self._label_to_id[label]

    def get_theta_specs(self) -> list[ThetaParamSpec]:
        depth_specs = [
            ThetaParamSpec(0.0, 1.0, "linear", 0.5)
        ] * (self.cutoff + 1)
        mixture_specs = [
            ThetaParamSpec(0.0, 1.0, "linear", (remaining - 1) / remaining)
            for remaining in range(len(self.granularities), 1, -1)
        ]
        return depth_specs + mixture_specs

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "cutoff": self.cutoff,
            "theta_count": len(self.get_theta_specs()),
            "kernel": "multi_granularity_shared_depth_wloa_histogram_intersection",
            "granularities": [name for name, _ in self.granularities],
        }

    def report_diagnostics(self, theta: np.ndarray) -> dict[str, float]:
        depth_weights, granularity_weights = self._theta_parts(theta)
        result = {
            f"w_{level}": float(depth_weights[level])
            for level in range(self.cutoff + 1)
        }
        result["w_sum"] = float(np.sum(depth_weights))
        result.update({
            f"{name}_weight": float(weight)
            for (name, _), weight in zip(self.granularities, granularity_weights)
        })
        return result

    def build_graph(self, G: DSGType) -> MultiGranularityWLFeatures:
        # noinspection PyTypeChecker
        nx_graph: nx.MultiDiGraph = G.graph
        nodes = list(nx_graph.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}
        edge_dict = {node_index[node]: {} for node in nodes}

        for src in nodes:
            src_idx = node_index[src]
            for tgt in nx_graph.successors(src):
                edge_dict[src_idx][node_index[tgt]] = 1.0
            for tgt in nx_graph.predecessors(src):
                edge_dict[src_idx][node_index[tgt]] = 1.0

        labels_by_granularity = tuple(
            {
                node_index[node]: self._relabel(
                    (name, node_label(node, depth_by_family=depth_by_family))
                )
                for node in nodes
            }
            for name, depth_by_family in self.granularities
        )
        structure_key = (
            tuple(
                tuple(labels[index] for index in range(len(nodes)))
                for labels in labels_by_granularity
            ),
            tuple(
                sorted(
                    (src_idx, tgt_idx)
                    for src_idx, successors in edge_dict.items()
                    for tgt_idx in successors.keys()
                )
            ),
        )
        cached = self._graph_feature_cache.get(structure_key)
        if cached is not None:
            return cached

        counts_by_granularity = self._counts_by_level_for_granularities(
            edge_dict,
            labels_by_granularity,
        )
        features_by_granularity = tuple(
            WLFeatures(
                edge_dict=edge_dict,
                labels=labels,
                counts_by_level=counts_by_level,
                feature_key=tuple(
                    tuple(sorted(level_counts.items()))
                    for level_counts in counts_by_level
                ),
            )
            for labels, counts_by_level in zip(
                labels_by_granularity,
                counts_by_granularity,
            )
        )
        feature_key = tuple(
            features.feature_key for features in features_by_granularity
        )
        graph = MultiGranularityWLFeatures(
            edge_dict=edge_dict,
            features_by_granularity=features_by_granularity,
            feature_key=feature_key,
        )
        self._graph_feature_cache[structure_key] = graph
        return graph

    def fit_transform(
        self,
        train_graphs: Sequence[MultiGranularityWLFeatures],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        train_graphs = tuple(train_graphs)
        depth_weights, granularity_weights = self._theta_parts(theta)
        kernels = tuple(
            matrix_kernel.fit_transform(
                tuple(
                    graph.features_by_granularity[index]
                    for graph in train_graphs
                ),
                depth_weights,
            )
            for index, matrix_kernel in enumerate(self._matrix_kernels)
        )
        return self._mix_kernels(kernels, granularity_weights)

    def transform(
        self,
        test_graphs: Sequence[MultiGranularityWLFeatures],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        test_graphs = tuple(test_graphs)
        depth_weights, granularity_weights = self._theta_parts(theta)
        kernels = tuple(
            matrix_kernel.transform(
                tuple(
                    graph.features_by_granularity[index]
                    for graph in test_graphs
                ),
                depth_weights,
            )
            for index, matrix_kernel in enumerate(self._matrix_kernels)
        )
        return self._mix_kernels(kernels, granularity_weights)

    def _counts_by_level_for_granularities(
        self,
        edge_dict: dict[int, dict[int, float]],
        labels_by_granularity: tuple[dict[int, int], ...],
    ) -> tuple[tuple[Counter, ...], ...]:
        neighbor_indices = {
            node: tuple(neighbors.keys())
            for node, neighbors in edge_dict.items()
        }
        counts_by_granularity = [
            [] for _ in labels_by_granularity
        ]
        current_labels_by_granularity = labels_by_granularity
        for level in range(self.cutoff + 1):
            for counts_by_level, current_labels in zip(
                counts_by_granularity,
                current_labels_by_granularity,
            ):
                counts_by_level.append(Counter(current_labels.values()))
            if level == self.cutoff:
                continue

            next_labels_by_granularity = tuple(
                {} for _ in current_labels_by_granularity
            )
            for node, neighbors in neighbor_indices.items():
                for next_labels, current_labels in zip(
                    next_labels_by_granularity,
                    current_labels_by_granularity,
                ):
                    neighbor_labels = tuple(
                        sorted(current_labels[neighbor] for neighbor in neighbors)
                    )
                    next_labels[node] = self._relabel(
                        (current_labels[node], neighbor_labels)
                    )
            current_labels_by_granularity = next_labels_by_granularity

        return tuple(
            tuple(counts_by_level)
            for counts_by_level in counts_by_granularity
        )

    def _theta_parts(
        self,
        theta: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta_arr = np.array(
            theta if theta is not None else [spec.init for spec in self.get_theta_specs()],
            dtype=float,
        ).ravel()
        expected = self.cutoff + len(self.granularities)
        if theta_arr.size != expected:
            raise ValueError(
                f"MultiGranularityLdWloa expects {expected} theta values, "
                f"got {theta_arr.size}"
            )
        theta_arr = np.clip(theta_arr, 0.0, 1.0)
        n_level = self.cutoff + 1
        depth_weights = theta_arr[:n_level]
        remaining_weight = 1.0
        granularity_weights = []
        for stick_fraction in theta_arr[n_level:]:
            granularity_weights.append(remaining_weight * (1.0 - stick_fraction))
            remaining_weight *= stick_fraction
        granularity_weights.append(remaining_weight)
        return depth_weights, np.asarray(granularity_weights)

    @staticmethod
    def _mix_kernels(
        kernels: tuple[np.ndarray, ...],
        granularity_weights: np.ndarray,
    ) -> np.ndarray:
        mixed = np.zeros_like(kernels[0], dtype=float)
        for weight, kernel in zip(granularity_weights, kernels):
            mixed += float(weight) * kernel
        return mixed


class Wloa(GraphKernel):
    """Fixed-depth Weisfeiler-Lehman Optimal Assignment kernel."""

    def __init__(self, depth: int, depth_by_family: dict[str, EncodingDepth]):
        if type(depth) is not int or depth < 0:
            raise ValueError("'depth' must be a non-negative integer")

        self.depth = depth
        self.depth_by_family = dict(depth_by_family)

        # caches
        self._train_counts_by_level: tuple[tuple[Counter, ...], ...] | None = None
        self._train_level_kernels: tuple[np.ndarray, ...] | None = None
        self._train_level_diags: tuple[np.ndarray, ...] | None = None
        self._train_fingerprints: tuple | None = None
        self._graph_feature_cache: dict[tuple, WLFeatures] = {}
        self._label_to_id: dict[Any, int] = {}

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            if key in {"_graph_feature_cache", "_label_to_id"}:
                setattr(copied, key, value)
            else:
                setattr(copied, key, copy.deepcopy(value, memo))
        return copied

    def _relabel(self, label: Any) -> int:
        if label not in self._label_to_id:
            self._label_to_id[label] = len(self._label_to_id)
        return self._label_to_id[label]

    def get_theta_specs(self) -> list:
        # No learned hyperparameters.
        return []

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "depth": self.depth,
            "theta_count": 0,
            "kernel": "wloa_histogram_intersection",
        }

    def report_diagnostics(self, theta: np.ndarray | None = None) -> dict[str, float]:
        return {
            "depth": float(self.depth),
            "num_levels": float(self.depth + 1),
        }

    def build_graph(self, G: DSGType) -> WLFeatures:
        # noinspection PyTypeChecker
        nx_graph: nx.MultiDiGraph = G.graph
        nodes = list(nx_graph.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}
        edge_dict = {node_index[node]: {} for node in nodes}

        for src in nodes:
            src_idx = node_index[src]
            for tgt in nx_graph.successors(src):
                edge_dict[src_idx][node_index[tgt]] = 1.0
            for tgt in nx_graph.predecessors(src):
                edge_dict[src_idx][node_index[tgt]] = 1.0

        labels = {
            node_index[node]: self._relabel(
                node_label(node, depth_by_family=self.depth_by_family)
            )
            for node in nodes
        }

        structure_key = (
            tuple(labels[i] for i in range(len(nodes))),
            tuple(
                sorted(
                    (src_idx, tgt_idx)
                    for src_idx, successors in edge_dict.items()
                    for tgt_idx in successors.keys()
                )
            ),
        )

        cached = self._graph_feature_cache.get(structure_key)
        if cached is not None:
            return cached

        counts_by_level = []
        current_labels = labels

        for level in range(self.depth + 1):
            counts_by_level.append(Counter(current_labels.values()))

            if level == self.depth:
                continue

            next_labels = {}
            for node in edge_dict.keys():
                neighbor_labels = tuple(
                    sorted(current_labels[nbr] for nbr in edge_dict.get(node, {}).keys())
                )
                next_labels[node] = self._relabel((current_labels[node], neighbor_labels))

            current_labels = next_labels

        feature_key = tuple(
            tuple(sorted((label, count) for label, count in level_counts.items()))
            for level_counts in counts_by_level
        )

        graph = WLFeatures(
            edge_dict=edge_dict,
            labels=labels,
            counts_by_level=tuple(counts_by_level),
            feature_key=feature_key,
        )

        self._graph_feature_cache[structure_key] = graph
        return graph

    def fit_transform(
        self,
        train_graphs: Sequence[WLFeatures],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        self._validate_no_theta(theta)

        train_graphs = tuple(train_graphs)
        counts_by_level = self._counts_by_level_for_graphs(train_graphs)
        fingerprints = tuple(graph.feature_key for graph in train_graphs)

        same_train_fingerprints = (
            self._train_fingerprints is not None
            and len(self._train_fingerprints) == len(fingerprints)
            and all(old is new for old, new in zip(self._train_fingerprints, fingerprints))
        )

        self._train_level_kernels = self._train_level_kernels_for_counts(
            counts_by_level,
            fingerprints,
        )

        if not same_train_fingerprints:
            self._train_level_diags = tuple(
                np.array([sum(counter.values()) for counter in level_counts], dtype=float)
                for level_counts in counts_by_level
            )

        self._train_counts_by_level = counts_by_level
        self._train_fingerprints = fingerprints

        return self._combine_level_kernels(
            self._train_level_kernels,
            counts_left=counts_by_level,
            counts_right=counts_by_level,
        )

    def transform(
        self,
        test_graphs: Sequence[WLFeatures],
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        self._validate_no_theta(theta)

        if self._train_counts_by_level is None:
            raise RuntimeError("Wloa.transform called before fit_transform")

        test_graphs = tuple(test_graphs)
        test_counts_by_level = self._counts_by_level_for_graphs(test_graphs)

        level_kernels = self._level_kernels_for_counts(
            test_counts_by_level,
            self._train_counts_by_level,
        )

        return self._combine_level_kernels(
            level_kernels,
            counts_left=test_counts_by_level,
            counts_right=self._train_counts_by_level,
        )

    @staticmethod
    def _validate_no_theta(theta: np.ndarray | None) -> None:
        if theta is not None and np.asarray(theta).size != 0:
            raise ValueError("Wloa does not use theta parameters")

    def _combine_level_kernels(
        self,
        level_kernels: tuple[np.ndarray, ...],
        *,
        counts_left,
        counts_right,
    ) -> np.ndarray:
        K = np.zeros_like(level_kernels[0], dtype=float)

        for level_kernel in level_kernels:
            K = K + level_kernel

        diag_left = self._self_kernel_diag_for_counts(counts_left)
        diag_right = self._self_kernel_diag_for_counts(counts_right)

        denom = np.sqrt(np.outer(diag_left, diag_right))

        return np.divide(K, denom, out=np.zeros_like(K), where=denom > 0.0)

    def _counts_by_level_for_graphs(
        self,
        graphs: tuple[WLFeatures, ...],
    ) -> tuple[tuple[Counter, ...], ...]:
        return tuple(
            tuple(graph.counts_by_level[level] for graph in graphs)
            for level in range(self.depth + 1)
        )

    def _level_diags_for_counts(self, counts_by_level) -> tuple[np.ndarray, ...]:
        if counts_by_level is self._train_counts_by_level and self._train_level_diags is not None:
            return self._train_level_diags

        return tuple(
            np.array([sum(counter.values()) for counter in level_counts], dtype=float)
            for level_counts in counts_by_level
        )

    def _train_level_kernels_for_counts(
        self,
        counts_by_level: tuple[tuple[Counter, ...], ...],
        fingerprints: tuple,
    ) -> tuple[np.ndarray, ...]:
        if (
            self._train_level_kernels is not None
            and self._train_fingerprints is not None
            and len(self._train_fingerprints) == len(fingerprints)
            and all(old is new for old, new in zip(self._train_fingerprints, fingerprints))
        ):
            return self._train_level_kernels

        # Incremental growth: if old fingerprints are a prefix of new ones,
        # only compute new x old and new x new blocks; reuse old x old.
        if (
            self._train_fingerprints is not None
            and self._train_level_kernels is not None
            and len(self._train_fingerprints) < len(fingerprints)
            and all(old is new for old, new in zip(self._train_fingerprints, fingerprints))
        ):
            old_n = len(self._train_fingerprints)
            level_kernels = []

            for level, old_kernel in enumerate(self._train_level_kernels):
                new_counts = counts_by_level[level][old_n:]
                old_counts = counts_by_level[level][:old_n]

                new_old = Wloa._kernel_matrix_for_level(new_counts, old_counts)
                new_new = Wloa._kernel_matrix_for_level(new_counts, new_counts)

                K = np.empty((len(fingerprints), len(fingerprints)), dtype=float)
                K[:old_n, :old_n] = old_kernel
                K[old_n:, :old_n] = new_old
                K[:old_n, old_n:] = new_old.T
                K[old_n:, old_n:] = new_new

                level_kernels.append(K)

            return tuple(level_kernels)

        return self._level_kernels_for_counts(counts_by_level, counts_by_level)

    def _level_kernels_for_counts(
        self,
        left_counts_by_level: tuple[tuple[Counter, ...], ...],
        right_counts_by_level: tuple[tuple[Counter, ...], ...],
    ) -> tuple[np.ndarray, ...]:
        return tuple(
            Wloa._kernel_matrix_for_level(left_counts, right_counts)
            for left_counts, right_counts in zip(left_counts_by_level, right_counts_by_level)
        )

    def _self_kernel_diag_for_counts(self, counts_by_level) -> np.ndarray:
        level_diags = self._level_diags_for_counts(counts_by_level)
        diag = np.zeros_like(level_diags[0], dtype=float)

        for level_diag in level_diags:
            diag += level_diag

        return diag

    @staticmethod
    def _kernel_matrix_for_level(left_counts, right_counts) -> np.ndarray:
        """Per-level histogram-intersection Gram via sparse CSC arithmetic."""
        labels = sorted(
            set().union(
                *(counter.keys() for counter in left_counts),
                *(counter.keys() for counter in right_counts),
            )
        )

        label_index = {label: i for i, label in enumerate(labels)}

        left = Wloa._counts_to_csc(left_counts, label_index)
        right = Wloa._counts_to_csc(right_counts, label_index)

        K = np.zeros((len(left_counts), len(right_counts)), dtype=float)

        left_csc = left.tocsc()
        right_csc = right.tocsc()

        for col in range(len(labels)):
            left_start, left_stop = left_csc.indptr[col], left_csc.indptr[col + 1]
            right_start, right_stop = right_csc.indptr[col], right_csc.indptr[col + 1]

            if left_start == left_stop or right_start == right_stop:
                continue

            left_rows = left_csc.indices[left_start:left_stop]
            right_rows = right_csc.indices[right_start:right_stop]
            left_values = left_csc.data[left_start:left_stop]
            right_values = right_csc.data[right_start:right_stop]

            K[np.ix_(left_rows, right_rows)] += np.minimum(
                left_values[:, None],
                right_values[None, :],
            )

        return K

    @staticmethod
    def _counts_to_csc(counts, label_index) -> sparse.csc_matrix:
        rows = []
        cols = []
        values = []

        for row, counter in enumerate(counts):
            for label, count in counter.items():
                rows.append(row)
                cols.append(label_index[label])
                values.append(float(count))

        return sparse.csc_matrix(
            (values, (rows, cols)),
            shape=(len(counts), len(label_index)),
        )
