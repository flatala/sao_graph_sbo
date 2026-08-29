from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch
from adsg_core.optimization.graph_processor import GraphProcessor
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from graph_bo.gnn.data.features import ADSGFeatureExtractor, ADSGNodeFeatureExtractor


class ADSGTensorBuilder:
    def __init__(
        self,
        graph_processor: GraphProcessor,
        feature_extractor_cls: type[ADSGFeatureExtractor] = ADSGNodeFeatureExtractor,
    ):
        self.graph_processor = graph_processor
        self.design_space_graph = graph_processor.graph
        self.parent_nx = self.design_space_graph._graph
        self.nodes = list(self.parent_nx.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.N = len(self.nodes)

        self.A_parent = self._build_parent_adjacency()
        self.feature_extractor = feature_extractor_cls(graph_processor, self.nodes, self.parent_nx)
        self.X_parent = self.feature_extractor.X_parent
        self.feature_dim = self.feature_extractor.feature_dim
        self.node_labels = self.feature_extractor.node_labels
        self.binary_feature_indices = self.feature_extractor.binary_feature_indices
        self.continuous_feature_indices = self.feature_extractor.continuous_feature_indices

    def print_summary(self) -> None:
        print(f"N = {self.N} nodes")
        self.feature_extractor.print_summary()

    def sample(self, n_samples: int, max_attempts_factor: int = 20) -> list[Data]:
        seen: set[tuple] = set()
        data_list: list[Data] = []
        max_attempts = max_attempts_factor * n_samples

        for _ in range(max_attempts):
            if len(data_list) >= n_samples:
                break

            graph, corrected_vector, is_active = self.graph_processor.get_graph(
                self.graph_processor.get_random_design_vector(),
                create=True,
            )
            key = tuple(corrected_vector)
            if key in seen:
                continue
            seen.add(key)
            data_list.append(self.from_graph(graph, corrected_vector, is_active))

        if len(data_list) < n_samples:
            warnings.warn(
                f"Only {len(data_list)} unique instances found after {max_attempts} attempts "
                f"(requested {n_samples}). Design space may be small.",
                stacklevel=2,
            )

        return data_list

    def from_vector(self, design_vector: Sequence[float | int]) -> Data:
        graph, corrected_vector, is_active = self.graph_processor.get_graph(
            np.asarray(design_vector, dtype=float),
            create=True,
        )
        return self.from_graph(graph, corrected_vector, is_active)

    def from_graph(
        self,
        adsg_instance,
        corrected_vector: Sequence[float | int],
        is_active: Sequence[bool],
    ) -> Data:
        mask = self._build_active_mask(adsg_instance)
        A_directed = self._build_instance_adjacency(adsg_instance)
        A_encoder = np.maximum(A_directed, A_directed.T).astype(np.float32)
        A_target = (A_encoder > 0.0).astype(np.float32)
        X_target = self.feature_extractor.encode_instance(adsg_instance, mask, corrected_vector, is_active)

        edge_index, _ = dense_to_sparse(torch.from_numpy(A_encoder))

        return Data(
            x=torch.from_numpy(X_target.copy()),
            edge_index=edge_index,
            mask=torch.from_numpy(mask),
            A_target=torch.from_numpy(A_target),
            A_encoder=torch.from_numpy(A_encoder),
            A_parent=torch.from_numpy(self.A_parent.copy()),
            X_target=torch.from_numpy(X_target),
            y=torch.from_numpy(self.node_labels.copy()),
            corrected_vector=torch.tensor(corrected_vector, dtype=torch.float32),
            is_active=torch.tensor(is_active, dtype=torch.bool),
            num_nodes=self.N,
        )

    def _build_parent_adjacency(self) -> np.ndarray:
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for src, tgt in self.parent_nx.edges():
            i, j = self.node_to_idx[src], self.node_to_idx[tgt]
            if i != j:
                A[i, j] = 1.0
        return A

    def _build_active_mask(self, adsg_instance) -> np.ndarray:
        mask = np.zeros((self.N,), dtype=np.float32)
        for node in adsg_instance._graph.nodes():
            if node not in self.node_to_idx:
                raise ValueError(f"Instance node is not in the design-space ADSG: {node!r}")
            mask[self.node_to_idx[node]] = 1.0
        return mask

    def _build_instance_adjacency(self, adsg_instance) -> np.ndarray:
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for src, tgt in adsg_instance._graph.edges():
            if src not in self.node_to_idx:
                raise ValueError(f"Instance edge source is not in the design-space ADSG: {src!r}")
            if tgt not in self.node_to_idx:
                raise ValueError(f"Instance edge target is not in the design-space ADSG: {tgt!r}")
            i, j = self.node_to_idx[src], self.node_to_idx[tgt]
            if i != j:
                A[i, j] += 1.0
        return A
