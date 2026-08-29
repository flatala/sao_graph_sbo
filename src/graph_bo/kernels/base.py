from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from adsg_core import DSGType

from graph_bo.surrogates.theta import ThetaParamSpec


class GraphKernel(ABC):
    def get_config(self) -> dict:
        return {"class": type(self).__name__}

    def get_theta_specs(self) -> list[ThetaParamSpec]:
        return []

    def report_diagnostics(self, theta: np.ndarray) -> dict[str, float]:
        """Optional per-kernel diagnostic values to log alongside the optimised theta.

        Subclasses override this when they have something meaningful to expose
        (e.g. learned-depth WLOA kernels report their active weights).
        Default: no diagnostics.
        """
        return {}

    @abstractmethod
    def build_graph(self, G: DSGType) -> Any:
        """Create the graph object used by the kernel."""

    @abstractmethod
    def fit_transform(self, train_graphs: Sequence[Any], theta: np.ndarray | None = None) -> Any:
        """Return K_train_train (n_train x n_train) and store fitted state internally."""

    @abstractmethod
    def transform(self, test_graphs: Sequence[Any], theta: np.ndarray | None = None) -> Any:
        """Return K_test_train (n_test x n_train) using stored fitted state."""
