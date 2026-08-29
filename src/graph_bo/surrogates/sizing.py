from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from adsg_core.graph.adsg_nodes import DesignVariableNode
from smt.utils.kriging import MixHrcKernelType

from graph_bo.surrogates.resolver import ArchInstanceResolver
from graph_bo.surrogates.theta import ThetaParamSpec


@dataclass(frozen=True)
class SizingVariableSpec:
    key: str
    kind: str
    source: str
    dv_index: int | None = None
    bounds: tuple[float, float] | None = None
    n_options: int | None = None
    attribute_comp_name: str | None = None
    attribute_key: str | None = None
    attribute_instance_idx: int | None = None
    attribute_option_values: tuple[Any, ...] | None = None


class SizingKernel:
    """Kernel over sizing design variables and semantic attribute choices.

    ``hierarchical`` retains the SMT-style activity mismatch penalty. ``imputed``
    instead embeds every corrected/imputed design in a fixed vector, using ordinary
    numeric distances and Gower mismatch for categorical coordinates.
    """

    name = "sizing"

    def __init__(
        self,
        *,
        hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
        power: float = 2.0,
        mode: str = "hierarchical",
    ):
        if mode not in {"hierarchical", "imputed"}:
            raise ValueError(f"Unsupported sizing kernel mode: {mode!r}")

        self.cache: ArchInstanceResolver | None = None
        self.hierarchical_kernel = hierarchical_kernel
        self.power = float(power)
        self.mode = mode

        self._sizing_variable_specs: tuple[SizingVariableSpec, ...] = ()
        self._sizing_variable_keys: tuple[str, ...] = ()
        self._numeric_mask = np.zeros((0,), dtype=bool)
        self._categorical_mask = np.zeros((0,), dtype=bool)
        self._has_attribute_specs = False

        # Caches for optimised sizing calculations
        self._corr_to_sizing_arrays: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

        self._train_values: np.ndarray | None = None
        self._train_is_active: np.ndarray | None = None
        self._train_pair_distances_powered: np.ndarray | None = None

    def set_shared_cache(self, cache: ArchInstanceResolver) -> None:
        self.cache = cache
        self._corr_to_sizing_arrays.clear()
        self.reset_train_cache()
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        # Determine the sizing variables in dv
        specs: list[SizingVariableSpec] = []
        for dv_index, des_var in enumerate(self.cache.gp.des_vars):
            node = des_var.node

            if not isinstance(node, DesignVariableNode):
                continue

            elif des_var.is_discrete:
                # Uses GOWER-style mismatch handling
                kind = "ordinal" if des_var.is_ordinal else "categorical"
                bounds = None
                n_options = des_var.n_opts
            else:
                kind = "continuous"
                lo, hi = des_var.bounds
                bounds = (float(lo), float(hi))
                n_options = None

            specs.append(
                SizingVariableSpec(
                    key=des_var.name,
                    kind=kind,
                    source="dv",
                    dv_index=dv_index,
                    bounds=bounds,
                    n_options=n_options,
                )
            )

        # Handle Attribute Nodes
        for instance_node, attr_node, decision_node, option_values in self.cache.get_attribute_schema():
            kind = "ordinal" if decision_node.is_ordinal else "categorical"
            bounds = None
            specs.append(
                SizingVariableSpec(
                    key=f"comp_att_{instance_node.comp_name}__{attr_node.key}_{instance_node.idx}",
                    kind=kind,
                    source="attribute",
                    bounds=bounds,
                    n_options=len(option_values),
                    attribute_comp_name=instance_node.comp_name,
                    attribute_key=attr_node.key,
                    attribute_instance_idx=instance_node.idx,
                    attribute_option_values=option_values,
                )
            )

        self._sizing_variable_specs = tuple(specs)
        self._sizing_variable_keys = tuple(spec.key for spec in specs)
        self._numeric_mask = np.array([spec.kind in {"continuous", "integer", "ordinal"} for spec in specs], dtype=bool)
        self._categorical_mask = np.array([spec.kind == "categorical" for spec in specs], dtype=bool)
        self._has_attribute_specs = any(spec.source == "attribute" for spec in specs)

    def has_variables(self) -> bool:
        return len(self._sizing_variable_specs) > 0

    def get_theta_specs(self) -> list[ThetaParamSpec]:
        return [ThetaParamSpec(1e-6, 20.0, "log", 1.0)] * len(self._sizing_variable_keys)

    def report_diagnostics(self, theta: np.ndarray) -> dict[str, float]:
        return {}

    def reset_train_cache(self) -> None:
        self._train_values = None
        self._train_is_active = None
        self._train_pair_distances_powered = None

    def _get_arrays_for_x_norm(self, x_row_norm) -> tuple[np.ndarray, np.ndarray]:
        dv_corr, is_active = self.cache.correct_normalised_dv(x_row_norm)
        cached = self._corr_to_sizing_arrays.get(dv_corr)
        if cached is not None:
            return cached

        semantic_attribute_choices = self.cache.get_attribute_choices(dv_corr) if self._has_attribute_specs else {}
        values = np.zeros((len(self._sizing_variable_specs),), dtype=float)
        is_active_array = np.zeros((len(self._sizing_variable_specs),), dtype=bool)
        dv_corr_norm = None
        if self.mode == "imputed":
            dv_corr_norm = self.cache.norm.forward(np.asarray(dv_corr, dtype=float)[None, :])[0]

        for i_var, spec in enumerate(self._sizing_variable_specs):
            if spec.source == "dv":
                assert spec.dv_index is not None
                if self.mode == "imputed":
                    values[i_var] = float(dv_corr_norm[spec.dv_index])
                    is_active_array[i_var] = True
                    continue

                value = dv_corr[spec.dv_index]
                is_active_array[i_var] = bool(is_active[spec.dv_index])
                if not is_active_array[i_var]:
                    continue
                if spec.kind == "categorical":
                    values[i_var] = float(value)
                else:
                    values[i_var] = self._normalize_numeric_value(value, spec)
            else:
                assert spec.attribute_comp_name is not None
                assert spec.attribute_instance_idx is not None
                assert spec.attribute_key is not None
                attr_key = (
                    spec.attribute_comp_name,
                    spec.attribute_instance_idx,
                    spec.attribute_key,
                )
                if attr_key not in semantic_attribute_choices:
                    if self.mode == "imputed":
                        values[i_var] = -1.0
                        is_active_array[i_var] = True
                    continue

                is_active_array[i_var] = True
                attr_value = semantic_attribute_choices[attr_key]
                if spec.kind == "categorical":
                    values[i_var] = float(self._categorical_value_to_index(attr_value, spec))
                else:
                    values[i_var] = self._normalize_numeric_value(
                        self._categorical_value_to_index(attr_value, spec),
                        spec,
                    )

        cached = (values, is_active_array)
        self._corr_to_sizing_arrays[dv_corr] = cached
        return cached

    @staticmethod
    def _categorical_value_to_index(value, spec: SizingVariableSpec) -> int:
        if spec.attribute_option_values is None:
            raise ValueError(f"Semantic attribute sizing variable {spec.key!r} requires attribute_option_values")
        return spec.attribute_option_values.index(value)

    @staticmethod
    def _normalize_numeric_value(value, spec: SizingVariableSpec) -> float:
        if spec.kind == "ordinal":
            if spec.n_options is None or spec.n_options <= 0:
                raise ValueError(f"Ordinal sizing variable is missing option count: {spec.key!r}")
            if spec.n_options == 1:
                return 0.0
            return float(value) / float(spec.n_options - 1)

        if spec.bounds is None:
            raise ValueError(f"Numeric sizing variable {spec.key!r} requires bounds")
        lower, upper = spec.bounds
        scale = upper - lower
        if scale <= 0.0:
            raise ValueError(f"Invalid bounds for sizing variable: {spec.key!r}")
        return (float(value) - lower) / scale

    def _arrays_for_x_norm_rows(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        n_vars = len(self._sizing_variable_specs)
        values = np.zeros((n_samples, n_vars), dtype=float)
        is_active = np.zeros((n_samples, n_vars), dtype=bool)
        for i_sample in range(n_samples):
            values[i_sample], is_active[i_sample] = self._get_arrays_for_x_norm(X[i_sample])
        return values, is_active

    def _distances_from_arrays(
        self,
        left_values: np.ndarray,
        left_is_active: np.ndarray,
        right_values: np.ndarray,
        right_is_active: np.ndarray,
    ) -> np.ndarray:

        # Validate if shapes check out
        if left_values.shape != left_is_active.shape:
            raise ValueError(f"Left sizing value/activeness shapes must match: {left_values.shape} != {left_is_active.shape}")
        if right_values.shape != right_is_active.shape:
            raise ValueError(f"Right sizing value/activeness shapes must match: {right_values.shape} != {right_is_active.shape}")
        if left_values.shape[-1] != len(self._sizing_variable_specs) or right_values.shape[-1] != len(self._sizing_variable_specs):
            raise ValueError(f"Sizing array last dim must match the schema: {left_values.shape[-1]} / {right_values.shape[-1]} vs {len(self._sizing_variable_specs)}")

        if self.mode == "imputed":
            distances = np.zeros(np.broadcast_shapes(left_values.shape, right_values.shape), dtype=float)
            if np.any(self._numeric_mask):
                distances[..., self._numeric_mask] = np.abs(
                    left_values[..., self._numeric_mask] - right_values[..., self._numeric_mask]
                )
            if np.any(self._categorical_mask):
                distances[..., self._categorical_mask] = np.where(
                    left_values[..., self._categorical_mask] == right_values[..., self._categorical_mask],
                    0.0,
                    1.0,
                )
            return distances

        # Prepare distances arr, mask for mismatched activeness + mask for both active
        distances = np.zeros(np.broadcast_shapes(left_values.shape, right_values.shape), dtype=float)
        active_mismatch = np.logical_xor(left_is_active, right_is_active)
        both_active = np.logical_and(left_is_active, right_is_active)

        # Calculate the distance between numeric variables
        if np.any(self._numeric_mask):
            # Get respective fields based on the mask
            numeric_mask = self._numeric_mask
            left_numeric = left_values[..., numeric_mask]
            right_numeric = right_values[..., numeric_mask]
            abs_delta = np.abs(left_numeric - right_numeric)

            # Distance in Alg Kernel: |X_1 - X_2| / (sqrt(|X_1| + 1) * sqrt(|X_2| + 1))
            if self.hierarchical_kernel == MixHrcKernelType.ALG_KERNEL:
                numeric_distances = (
                    2.0
                    * abs_delta
                    / (
                        np.sqrt(1.0 + left_numeric**2)
                        * np.sqrt(1.0 + right_numeric**2)
                    )
                )

            # We also support ARC kernel, idk why
            elif self.hierarchical_kernel == MixHrcKernelType.ARC_KERNEL:
                numeric_distances = np.sqrt(2.0) * np.sqrt(1.0 - np.cos(np.pi * abs_delta))

            # Unsupported ones, idk if there are any actually
            else:
                raise ValueError(f"Unsupported sizing kernel: {self.hierarchical_kernel!r}")

            # Apply the active / non-active distance of 1.0 (matches SMT I think?)
            distances[..., numeric_mask] = np.where(
                active_mismatch[..., numeric_mask],
                1.0,
                np.where(both_active[..., numeric_mask], numeric_distances, 0.0),
            )

        # Same for categorical, just with 0/1 Gower distance
        if np.any(self._categorical_mask):
            categorical_mask = self._categorical_mask
            categorical_distances = np.where(
                left_values[..., categorical_mask] == right_values[..., categorical_mask],
                0.0,
                1.0,
            )

            # Apply the active / non-active distance of 1.0 (matches SMT I think?)
            distances[..., categorical_mask] = np.where(
                active_mismatch[..., categorical_mask],
                1.0,
                np.where(both_active[..., categorical_mask], categorical_distances, 0.0),
            )

        return distances

    @staticmethod
    def _kernel_from_powered_distances(powered_distances: np.ndarray, theta_sizing: np.ndarray) -> np.ndarray:
        if powered_distances.shape[-1] != theta_sizing.shape[0]:
            raise ValueError(
                f"Powered distances and theta_sizing must have matching final dimensions: "
                f"{powered_distances.shape[-1]} != {theta_sizing.shape[0]}"
            )
        weighted_distance = np.tensordot(powered_distances, theta_sizing, axes=([-1], [0]))
        return np.exp(-weighted_distance)

    def cov_vector_train(self, X_train: np.ndarray, ij: np.ndarray, theta_sizing: np.ndarray) -> np.ndarray:
        if not self.has_variables():
            raise RuntimeError("SizingKernel was asked to evaluate with no sizing variables present")
        if theta_sizing.size != len(self._sizing_variable_keys):
            raise ValueError(
                f"SizingKernel expects {len(self._sizing_variable_keys)} theta values, got {theta_sizing.size}"
            )

        if self._train_values is None or self._train_is_active is None:
            self._train_values, self._train_is_active = self._arrays_for_x_norm_rows(X_train)

        if self._train_pair_distances_powered is None:
            # Caches for optimized sizing calculations
            left_values = self._train_values[ij[:, 0]]
            left_is_active = self._train_is_active[ij[:, 0]]
            right_values = self._train_values[ij[:, 1]]
            right_is_active = self._train_is_active[ij[:, 1]]
            pair_distances = self._distances_from_arrays(left_values, left_is_active, right_values, right_is_active)
            self._train_pair_distances_powered = pair_distances ** self.power

        assert  self._train_pair_distances_powered is not None
        return self._kernel_from_powered_distances(self._train_pair_distances_powered, theta_sizing).reshape(-1, 1)

    def cov_vector_predict(self, X_test: np.ndarray, theta_sizing: np.ndarray) -> np.ndarray:
        if not self.has_variables():
            raise RuntimeError("SizingKernel was asked to evaluate with no sizing variables present")
        if theta_sizing.size != len(self._sizing_variable_keys):
            raise ValueError(
                f"SizingKernel expects {len(self._sizing_variable_keys)} theta values, got {theta_sizing.size}"
            )
        if self._train_values is None or self._train_is_active is None:
            raise RuntimeError("SizingKernel predict path requires prepared training representations")

        X_test = np.asarray(X_test, dtype=float)
        test_values, test_is_active = self._arrays_for_x_norm_rows(X_test)
        pair_distances = self._distances_from_arrays(
            test_values[:, None, :],
            test_is_active[:, None, :],
            self._train_values[None, :, :],
            self._train_is_active[None, :, :],
        )
        powered_distances = pair_distances ** self.power
        K_x_train = self._kernel_from_powered_distances(powered_distances, theta_sizing)
        return K_x_train.reshape(-1, 1)

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "n_sizing_variables": len(self._sizing_variable_keys),
            "sizing_variable_keys": list(self._sizing_variable_keys),
            "n_attribute_choice_variables": int(
                sum(spec.source == "attribute" for spec in self._sizing_variable_specs)
            ),
            "mode": self.mode,
            "power": self.power,
            "hierarchical_kernel": str(self.hierarchical_kernel),
        }
