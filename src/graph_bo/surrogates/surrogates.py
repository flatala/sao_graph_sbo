from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from pymoo.util.normalization import Normalization

from adsg_core.optimization.graph_processor import GraphProcessor
from graph_bo.kernels import GraphKernel, ThetaParamSpec
from graph_bo.surrogates.resolver import ArchInstanceResolver
from graph_bo.surrogates.sizing import SizingKernel
from graph_bo.surrogates.training_logger import TrainingLogger

from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging import cross_distances
from smt.surrogate_models.krg_based.hyperparam_optim import CobylaOptimizer
from smt.sampling_methods import LHS
import numpy as np

__all__ = [
    "KernelThetaSlice",
    "GraphKernelHandler",
    "ADSGKriging",
]


_WEIGHTED_COMPOSITIONS = {
    "additive",
    "additive_interaction",
}
_INTERACTION_COMPOSITIONS = {
    "additive_interaction",
}
_SUPPORTED_COMPOSITIONS = _WEIGHTED_COMPOSITIONS | {"multiplicative"}


@dataclass(frozen=True)
class KernelThetaSlice:
    """Metadata describing where a branch's parameters live inside the flat theta vector."""
    name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


class GraphKernelHandler:
    """Branch-level wrapper that adapts a `GraphKernel` to the surrogate's covariance-vector API.

    The handler owns:
      - a per-instance graph cache (dv -> kernel-specific graph repr), built lazily,
      - a per-training-set fitted kernel matrix, keyed by the current branch theta.
    """

    def __init__(self, kernel: GraphKernel, *, name: str | None = None):
        self.cache: ArchInstanceResolver | None = None
        self.kernel = kernel
        self.name = name or type(kernel).__name__

        # Native to the realized design, so keeping it across trainings is fine.
        self._corrected_dv_to_graph_dict: dict[tuple, Any] = {}

        # Train-set specific, so this one gets reset whenever the surrogate is retrained.
        self._train_kernel_matrix: np.ndarray | None = None
        self._train_kernel_theta_key: tuple[float, ...] | None = None

    def set_shared_cache(self, cache: ArchInstanceResolver) -> None:
        """Attach the surrogate-owned `ArchInstanceResolver` and clear branch-local caches."""
        self.cache = cache
        self._corrected_dv_to_graph_dict.clear()
        self.reset_train_cache()

    def reset_train_cache(self) -> None:
        """Drop the fitted train-train kernel matrix (called between training rounds)."""
        self._train_kernel_matrix = None
        self._train_kernel_theta_key = None

    def get_theta_specs(self) -> list[ThetaParamSpec]:
        """Per-parameter specs (bounds, space, init) for the wrapped kernel."""
        return self.kernel.get_theta_specs()

    def report_diagnostics(self, theta: np.ndarray) -> dict[str, float]:
        """Forward to the wrapped kernel's diagnostic reporter (can be empty)."""
        return self.kernel.report_diagnostics(theta)

    def _get_graph_from_normalised_dv(self, dv_norm) -> Any:
        """Resolve a normalized design vector to the kernel's graph repr, caching by corrected dv."""
        assert self.cache is not None
        dv_corr, _ = self.cache.correct_normalised_dv(dv_norm)

        graph = self._corrected_dv_to_graph_dict.get(dv_corr)
        if graph is not None:
            return graph

        adsg = self.cache.get_adsg_from_corrected_dv(dv_corr)
        graph = self.kernel.build_graph(adsg)
        self._corrected_dv_to_graph_dict[dv_corr] = graph
        return graph

    def cov_vector_train(self, X_train: np.ndarray, ij: np.ndarray, branch_thetas: np.ndarray) -> np.ndarray:
        """Return the train-train covariance vector for the upper-triangular pairs in `ij`.

        The full train-train kernel matrix is computed once per `(train set, branch_thetas)`
        pair and cached, so repeated calls inside one COBYLA step are cheap.

        Args:
            X_train: (n_train, n_dv) normalized design vectors.
            ij: (n_pairs, 2) index pairs as returned by `cross_distances`.
            branch_thetas: theta slice for this branch.

        Returns:
            (n_pairs, 1) covariance values for the pairs in `ij`.
        """
        branch_thetas = np.asarray(branch_thetas, dtype=float).ravel()
        branch_theta_hash_key = tuple(branch_thetas.tolist())

        if self._train_kernel_matrix is None or self._train_kernel_theta_key != branch_theta_hash_key:
            train_graphs = [self._get_graph_from_normalised_dv(X_train[i]) for i in range(X_train.shape[0])]
            self._train_kernel_matrix = np.asarray(self.kernel.fit_transform(train_graphs, branch_thetas), dtype=float)
            self._train_kernel_theta_key = branch_theta_hash_key

        return self._train_kernel_matrix[ij[:, 0], ij[:, 1]].reshape(-1, 1)

    def cov_vector_predict(self, X_test: np.ndarray, branch_thetas: np.ndarray) -> np.ndarray:
        """Return the test-train covariance vector for the test points.

        Args:
            X_test: (n_test, n_dv) normalized design vectors.
            branch_thetas: theta slice for this branch.

        Returns:
            (n_test * n_train, 1) test-train covariance values, row-major per test point.
        """
        branch_thetas = np.asarray(branch_thetas, dtype=float).ravel()
        test_graphs = [self._get_graph_from_normalised_dv(X_test[i]) for i in range(X_test.shape[0])]
        return np.asarray(self.kernel.transform(test_graphs, branch_thetas), dtype=float).reshape(-1, 1)

    def get_config(self) -> dict:
        """Serializable snapshot of the handler + wrapped kernel config."""
        return {
            "class": type(self).__name__,
            "name": self.name,
            "theta_count": len(self.get_theta_specs()),
            "kernel": self.kernel.get_config(),
        }


class ADSGKriging(KrgBased):
    """Active ADSG kriging surrogate with one or more graph kernel branches and an optional sizing branch.

    Branches are combined either additively (with normalized, fixed-first weights) or
    multiplicatively. Hyperparameter optimization uses COBYLA on a mixed log/linear theta space
    derived from each branch's `ThetaParamSpec`s; the SMT `theta0` option is treated as the
    seed value (scalar) or full override (vector).
    """

    name = "ADSGKriging"
    _instance_counter = 0

    def __init__(
        self,
        graph_processor: GraphProcessor,
        normalization: Normalization,
        structure_kernels: GraphKernelHandler | Sequence[GraphKernelHandler],
        sizing_kernel: bool = True,
        composition: str = "additive",
        lambda_l2: float = 0.0,
        use_branch_theta0: bool = False,
        log_path=None,
        enable_timing: bool = False,
        sizing_kernel_mode: str = "hierarchical",
        **kwargs,
    ):
        """
        Args:
            graph_processor: ADSG graph processor for the problem.
            normalization: pymoo normalization used to map DVs to the unit cube.
            structure_kernels: one or more graph kernel handlers. Each becomes its own branch.
            sizing_kernel: if True, add a sizing branch whenever the problem has sizing variables.
            sizing_kernel_mode: sizing distance rule; either the existing SMT-style
                "hierarchical" rule or fixed corrected/imputed mixed-variable distances.
            composition: branch-combination rule. In addition to "additive" and
                "multiplicative", "additive_interaction" adds every pairwise branch product.
            lambda_l2: L2 regulariser on normalized mixture-term weights (0.0 disables).
            use_branch_theta0: if True, seed COBYLA from each branch's own `ThetaParamSpec.init`;
                if False, seed every entry from the scalar `options["theta0"][0]`.
            log_path: optional path the training logger should append to.
            enable_timing: if True, the training logger collects per-phase timings.
            **kwargs: forwarded to `KrgBased.__init__` (SMT options).
        """
        super().__init__(**kwargs)

        if float(lambda_l2) < 0.0:
            raise ValueError(f"lambda_l2 must be non-negative, got {float(lambda_l2)}")
        if composition not in _SUPPORTED_COMPOSITIONS:
            raise ValueError(f"Unsupported composition: {composition!r}.")

        self.gp = graph_processor
        self.norm = normalization
        self.composition = composition
        self.lambda_l2 = float(lambda_l2)
        self.use_branch_theta0 = bool(use_branch_theta0)

        # Output identity (used as the log label)
        self.output_label = None
        self.output_name = None
        self.instance_index = ADSGKriging._instance_counter
        ADSGKriging._instance_counter += 1

        # `label_provider` closes over self so every log/timing entry carries
        # the current output and n_train context.
        self.logger = TrainingLogger(
            label_provider=self._log_label,
            log_path=None if log_path is None else Path(log_path),
            enable_timing=bool(enable_timing),
        )

        self.shared_cache = ArchInstanceResolver(graph_processor=graph_processor, normalization=normalization)

        if isinstance(structure_kernels, GraphKernelHandler):
            self.structure_kernels = [structure_kernels]
        else:
            self.structure_kernels = list(structure_kernels)
        if len(self.structure_kernels) == 0:
            raise ValueError("ADSGKriging requires at least one graph kernel handler")
        for handler in self.structure_kernels:
            handler.set_shared_cache(self.shared_cache)

        # Sizing branch is on by default; can be disabled for pure-structure tests.
        if not sizing_kernel:
            self.sizing_kernel = None
        else:
            sizing_kernel = SizingKernel(
                hierarchical_kernel=self.options["hierarchical_kernel"],
                power=float(self.options["pow_exp_power"]),
                mode=sizing_kernel_mode,
            )
            sizing_kernel.set_shared_cache(self.shared_cache)
            self.sizing_kernel = sizing_kernel if sizing_kernel.has_variables() else None

        branches: list[tuple[str, GraphKernelHandler | SizingKernel]] = [
            (f"graph_{i}:{handler.name}", handler) for i, handler in enumerate(self.structure_kernels)
        ]
        if self.sizing_kernel is not None and self.sizing_kernel.has_variables():
            branches.append(("sizing", self.sizing_kernel))
        self._branches = branches

        self._ij_train: np.ndarray | None = None
        self._theta_slices: dict[str, KernelThetaSlice] = {}
        self.last_theta_ = None

        # Skip COBYLA when there is nothing to optimize.
        branch_theta_dim = sum(len(branch.get_theta_specs()) for _, branch in self._branches)
        needs_mixture_weights = self._uses_mixture_weights()
        self.options["hyper_opt"] = (
            "Cobyla"
            if needs_mixture_weights or branch_theta_dim > 0
            else "NoOp"
        )

    def get_config(self) -> dict:
        """Serializable snapshot of surrogate-level + branch-level config."""
        return {
            "class": type(self).__name__,
            "composition": self.composition,
            "lambda_l2": self.lambda_l2,
            "use_branch_theta0": self.use_branch_theta0,
            "enable_timing": self.logger.enable_timing,
            "hyper_opt": self.options["hyper_opt"],
            "theta0": self.options["theta0"],
            "theta_bounds": self.options["theta_bounds"],
            "structure_kernels": [handler.get_config() for handler in self.structure_kernels],
            "sizing_kernel": None if self.sizing_kernel is None else self.sizing_kernel.get_config(),
        }

    def set_log_path(self, log_path) -> None:
        """Redirect the training logger to `log_path` (or detach if None)."""
        self.logger.set_log_path(None if log_path is None else Path(log_path))

    def flush_prediction_timing_summary(self) -> None:
        """Emit accumulated prediction-phase timing and reset the predict bucket."""
        self.logger.flush_phase("predict")

    def _log_label(self) -> str:
        """Build the `output=..., n_train=...` prefix used by every log/timing entry."""
        parts = []
        if self.output_label is not None:
            parts.append(f"output={self.output_label}")
        if self.output_name is not None:
            parts.append(f"output_name={self.output_name}")
        if self.output_label is None and self.output_name is None:
            parts.append(f"model={self.instance_index}")
        points = self.training_points.get(None) if hasattr(self, "training_points") else None
        n_train = int(points[0][0].shape[0]) if points else -1
        parts.append(f"n_train={n_train}")
        return ", ".join(parts)

    @staticmethod
    def _kernel_weight_metric_name(branch_name: str) -> str:
        """Sanitize a branch name (e.g. `graph_0:ld_wloa`) into a metric-safe key (`graph_0.ld_wloa`)."""
        def sanitise(part: str) -> str:
            cleaned = "".join(ch if ch.isalnum() else "_" for ch in part)
            return "_".join(token for token in cleaned.split("_") if token)

        return ".".join(sanitise(part) for part in branch_name.split(":", maxsplit=1))

    def _interaction_pairs(self) -> tuple[tuple[int, int], ...]:
        if self.composition not in _INTERACTION_COMPOSITIONS:
            return ()
        return tuple(combinations(range(len(self._branches)), 2))

    def _composition_term_names(self) -> tuple[str, ...]:
        branch_names = tuple(name for name, _ in self._branches)
        if self.composition not in _INTERACTION_COMPOSITIONS:
            return branch_names
        interaction_names = tuple(
            f"interaction:{branch_names[i]}__{branch_names[j]}"
            for i, j in self._interaction_pairs()
        )
        return branch_names + interaction_names

    def _uses_mixture_weights(self) -> bool:
        return self.composition in _WEIGHTED_COMPOSITIONS and len(self._composition_term_names()) > 1

    def get_kernel_weight_names(self) -> tuple[str, ...]:
        """Metric names for mixture-term weights, empty before an optimized mixture exists."""
        if not self._uses_mixture_weights():
            return ()
        return tuple(self._kernel_weight_metric_name(name) for name in self._composition_term_names())

    def get_kernel_weight_values(self) -> dict[str, float]:
        """Current normalized mixture weights, keyed by sanitized term name. Empty before training."""
        names = self.get_kernel_weight_names()
        if len(names) == 0 or self.last_theta_ is None:
            return {}

        branch_weights, _ = self._decode_theta(self.last_theta_)
        if branch_weights is None:
            return {}

        return {
            name: float(weight)
            for name, weight in zip(names, branch_weights)
        }

    def _branch_diagnostics(self, theta: np.ndarray) -> dict[str, float]:
        """Per-branch diagnostics flattened into `<branch>.<key>` entries.

        Args:
            theta: full surrogate-level theta vector (must be decodable via `_theta_slices`).
        """
        _, branch_theta = self._decode_theta(theta)
        out: dict[str, float] = {}
        for name, branch in self._branches:
            for key, value in branch.report_diagnostics(branch_theta[name]).items():
                out[f"{self._kernel_weight_metric_name(name)}.{key}"] = float(value)
        return out

    def get_kernel_diagnostic_names(self) -> tuple[str, ...]:
        """Diagnostic metric names exposed by all branches (probed before training)."""
        # Probe each branch with its own theta0 so we can discover diagnostics
        # before training (i.e. before `_check_param` builds `_theta_slices`).
        names: list[str] = []
        for branch_name, branch in self._branches:
            theta0 = np.array([s.init for s in branch.get_theta_specs()], dtype=float)
            for key in branch.report_diagnostics(theta0).keys():
                names.append(f"{self._kernel_weight_metric_name(branch_name)}.{key}")
        return tuple(names)

    def get_kernel_diagnostic_values(self) -> dict[str, float]:
        """Current diagnostic values for all branches. Empty before training."""
        if self.last_theta_ is None:
            return {}
        return self._branch_diagnostics(np.asarray(self.last_theta_, dtype=float).ravel())

    def _build_theta_layout(self, branch_theta0_value: float | None = None) -> tuple[np.ndarray, dict[str, KernelThetaSlice]]:
        """Concatenate per-branch theta0s into a single vector and record per-branch slices.

        Layout:
          [composition_weights] [branch_0 theta] [branch_1 theta] ...

        Args:
            branch_theta0_value: if given, replace every per-branch init with this scalar
                (the `use_branch_theta0=False` path).

        Returns:
            (theta0, theta_slices) where `theta0` is the assembled init vector and
            `theta_slices[name]` locates each branch (and any composition weights) inside it.
        """
        branches = self._branches
        theta_chunks: list[np.ndarray] = []
        theta_slices: dict[str, KernelThetaSlice] = {}
        specs: list[ThetaParamSpec] = []
        cursor = 0

        # Fixed-first re-param: optimize n_terms-1 free weights, with the first fixed at 1.0.
        if self._uses_mixture_weights():
            n_free = len(self._composition_term_names()) - 1
            theta_chunks.append(np.ones((n_free,), dtype=float))
            theta_slices["composition_weights"] = KernelThetaSlice("composition_weights", cursor, cursor + n_free)
            specs.extend([ThetaParamSpec(0.01, 100.0, "linear", 1.0)] * n_free)
            cursor += n_free

        for branch_name, branch in branches:
            branch_specs = branch.get_theta_specs()
            branch_theta0 = np.array([s.init for s in branch_specs], dtype=float)
            if branch_theta0_value is not None and branch_theta0.size > 0:
                branch_theta0 = np.full(branch_theta0.shape, float(branch_theta0_value), dtype=float)
            if branch_theta0.size > 0:
                theta_chunks.append(branch_theta0)
            theta_slices[branch_name] = KernelThetaSlice(branch_name, cursor, cursor + branch_theta0.size)
            specs.extend(branch_specs)
            cursor += branch_theta0.size

        # KrgBased expects a non-empty theta vector even when the pure-structure
        # configuration has no SMT-side trainable hyperparameters; keep one dummy entry.
        if len(theta_chunks) == 0:
            theta0 = np.ones((1,), dtype=float)
            theta_slices["compatibility_theta"] = KernelThetaSlice("compatibility_theta", 0, 1)
            self._theta_param_specs: list[ThetaParamSpec] = [ThetaParamSpec(0.01, 100.0, "linear", 1.0)]
            return theta0, theta_slices

        self._theta_param_specs = specs
        return np.concatenate(theta_chunks), theta_slices

    def _decode_theta(self, theta) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
        """Split theta into mixture weights and branch parameters.

        Args:
            theta: full surrogate-level theta vector.

        Returns:
            (branch_weights, branch_theta) where:
              - `branch_weights` is the normalized mixture-term vector (or None),
              - `branch_theta[name]` is the theta slice for each branch.

            In the pure-compatibility case (no real trainable params), returns
            (None, {branch_name: empty array, ...}).
        """
        theta = np.asarray(theta, dtype=float).ravel()
        branches = self._branches

        if "compatibility_theta" in self._theta_slices:
            return None, {
                branch_name: np.zeros((0,), dtype=float) for branch_name, _ in branches
            }

        branch_weights = None
        branch_theta: dict[str, np.ndarray] = {}

        if "composition_weights" in self._theta_slices:
            s = self._theta_slices["composition_weights"]
            free_weights = np.maximum(theta[s.start:s.stop], 0.0)
            raw_weights = np.concatenate([[1.0], free_weights])
            branch_weights = raw_weights / float(np.sum(raw_weights))

        for branch_name, _ in branches:
            s = self._theta_slices[branch_name]
            branch_theta[branch_name] = theta[s.start:s.stop]

        return branch_weights, branch_theta

    def _to_optimizer_space(self, theta: np.ndarray) -> np.ndarray:
        """Map a native theta vector into the COBYLA search space (log10 for log specs)."""
        result = np.array(theta, dtype=float)
        for i, spec in enumerate(self._theta_param_specs):
            if spec.space == "log":
                result[i] = np.log10(np.clip(theta[i], spec.lb, spec.ub))
        return result

    def _from_optimizer_space(self, opt_theta: np.ndarray) -> np.ndarray:
        """Inverse of `_to_optimizer_space`: COBYLA-space vector back to native theta."""
        result = np.array(opt_theta, dtype=float)
        for i, spec in enumerate(self._theta_param_specs):
            if spec.space == "log":
                result[i] = 10.0 ** opt_theta[i]
        return result

    def _optimizer_bounds(self) -> np.ndarray:
        """(n_param, 2) lower/upper bounds in optimizer space."""
        limits = []
        for spec in self._theta_param_specs:
            if spec.space == "log":
                limits.append([np.log10(spec.lb), np.log10(spec.ub)])
            else:
                limits.append([spec.lb, spec.ub])
        return np.array(limits, dtype=float)

    def _optimizer_constraints(self) -> list:
        """COBYLA inequality constraints encoding the per-parameter bounds (in optimizer space)."""
        constraints = []
        for i, spec in enumerate(self._theta_param_specs):
            lb = np.log10(spec.lb) if spec.space == "log" else spec.lb
            ub = np.log10(spec.ub) if spec.space == "log" else spec.ub
            constraints.append(lambda t, i=i, lb=lb: t[i] - lb)
            constraints.append(lambda t, i=i, ub=ub: ub - t[i])
        return constraints

    def _check_param(self):
        """Run SMT's parent param check on a scalar theta seed, then install the ADSG-specific layout."""
        requested_theta0 = np.asarray(self.options["theta0"], dtype=float).ravel()
        if requested_theta0.size == 0:
            raise ValueError("ADSGKriging requires a non-empty theta0 option.")

        # KrgBased initialises internal state like _eval_noise and _noise0 here, and expects
        # a standard SMT theta shape — give it a single scalar seed, then overwrite with the
        # ADSGKriging-specific theta layout.
        self.options["theta0"] = np.array([float(requested_theta0[0])])
        super()._check_param()

        branch_theta0_value = None if self.use_branch_theta0 else float(requested_theta0[0])
        theta0, theta_slices = self._build_theta_layout(branch_theta0_value=branch_theta0_value)
        if requested_theta0.size > 1:
            if requested_theta0.size != theta0.size:
                raise ValueError(
                    f"ADSGKriging expects theta0 with 1 or {theta0.size} values, got {requested_theta0.size}"
                )
            theta0 = requested_theta0

        self.options["theta0"] = theta0
        self._theta0 = list(np.asarray(theta0, dtype=float))
        self._theta_slices = theta_slices
        self.n_param = int(theta0.size)

    def _new_train(self):
        """Prepare for retraining: keep shared/per-instance caches, clear train-set-specific state.

        - keeps shared realization caches (corrected dvs, activeness, ADSGs);
        - keeps branch-local instance representations (graph reprs, sizing reprs);
        - clears the train-train fitted matrices, `_ij_train`, and `last_theta_`.
        """
        self.logger.flush_phase("predict")
        self.logger.reset_timings("train")
        self.logger.reset_theta_logged()

        self._ij_train = None
        self.last_theta_ = None

        for _, branch in self._branches:
            branch.reset_train_cache()

        try:
            with self.logger.time(name="total", phase="train"):
                super()._new_train()
        finally:
            self.logger.log_timing_summary("train")

    def _reduced_likelihood_function(self, theta):
        """Wrap SMT's reduced likelihood with an optional branch-weight regularizer.

        Args:
            theta: full surrogate-level theta vector.

        Returns:
            (rlf_value, par): SMT's likelihood scalar (minus the L2 penalty) and the par dict.
        """
        rlf_value, par = super()._reduced_likelihood_function(theta)

        rlf_value -= self._mixture_weight_penalty(theta)

        return rlf_value, par

    def _mixture_weight_penalty(self, theta) -> float:
        """Return the L2 penalty on normalized mixture-term weights."""
        if self.lambda_l2 <= 0.0 or not self._uses_mixture_weights():
            return 0.0
        branch_weights, _ = self._decode_theta(theta)
        if branch_weights is None:
            raise RuntimeError("Missing weights for mixture-weight regularization")
        target_weights = np.full(branch_weights.shape, 1.0 / branch_weights.size, dtype=float)
        return self.lambda_l2 * float(np.sum((branch_weights - target_weights) ** 2))

    def _optimize_hyperparam(self, D, use_multistart=True, limit=None):
        """Run COBYLA over the optimizer-space theta with multiple starts, return optimal theta + rlf.

        Args:
            D: SMT pairwise distance matrix (unused by the graph-kernel correlation path;
                stored on `self.D` because SMT's likelihood evaluator reads it).
            use_multistart: ignored (we always do at least 2 starts and add LHS samples when
                `options["n_start"] > 1`); kept for parent-signature compatibility.
            limit: ignored (we size the per-start COBYLA budget from the param count);
                kept for parent-signature compatibility.

        Returns:
            (optimal_rlf_value, optimal_par, optimal_theta).
        """
        # `self.D` is required by SMT's likelihood evaluator but unused by the graph-kernel
        # correlation path (see `_matrix_data_corr`; `dx` is ignored).
        del use_multistart, limit
        self.D = D

        is_noop = self.options["hyper_opt"] == "NoOp"
        self.noise0 = np.array(self.options["noise0"] if is_noop else self._noise0)
        theta0_native = np.array(self.options["theta0"], dtype=float).ravel()

        if is_noop:
            optimal_theta = theta0_native
        else:
            specs = self._theta_param_specs
            bounds = self._optimizer_bounds()
            constraints = self._optimizer_constraints()

            for i, spec in enumerate(specs):
                theta0_native[i] = np.clip(theta0_native[i], spec.lb, spec.ub)
            theta0_in_opt = self._to_optimizer_space(theta0_native)

            def minus_rlf(opt_theta: np.ndarray) -> float:
                return -self._reduced_likelihood_function(theta=self._from_optimizer_space(opt_theta))[0]

            # Always start from (a) the configured theta0 and (b) one uniform random sample.
            theta_rand_in_opt = bounds[:, 0] + self.rng.random(len(specs)) * (bounds[:, 1] - bounds[:, 0])
            theta_starts = [theta0_in_opt, theta_rand_in_opt]

            # Add additional LHS starts when SMT's n_start option requests them.
            n_start = self.options["n_start"]
            if n_start > 1:
                sampling = LHS(xlimits=bounds, criterion="maximin", seed=self.rng)
                theta_starts.extend(sampling(n_start))

            theta_starts = np.vstack(theta_starts)
            result = CobylaOptimizer().optimize(
                objective=minus_rlf,
                theta_starts=theta_starts,
                constraints=constraints,
                limit=max(12 * len(specs), 50),
            )

            if result is None or "x" not in result:
                optimal_theta = theta0_native
            else:
                optimal_theta = self._from_optimizer_space(result["x"])

        optimal_rlf_value, optimal_par = self._reduced_likelihood_function(theta=optimal_theta)
        self.last_theta_ = optimal_theta.copy()

        branch_weights, branch_theta = self._decode_theta(optimal_theta)
        self.logger.log_theta(
            self.composition,
            branch_weights,
            branch_theta,
            weight_names=self.get_kernel_weight_names(),
        )
        return optimal_rlf_value, optimal_par, optimal_theta

    def _matrix_data_corr(self, theta, *, x=None, kplsk_second_loop=False, **_):
        """Branch-composed correlation vector: train-train pairs when `x is None`, else test-train.

        Args:
            theta: full surrogate-level theta vector.
            x: test design vectors; if None, fit/use the train-train pairs from `cross_distances`.
            kplsk_second_loop: not supported (KPLSK is unrelated to graph kernels).

        Returns:
            A correlation column compatible with `KrgBased`'s post-processing: either the
            single-branch correlation, the weighted sum (additive), or the elementwise product
            (multiplicative).
        """
        # `dx` (SMT's componentwise distance) is intentionally ignored — graph kernels build
        # their own correlations from x_train/x_test, not from a precomputed dx.
        if kplsk_second_loop:
            raise NotImplementedError("ADSGKriging does not support KPLSK")

        branch_weights, branch_theta = self._decode_theta(theta)

        if x is None:
            # Train-train: ensure `ij` pairs are available, then build per-branch covariance vectors.
            x_train = self.training_points[None][0][0]
            if self._ij_train is None:
                with self.logger.time(name="cross_distances", phase="train"):
                    _, self._ij_train = cross_distances(x_train)

            branch_corrs = []
            assert self._ij_train is not None
            for branch_name, branch in self._branches:
                with self.logger.time(f"branch.{branch_name}", phase="train"):
                    branch_corrs.append(branch.cov_vector_train(x_train, self._ij_train, branch_theta[branch_name]))

        else:
            # Test-train: train-train must already be fitted by the time we get here.
            x_test = np.asarray(x, dtype=float)
            branch_corrs = []
            for branch_name, branch in self._branches:
                with self.logger.time(f"branch.{branch_name}", phase="predict"):
                    branch_corrs.append(branch.cov_vector_predict(x_test, branch_theta[branch_name]))

        if len(branch_corrs) == 1:
            return branch_corrs[0]

        return self._combine_branch_correlations(
            branch_corrs,
            branch_weights=branch_weights,
        )

    def _combine_branch_correlations(
        self,
        branch_corrs: Sequence[np.ndarray],
        *,
        branch_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Combine already-computed branch correlations according to ``composition``."""
        if len(branch_corrs) == 0:
            raise ValueError("At least one branch correlation is required")
        if len(branch_corrs) == 1:
            return branch_corrs[0]

        if self.composition in _WEIGHTED_COMPOSITIONS:
            if branch_weights is None:
                raise RuntimeError("Missing mixture-term weights")

            term_corrs = list(branch_corrs)
            for i, j in self._interaction_pairs():
                term_corrs.append(branch_corrs[i] * branch_corrs[j])

            if len(branch_weights) != len(term_corrs):
                raise ValueError(
                    f"Expected {len(term_corrs)} mixture weights, got {len(branch_weights)}"
                )
            result = np.zeros_like(branch_corrs[0])
            for weight, corr_vec in zip(branch_weights, term_corrs):
                result = result + float(weight) * corr_vec
            return result

        if self.composition == "multiplicative":
            result = branch_corrs[0]
            for corr_vec in branch_corrs[1:]:
                result = result * corr_vec
            return result

        raise ValueError(f"Unsupported composition: {self.composition!r}")
