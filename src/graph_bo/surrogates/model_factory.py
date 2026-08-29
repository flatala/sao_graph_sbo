from sb_arch_opt.algo.arch_sbo.models import ModelFactory, check_dependencies, MultiSurrogateModel
from smt.surrogate_models.krg_based import MixIntKernelType, MixHrcKernelType
from graph_bo.surrogates.surrogates import ADSGKriging, GraphKernelHandler
from typing import Any
import numpy as np

from importlib.util import find_spec
if find_spec("torch") is not None:
    from graph_bo.gnn.surrogate.heads import RegressionModel
    from graph_bo.gnn import Arch2VecEmbedder, DGBOSurrogate, GNNMultiSurrogate
else:
    Arch2VecEmbedder = None
    DGBOSurrogate = None
    GNNMultiSurrogate = None
    RegressionModel = None

__all__ = ["GraphModelFactory"]


class LoggedSurrogateModel:
    """Thin wrapper so single-output runs get the same logging/config plumbing."""

    def __init__(self, surrogate: ADSGKriging, output_name: str):
        self._surrogate: ADSGKriging = surrogate
        self._output_name = output_name

    def __getattr__(self, item):
        return getattr(self._surrogate, item)

    def set_training_values(self, xt: np.ndarray, yt: np.ndarray, name=None, is_acting=None) -> None:
        # Keep single-output logs consistent with the multi-output case
        self._surrogate.flush_prediction_timing_summary()
        self._surrogate.set_training_values(xt, yt, name=name, is_acting=is_acting)
        self._surrogate.output_label = 0
        self._surrogate.output_name = self._output_name

    def set_log_path(self, log_path) -> None:
        self._surrogate.set_log_path(log_path)

    def get_kernel_weight_names(self) -> tuple[str, ...]:
        return self._surrogate.get_kernel_weight_names()

    def get_kernel_weight_values(self) -> dict[str, float]:
        return self._surrogate.get_kernel_weight_values()

    def get_kernel_diagnostic_names(self) -> tuple[str, ...]:
        return self._surrogate.get_kernel_diagnostic_names()

    def get_kernel_diagnostic_values(self) -> dict[str, float]:
        return self._surrogate.get_kernel_diagnostic_values()

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "output_name": self._output_name,
            "base_surrogate": self._surrogate.get_config(),
        }


class LoggedMultiSurrogateModel(MultiSurrogateModel):
    """Multi-output wrapper that keeps logging/config behavior sane."""

    def __init__(self, surrogate: ADSGKriging, output_names: list[str]):
        # noinspection PyTypeChecker
        super().__init__(surrogate)  # MultiSurrogateModel uses a forward-ref 'SurrogateModel'
        self._output_names = output_names

    @property
    def _adsg(self) -> ADSGKriging:
        # Parent stores the template as `self._surrogate: SurrogateModel`; for our use it's
        # always an ADSGKriging passed in by GraphModelFactory.
        return self._surrogate  # type: ignore[return-value]

    @property
    def _adsg_models(self) -> list[ADSGKriging]:
        # All entries are deep-copies of the ADSGKriging template populated by MultiSurrogateModel.
        return self._models  # type: ignore[return-value]

    def set_training_values(self, xt: np.ndarray, yt: np.ndarray, name=None, is_acting=None) -> None:
        for model in self._adsg_models:
            model.flush_prediction_timing_summary()
        super().set_training_values(xt, yt, name=name, is_acting=is_acting)

        if len(self._output_names) != len(self._models):
            raise ValueError(
                f"Expected {len(self._output_names)} output labels, got {len(self._models)} trained models"
            )

        for iy, model in enumerate(self._adsg_models):
            model.output_label = iy
            model.output_name = self._output_names[iy]

    def set_log_path(self, log_path) -> None:
        self._adsg.set_log_path(log_path)
        for model in self._adsg_models:
            model.set_log_path(log_path)

    def get_kernel_weight_names(self) -> tuple[str, ...]:
        return self._adsg.get_kernel_weight_names()

    def get_kernel_weight_values(self) -> dict[str, float]:
        values = {}
        for iy, model in enumerate(self._adsg_models):
            for name, value in model.get_kernel_weight_values().items():
                values[f"{self._output_names[iy]}.{name}"] = value
        return values

    def get_kernel_diagnostic_names(self) -> tuple[str, ...]:
        return self._adsg.get_kernel_diagnostic_names()

    def get_kernel_diagnostic_values(self) -> dict[str, float]:
        values = {}
        for iy, model in enumerate(self._adsg_models):
            for name, value in model.get_kernel_diagnostic_values().items():
                values[f"{self._output_names[iy]}.{name}"] = value
        return values

    def get_config(self) -> dict:
        if len(self._models) > 0:
            return {
                "class": type(self).__name__,
                "n_models": len(self._models),
                "output_names": self._output_names,
                "base_surrogate": self._adsg_models[0].get_config(),
            }
        return {
            "class": type(self).__name__,
            "output_names": self._output_names,
            "base_surrogate": self._adsg.get_config(),
        }


class GraphModelFactory(ModelFactory):
    """ModelFactory extension with graph kernel support."""

    def get_md_adsg_gnn_model(
        self,
        embedder: Arch2VecEmbedder,
        downstream: RegressionModel,
        reconstruction_adj_cutoff: float,
        multi: bool = True,
        ignore_hierarchy: bool = False,
        device=None,
    ) -> tuple[GNNMultiSurrogate, Any]:
        check_dependencies()

        normalization = self.get_md_normalization()
        # noinspection PyUnresolvedReferences
        evaluator = self.problem.evaluator
        gp = evaluator.translator.graph_processor
        output_names = self._get_output_names(evaluator)

        embedder.prepare(
            graph_processor=gp,
            normalization=normalization,
            device=device,
        )

        surrogate = GNNMultiSurrogate(
            embedder=embedder,
            downstream=downstream,
            output_names=output_names if multi else output_names[:1],
            reconstruction_adj_cutoff=reconstruction_adj_cutoff,
        )

        if ignore_hierarchy:
            surrogate.supports["x_hierarchy"] = False

        return surrogate, normalization

    def get_md_adsg_dgbo_model(
        self,
        multi: bool = True,
        ignore_hierarchy: bool = False,
        device=None,
        **kwargs,
    ) -> tuple[DGBOSurrogate, Any]:
        check_dependencies()

        normalization = self.get_md_normalization()
        # noinspection PyUnresolvedReferences
        evaluator = self.problem.evaluator
        gp = evaluator.translator.graph_processor
        output_names = self._get_output_names(evaluator)

        surrogate = DGBOSurrogate(
            graph_processor=gp,
            normalization=normalization,
            output_names=output_names if multi else output_names[:1],
            device=device,
            **kwargs,
        )

        if ignore_hierarchy:
            surrogate.supports["x_hierarchy"] = False

        return surrogate, normalization

    def get_md_adsg_kriging_model(
        self,
        structure_kernels: GraphKernelHandler | list[GraphKernelHandler],
        sizing_kernel: bool = True,
        composition: str = "additive",
        multi: bool = True,
        ignore_hierarchy: bool = False,
        sizing_kernel_mode: str = "hierarchical",
        **kwargs_,
    ) -> tuple[LoggedMultiSurrogateModel | LoggedSurrogateModel, Any]:
        check_dependencies()

        normalization = self.get_md_normalization()
        design_space = self.problem.design_space
        norm_ds_spec = self.create_smt_design_space_spec(
            design_space,
            md_normalize=True,
            ignore_hierarchy=ignore_hierarchy,
        )

        kwargs = dict(
            design_space=norm_ds_spec.design_space,
            categorical_kernel=MixIntKernelType.GOWER,
            hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
            print_global=False,
            print_training=False,
            print_prediction=False,
            print_problem=False,
            print_solver=False,
        )
        kwargs.update(kwargs_)

        # ADORE problems always set this
        # noinspection PyUnresolvedReferences
        evaluator = self.problem.evaluator
        gp = evaluator.translator.graph_processor

        output_names = self._get_output_names(evaluator)

        surrogate = ADSGKriging(
            graph_processor=gp,
            normalization=normalization,
            structure_kernels=structure_kernels,
            sizing_kernel=sizing_kernel,
            sizing_kernel_mode=sizing_kernel_mode,
            composition=composition,
            **kwargs,
        )

        if ignore_hierarchy:
            surrogate.supports["x_hierarchy"] = False
        if multi:
            surrogate = LoggedMultiSurrogateModel(surrogate, output_names=output_names)
        else:
            surrogate = LoggedSurrogateModel(surrogate, output_name=output_names[0])

        return surrogate, normalization

    @staticmethod
    def _get_output_names(evaluator) -> list[str]:
        objective_names = [obj.ref if obj.ref is not None else obj.name for obj in evaluator.objectives]
        constraint_names = [con.ref if con.ref is not None else con.name for con in evaluator.constraints]
        return [
            f"f{i}_{ADSGKriging._kernel_weight_metric_name(name)}"
            for i, name in enumerate(objective_names)
        ] + [
            f"g{i}_{ADSGKriging._kernel_weight_metric_name(name)}"
            for i, name in enumerate(constraint_names)
        ]
