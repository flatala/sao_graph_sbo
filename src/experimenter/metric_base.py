from typing import Dict, List

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.indicator import Indicator
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sb_arch_opt.problem import ArchOptProblemBase


class Metric:
    """Metric that records one or more values at each optimization step."""

    def __init__(self):
        self.values = {name: [] for name in self.value_names}
        self.values_std = None
        self.values_agg = None

    def calculate_step(self, algorithm: Algorithm):
        values = self._calculate_values(algorithm)
        if len(values) != len(self.value_names):
            raise ValueError("Values should have the same length as the number of values")
        for name, value in zip(self.value_names, values):
            self.values[name].append(value)

    def results(self) -> Dict[str, np.ndarray]:
        return {key: np.array(value) for key, value in self.values.items()}

    def results_std(self) -> Dict[str, np.ndarray]:
        if self.values_std is None:
            return {}
        return {key: np.array(value) for key, value in self.values_std.items()}

    def results_agg(self, agg_key) -> Dict[str, np.ndarray]:
        if self.values_agg is None:
            return {}
        return {key: np.array(value[agg_key]) for key, value in self.values_agg.items()}

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def value_names(self) -> List[str]:
        raise NotImplementedError

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        raise NotImplementedError

    @classmethod
    def _get_pop_x(cls, algorithm: Algorithm, feasible_only=False, valid_only=False) -> np.ndarray:
        return cls._get_pop(algorithm, feasible_only=feasible_only, valid_only=valid_only).get("X")

    @classmethod
    def _get_pop_f(cls, algorithm: Algorithm, feasible_only=False, valid_only=False) -> np.ndarray:
        return cls._get_pop(algorithm, feasible_only=feasible_only, valid_only=valid_only).get("F")

    @classmethod
    def _get_pop_g(cls, algorithm: Algorithm, feasible_only=False, valid_only=False) -> np.ndarray:
        return cls._get_pop(algorithm, feasible_only=feasible_only, valid_only=valid_only).get("G")

    @classmethod
    def _get_pop_cv(cls, algorithm: Algorithm, feasible_only=False, valid_only=False) -> np.ndarray:
        return cls._get_pop(algorithm, feasible_only=feasible_only, valid_only=valid_only).get("CV")

    @classmethod
    def _get_pop(cls, algorithm: Algorithm, feasible_only=False, valid_only=False):
        pop = algorithm.pop
        if valid_only or feasible_only:
            pop = cls.get_valid_pop(pop)
        if feasible_only:
            pop = pop[np.where(pop.get("feasible"))[0]]
        return pop

    @staticmethod
    def get_valid_pop(population: Population) -> Population:
        return population[~ArchOptProblemBase.get_failed_points(population)]

    @classmethod
    def _get_opt_x(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        return cls._get_opt(algorithm, feasible_only=feasible_only).get("X").astype(float)

    @classmethod
    def _get_opt_f(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        return cls._get_opt(algorithm, feasible_only=feasible_only).get("F").astype(float)

    @classmethod
    def _get_opt_g(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        return cls._get_opt(algorithm, feasible_only=feasible_only).get("G").astype(float)

    @classmethod
    def _get_opt_cv(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        return cls._get_opt(algorithm, feasible_only=feasible_only).get("CV").astype(float)

    @staticmethod
    def _get_opt(algorithm: Algorithm, feasible_only=False):
        opt = algorithm.opt
        if feasible_only:
            return opt[np.where(opt.get("feasible"))[0]]
        return opt

    @staticmethod
    def get_pareto_front(f: np.ndarray) -> np.ndarray:
        indices = NonDominatedSorting().do(f, only_non_dominated_front=True)
        return np.copy(f[indices, :])


class IndicatorMetric(Metric):
    def __init__(self, indicator: Indicator):
        super().__init__()
        self.indicator = indicator

    @property
    def name(self) -> str:
        return self.indicator.__class__.__name__

    @property
    def value_names(self) -> List[str]:
        return ["indicator"]

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        f_opt = self._get_opt_f(algorithm)
        if f_opt.shape[0] == 0 or f_opt.shape[1] == 0:
            return [np.nan]
        return [self.indicator.do(f_opt)]
