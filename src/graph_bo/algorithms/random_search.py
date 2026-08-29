from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.util.misc import has_feasible
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.operators.sampling.rnd import FloatRandomSampling
from sb_arch_opt.util import set_global_random_seed
import numpy as np

__all__ = ["RandomSearchAlgorithm"]

class RandomSearchAlgorithm(Algorithm):
    """
    Random search for pymoo>=0.6:
    - each iteration samples `pop_size` new random points
    - evaluates them
    - merges with current population
    - keeps the best `pop_size` via rank + crowding survival
    """

    def __init__(
        self,
        pop_size=100,
        init_size: int | None = None,
        sampling: Sampling = None,
        repair: Repair = None,
        eliminate_duplicates=DefaultDuplicateElimination(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pop_size = pop_size
        self.init_size = init_size if init_size is not None else pop_size

        if sampling is None:
            sampling = FloatRandomSampling()

        if isinstance(eliminate_duplicates, bool):
            eliminate_duplicates = (
                DefaultDuplicateElimination()
                if eliminate_duplicates
                else NoDuplicateElimination()
            )

        self.initialization = Initialization(
            sampling,
            repair=repair,
            eliminate_duplicates=eliminate_duplicates,
        )

        self.survival = RankAndCrowdingSurvival()

    def _setup(self, problem, **kwargs):
        set_global_random_seed(self.seed)

    def _initialize_infill(self):
        return self.initialization.do(
            self.problem,
            self.init_size,
            algorithm=self,
            random_state=self.random_state,
        )

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survival.do(
            self.problem,
            infills,
            len(infills),
            algorithm=self,
        )

    def _infill(self):
        return self.initialization.do(
            self.problem,
            self.pop_size,
            algorithm=self,
            random_state=self.random_state,
        )

    def _advance(self, infills=None, **kwargs):
        merged = Population.merge(self.pop, infills)
        self.pop = self.survival.do(
            self.problem,
            merged,
            self.pop_size,
            algorithm=self,
        )

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        elif self.problem.n_obj == 1:
            feasible = self.pop[np.ravel(self.pop.get("FEAS"))]
            self.opt = feasible[[np.argmin(np.ravel(feasible.get("F")))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]
