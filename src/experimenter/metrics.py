import numpy as np

from typing import *
from experimenter.metric_base import IndicatorMetric, Metric

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.indicators.distance_indicator import euclidean_distance

class SpreadMetric(Metric):
    """
    Spread measures how well-spread a Pareto front is, representing the exploration performance of the algorithm. This
    metric only works for problems with 2 objectives. A value of 0 indicates a perfectly uniform spread.

    Implementation based on:
    Deb, K., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", 2002, 10.1109/4235.996017
    """

    @property
    def name(self) -> str:
        return 'spread'

    @property
    def value_names(self) -> List[str]:
        return ['delta']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if algorithm.problem.n_obj != 2:
            return [np.nan]

        # Get objective values of the current Pareto front (n_opt, n_obj), and sort along the first objective
        f = self._get_opt_f(algorithm)
        f = f[np.argsort(f[:, 0]), :]

        if f.shape[0] < 3:
            return [1.]

        dists = euclidean_distance(f[:-1, :], f[1:, :], norm=1)
        extreme_dists = dists[0]+dists[-1]  # d_f + d_i

        internal_dists = dists[1:-1]
        d_mean = np.mean(internal_dists)
        n_internal = len(internal_dists)

        # Equation (1), page 7 (188)
        delta = (extreme_dists + np.sum(np.abs(internal_dists - d_mean))) /\
                (extreme_dists + n_internal*d_mean)
        return [delta]


class DeltaHVMetric(Metric):
    """
    Metric measuring the difference to the pre-known hypervolume. It has a value between 1 and 0, where 0 means the
    hypervolume is exactly the same, meaning the true Pareto front has been found.

    Implementation based on:
    Palar, P.S., "On Multi-Objective Efficient Global Optimization Via Universal Kriging Surrogate Model", 2017,
        10.1109/CEC.2017.7969368
    """

    def __init__(self, pf: np.ndarray, perc_pass: List[float] = None):
        self.is_one_dim = pf.shape[0] == 1
        self.max_f = np.max(pf, axis=0)
        self.pf_0 = pf[0, :]
        self._hv = hv = Hypervolume(pf=pf, zero_to_one=True)
        self.hv_true = hv.do(pf)
        if self.hv_true > 1.:
            raise RuntimeError(f'Check normalization: HV = {self.hv_true}')
        self.delta_hv0 = None

        self.perc_pass = perc_pass if perc_pass is not None else []
        self.i_iter = 0
        self.prev_regret = 0
        self.prev_regret_abs = 0
        self.prev_ratio = 1
        self.prev_abs = None
        self.is_passed = None
        self.prev_n_eval = None

        super(DeltaHVMetric, self).__init__()

    @property
    def name(self) -> str:
        return 'delta_hv'

    @property
    def value_names(self) -> List[str]:
        pp_names = [f'pass_{pp*100:.0f}' for pp in self.perc_pass]
        return ['delta_hv', 'hv', 'true_hv', 'ratio', 'regret', 'abs_regret']+pp_names

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        f_opt = self._get_opt_f(algorithm, feasible_only=True)
        f_all = self._get_pop_f(algorithm, valid_only=True)
        return self.calculate_delta_hv(f_opt, f_all, algorithm=algorithm)

    def calculate_delta_hv(self, f_opt: np.ndarray, f_all: np.ndarray, algorithm=None) -> List[float]:
        def _get_regret(abs_rel_dist, ratio):
            if algorithm is None:
                n_infill = 1
            else:
                n_eval = algorithm.evaluator.n_eval
                if self.prev_n_eval is None:
                    n_infill = 0
                else:
                    n_infill = n_eval-self.prev_n_eval
                self.prev_n_eval = n_eval

            # The target value is zero delta ratio, so regret is simply the integral under the ratio curve
            new_regret = self.prev_regret + .5*(ratio+self.prev_ratio)*n_infill
            self.prev_regret = new_regret
            self.prev_ratio = ratio

            if self.prev_abs is None:
                self.prev_abs = abs_rel_dist
            new_abs_regret = self.prev_regret_abs + .5*(abs_rel_dist+self.prev_abs)*n_infill
            self.prev_regret_abs = new_abs_regret
            self.prev_abs = abs_rel_dist

            return new_regret, new_abs_regret

        def _get_iter_p_passed(ratio):
            if self.is_passed is None:
                self.is_passed = [None]*len(self.perc_pass)
            for i_pass, perc in enumerate(self.perc_pass):
                if self.is_passed[i_pass] is None and ratio <= perc:
                    self.is_passed[i_pass] = self.i_iter

            self.i_iter += 1
            return [passed if passed is not None else np.nan for passed in self.is_passed]

        # If there are no optimal points (e.g. if all points are infeasible or failed)
        if len(f_opt) == 0 or f_opt.shape[0] == 0 or f_opt.shape[1] == 0:
            if self.prev_n_eval is None and algorithm is not None:
                self.prev_n_eval = algorithm.evaluator.n_eval

            _get_iter_p_passed(1.)
            res = [np.nan for _ in range(len(self.value_names))]
            res[3] = 1.
            return res

        # If we have only one point, calculate the relative distance to the optimal point instead (because true HV is 0)
        if self.is_one_dim:
            # Update max points
            if len(f_all) == 0:
                if self.max_f is not None:
                    return [np.nan]*len(self.value_names)
                max_f = self.max_f
            else:
                self.max_f = max_f = np.max(np.row_stack([f_all, [self.max_f]]), axis=0)

            # Update maximum distance to the optimal point (this represents the extent of the design space)
            true_dist = max_f-self.pf_0
            true_dist[true_dist == 0] = 1
            true_dist_m = np.sqrt(np.sum(true_dist**2))

            f_rel_dist = (f_opt-self.pf_0)/true_dist
            f_rel_min_dist = np.min(np.sum(f_rel_dist, axis=1))

            if self.delta_hv0 is None:
                self.delta_hv0 = f_rel_min_dist
            f_ratio = f_rel_min_dist/self.delta_hv0

            regret, abs_regret = _get_regret(f_rel_min_dist, f_ratio)
            res = [f_rel_min_dist, f_rel_min_dist*true_dist_m, true_dist_m, f_ratio, regret, abs_regret]+_get_iter_p_passed(f_ratio)
            return res

        # Calculate current hypervolume
        try:
            hv = self._hv.do(f_opt)
        except IndexError:
            print(f_opt, len(f_opt), repr(f_opt))
            raise

        # Calculate error metric
        delta_hv = (self.hv_true-hv)/self.hv_true

        if self.delta_hv0 is None:
            self.delta_hv0 = delta_hv
        delta_hv_ratio = delta_hv/self.delta_hv0

        regret, abs_regret = _get_regret(delta_hv, delta_hv_ratio)
        return [delta_hv, hv, self.hv_true, delta_hv_ratio, regret, abs_regret]+_get_iter_p_passed(delta_hv_ratio)


class SingleObjectiveGapMetric(Metric):
    """
    Tracks the remaining single-objective optimality gap to a known optimum.
    """

    def __init__(self, pf: np.ndarray):
        self.f_opt = float(np.min(pf[:, 0]))
        self.gap0 = None
        self.prev_regret = 0
        self.prev_abs_regret = 0
        self.prev_ratio = 1
        self.prev_gap = None
        self.prev_n_eval = None
        super(SingleObjectiveGapMetric, self).__init__()

    @property
    def name(self) -> str:
        return 'best_gap'

    @property
    def value_names(self) -> List[str]:
        return ['gap', 'ratio', 'regret', 'abs_regret', 'f_best', 'f_opt']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        f_opt = self._get_opt_f(algorithm, feasible_only=True)

        if self.prev_n_eval is None:
            n_infill = 0
        else:
            n_infill = algorithm.evaluator.n_eval-self.prev_n_eval
        self.prev_n_eval = algorithm.evaluator.n_eval

        if len(f_opt) == 0 or f_opt.shape[0] == 0 or f_opt.shape[1] == 0:
            self.prev_regret += self.prev_ratio*n_infill
            return [np.nan, 1., self.prev_regret, self.prev_abs_regret, np.nan, self.f_opt]

        f_best = float(np.min(f_opt[:, 0]))
        gap = max(f_best-self.f_opt, 0.)
        if self.gap0 is None:
            self.gap0 = gap if gap != 0 else 1.

        ratio = gap/self.gap0
        self.prev_regret += .5*(ratio+self.prev_ratio)*n_infill
        self.prev_ratio = ratio

        if self.prev_gap is None:
            self.prev_gap = gap
        self.prev_abs_regret += .5*(gap+self.prev_gap)*n_infill
        self.prev_gap = gap

        return [gap, ratio, self.prev_regret, self.prev_abs_regret, f_best, self.f_opt]


class IGDMetric(IndicatorMetric):
    """Inverse generational distance to the known pareto front."""

    def __init__(self, pf):
        super(IGDMetric, self).__init__(IGD(pf, normalize=True))


class IGDPlusMetric(IndicatorMetric):
    """Inverse generational distance (improved) to the known pareto front."""

    def __init__(self, pf):
        super(IGDPlusMetric, self).__init__(IGDPlus(pf, normalize=True))


class MaxConstraintViolationMetric(Metric):
    """Metric that simply returns the maximum constraint violation of the current population."""

    def __init__(self):
        super(MaxConstraintViolationMetric, self).__init__()

        self._total_pop = None
        self._el_dup = DefaultDuplicateElimination()

    @property
    def name(self) -> str:
        return 'max_cv'

    @property
    def value_names(self) -> List[str]:
        return ['max_cv', 'min_cv', 'pop_max_cv', 'pop_min_cv', 'frac_nan']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if self._total_pop is None:
            self._total_pop = self._get_pop(algorithm)
        else:
            pop = Population.merge(self._total_pop, self._get_pop(algorithm))
            self._total_pop = self._el_dup.do(pop)

        cv = self._get_opt_cv(algorithm)
        if len(cv) == 0:
            return [0., 0., 0., 0., 0.]
        cv[np.isinf(cv)] = np.nan

        cv_pop = self._get_pop_cv(algorithm)
        cv_pop[np.isinf(cv_pop)] = np.nan

        cv_total_pop = self._total_pop.get('CV')
        cv_total_pop[np.isinf(cv_total_pop)] = np.nan
        frac_nan = np.sum(np.isnan(cv_total_pop))/len(cv_total_pop)

        return [np.nanmax(cv), np.nanmin(cv), np.nanmax(cv_pop), np.nanmin(cv_pop), frac_nan]


class NrEvaluationsMetric(Metric):
    """Metric that tracks the number of function evaluations after each algorithm step."""

    @property
    def name(self) -> str:
        return 'n_eval'

    @property
    def value_names(self) -> List[str]:
        return ['n_eval']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        return [algorithm.evaluator.n_eval]


class BestObjMetric(Metric):
    """Metric that tracks the current best (feasible) objective values."""

    def __init__(self, i_f=0):
        self.i_f = i_f
        super(BestObjMetric, self).__init__()

    @property
    def name(self):
        return 'f_best'

    @property
    def value_names(self) -> List[str]:
        return ['f_best']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if algorithm.opt is not None:
            return [algorithm.opt.get('F')[self.i_f, 0]]
        return [np.nan]


class SurrogateTrainFitMetric(Metric):
    """Prediction quality on the current SBO surrogate training set."""

    @property
    def name(self) -> str:
        return 'surrogate_train_fit'

    @property
    def value_names(self) -> List[str]:
        return ['pearson_mean', 'pearson_min', 'pearson_max', 'mse_norm', 'rmse_norm', 'mae_norm', 'n_points', 'n_valid']

    @staticmethod
    def _pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.sum(mask) < 2:
            return np.nan

        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
            return np.nan

        return float(np.corrcoef(y_true, y_pred)[0, 1])

    @staticmethod
    def _as_2d(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return values[:, None]
        return values

    @staticmethod
    def _empty_values() -> List[float]:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0]

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        infill = getattr(algorithm, 'infill_obj', None)
        if infill is None:
            return self._empty_values()

        x_train = getattr(infill, 'x_train', None)
        y_train = getattr(infill, 'y_train', None)
        if x_train is None or y_train is None:
            return self._empty_values()

        x_train = self._as_2d(x_train)
        y_train = self._as_2d(y_train)

        surrogate_model = getattr(infill, 'surrogate_model', None)
        normalization = getattr(infill, 'normalization', None)
        if surrogate_model is None or normalization is None:
            return self._empty_values()

        is_active_train = getattr(infill, 'is_active_train', None)

        x_norm = normalization.forward(x_train)

        kwargs = {}
        supports = getattr(surrogate_model, 'supports', {})
        if supports.get('x_hierarchy', False) and is_active_train is not None:
            kwargs['is_acting'] = is_active_train

        try:
            y_pred = surrogate_model.predict_values(x_norm, **kwargs)
        except TypeError:
            y_pred = surrogate_model.predict_values(x_norm)
        except Exception:
            return self._empty_values()

        y_pred = self._as_2d(y_pred)
        n_obj = int(algorithm.problem.n_obj)
        n_outputs = min(n_obj, y_train.shape[1], y_pred.shape[1])
        if n_outputs == 0:
            return self._empty_values()

        y_true_obj = y_train[:, :n_outputs]
        y_pred_obj = y_pred[:, :n_outputs]

        corrs = [
            self._pearson_corr(y_true_obj[:, i], y_pred_obj[:, i])
            for i in range(n_outputs)
        ]
        corrs = np.asarray(corrs, dtype=float)
        valid_corrs = corrs[np.isfinite(corrs)]

        err = y_pred_obj - y_true_obj
        finite_err = np.isfinite(err)
        if not np.any(finite_err):
            return self._empty_values()

        sq_err = err[finite_err] ** 2
        abs_err = np.abs(err[finite_err])
        if valid_corrs.size == 0:
            pearson_mean = np.nan
            pearson_min = np.nan
            pearson_max = np.nan
        else:
            pearson_mean = float(np.mean(valid_corrs))
            pearson_min = float(np.min(valid_corrs))
            pearson_max = float(np.max(valid_corrs))

        mse_mean = float(np.mean(sq_err))
        return [
            pearson_mean,
            pearson_min,
            pearson_max,
            mse_mean,
            float(np.sqrt(mse_mean)),
            float(np.mean(abs_err)),
            int(x_train.shape[0]),
            int(np.sum(finite_err)),
        ]


SurrogatePearsonMetric = SurrogateTrainFitMetric


class DoeSizeMetric(Metric):
    """Tracks the surrogate training set size at the current algorithm step."""

    @property
    def name(self) -> str:
        return 'doe'

    @property
    def value_names(self) -> List[str]:
        return ['size']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        infill = getattr(algorithm, 'infill_obj', None)
        x_train = getattr(infill, 'x_train', None) if infill is not None else None
        if x_train is None:
            return [np.nan]
        return [int(x_train.shape[0])]


class SBOTimeMetric(Metric):
    """Tracks SBO model training and surrogate infill search timings."""

    @property
    def name(self) -> str:
        return 'time'

    @property
    def value_names(self) -> List[str]:
        return ['sample', 'pretrain', 'train', 'infill', 'infill_eval', 'infill_n_eval', 'infill_eval_per_point']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        infill = getattr(algorithm, 'infill_obj', None)
        if infill is None:
            return [np.nan] * len(self.value_names)

        # One-time encoder setup cost, if the surrogate uses a pretrained embedder:
        # 'sample' = time to resample the embedder's training graphs, 'pretrain' = time
        # to train the encoder to its snapshot epoch. Surrogates without an embedder
        # (e.g. GP models) report NaN for both.
        surrogate = getattr(infill, '_surrogate_model_base', None)
        embedder = getattr(surrogate, 'embedder', None)
        sample_seconds = getattr(embedder, 'sample_seconds', None)
        pretrain_seconds = getattr(embedder, 'pretrain_seconds', None)
        sample_time = float(sample_seconds) if sample_seconds is not None else np.nan
        pretrain_time = float(pretrain_seconds) if pretrain_seconds is not None else np.nan

        train_time = infill.time_train if infill.time_train is not None else np.nan
        infill_time = infill.time_infill if infill.time_infill is not None else np.nan
        infill_obj = getattr(infill, 'infill', None)
        if infill_obj is None:
            return [sample_time, pretrain_time, train_time, infill_time, np.nan, np.nan, np.nan]

        eval_time = float(infill_obj.time_eval_infill)
        n_eval = int(infill_obj.n_eval_infill)
        eval_per_point = eval_time / n_eval if n_eval > 0 else np.nan
        return [sample_time, pretrain_time, train_time, infill_time, eval_time, n_eval, eval_per_point]


class ADSGKernelWeightMetric(Metric):
    """Tracks optimized additive ADSG kernel branch weights when exposed by the surrogate."""

    def __init__(self):
        self._value_names = []
        self._n_steps = 0
        super(ADSGKernelWeightMetric, self).__init__()

    @property
    def name(self) -> str:
        return 'kernel_weight'

    @property
    def value_names(self) -> List[str]:
        return self._value_names

    def calculate_step(self, algorithm: Algorithm):
        values = self._get_kernel_weight_values(algorithm)

        for value_name in values:
            if value_name not in self.values:
                self._value_names.append(value_name)
                self.values[value_name] = [np.nan] * self._n_steps

        for value_name in self._value_names:
            self.values[value_name].append(values.get(value_name, np.nan))

        self._n_steps += 1

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        raise NotImplementedError("ADSGKernelWeightMetric implements calculate_step directly")

    @staticmethod
    def _get_surrogate(algorithm: Algorithm):
        infill = getattr(algorithm, 'infill_obj', None)
        if infill is None:
            return None

        surrogate = getattr(infill, 'surrogate_model', None)
        if surrogate is not None:
            return surrogate

        return getattr(infill, '_surrogate_model_base', None)

    def _get_kernel_weight_values(self, algorithm: Algorithm) -> dict[str, float]:
        surrogate = self._get_surrogate(algorithm)
        if surrogate is None:
            return {}
        return surrogate.get_kernel_weight_values()


class ADSGKernelDiagnosticsMetric(Metric):
    """Tracks per-branch kernel diagnostics (e.g. effective depth) exposed by the surrogate."""

    def __init__(self):
        self._value_names = []
        self._n_steps = 0
        super(ADSGKernelDiagnosticsMetric, self).__init__()

    @property
    def name(self) -> str:
        return 'kernel_diagnostics'

    @property
    def value_names(self) -> List[str]:
        return self._value_names

    def calculate_step(self, algorithm: Algorithm):
        values = self._get_kernel_diagnostic_values(algorithm)

        for value_name in values:
            if value_name not in self.values:
                self._value_names.append(value_name)
                self.values[value_name] = [np.nan] * self._n_steps

        for value_name in self._value_names:
            self.values[value_name].append(values.get(value_name, np.nan))

        self._n_steps += 1

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        raise NotImplementedError("ADSGKernelDiagnosticsMetric implements calculate_step directly")

    def _get_kernel_diagnostic_values(self, algorithm: Algorithm) -> dict[str, float]:
        surrogate = ADSGKernelWeightMetric._get_surrogate(algorithm)
        if surrogate is None:
            return {}
        return surrogate.get_kernel_diagnostic_values()


class SurrogateMetricsMetric(Metric):
    """Tracks generic metrics exposed by a surrogate through get_metric_values()."""

    def __init__(self):
        self._value_names = []
        self._n_steps = 0
        super(SurrogateMetricsMetric, self).__init__()

    @property
    def name(self) -> str:
        return 'surrogate'

    @property
    def value_names(self) -> List[str]:
        return self._value_names

    def calculate_step(self, algorithm: Algorithm):
        values = self._get_surrogate_metric_values(algorithm)

        for value_name in values:
            if value_name not in self.values:
                self._value_names.append(value_name)
                self.values[value_name] = [np.nan] * self._n_steps

        for value_name in self._value_names:
            self.values[value_name].append(values.get(value_name, np.nan))

        self._n_steps += 1

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        raise NotImplementedError("SurrogateMetricsMetric implements calculate_step directly")

    @staticmethod
    def _get_surrogate_metric_values(algorithm: Algorithm) -> dict[str, float]:
        surrogate = ADSGKernelWeightMetric._get_surrogate(algorithm)
        if surrogate is None:
            return {}
        get_values = getattr(surrogate, 'get_metric_values', None)
        if not callable(get_values):
            return {}
        try:
            return get_values(algorithm=algorithm)
        except TypeError:
            return get_values()


def get_kernel_weight_metric(algorithm: Algorithm):
    infill = getattr(algorithm, 'infill_obj', None)
    if infill is None:
        return None

    surrogate = getattr(infill, '_surrogate_model_base', None)
    if surrogate is None:
        return None

    get_names = getattr(surrogate, 'get_kernel_weight_names', None)
    if not callable(get_names):
        return None

    if len(get_names()) <= 1:
        return None

    return ADSGKernelWeightMetric()


def get_kernel_diagnostic_metric(algorithm: Algorithm):
    infill = getattr(algorithm, 'infill_obj', None)
    if infill is None:
        return None

    surrogate = getattr(infill, '_surrogate_model_base', None)
    if surrogate is None:
        return None

    get_names = getattr(surrogate, 'get_kernel_diagnostic_names', None)
    if not callable(get_names):
        return None

    if len(get_names()) == 0:
        return None

    return ADSGKernelDiagnosticsMetric()


def get_surrogate_metric(algorithm: Algorithm):
    infill = getattr(algorithm, 'infill_obj', None)
    if infill is None:
        return None

    surrogate = getattr(infill, '_surrogate_model_base', None)
    if surrogate is None:
        return None

    get_values = getattr(surrogate, 'get_metric_values', None)
    if not callable(get_values):
        return None

    return SurrogateMetricsMetric()


def get_metrics_single(pf: np.ndarray):
    return [
        SingleObjectiveGapMetric(pf),
        # MaxConstraintViolationMetric(),
        NrEvaluationsMetric(),
        BestObjMetric(),
        DoeSizeMetric(),
        SBOTimeMetric(),
    ]


def get_metrics_multi(pf: np.ndarray):
    return [
        DeltaHVMetric(pf),
        IGDMetric(pf),
        SpreadMetric(),
        # MaxConstraintViolationMetric(),
        NrEvaluationsMetric(),
        BestObjMetric(),
        DoeSizeMetric(),
        SBOTimeMetric(),
    ]
