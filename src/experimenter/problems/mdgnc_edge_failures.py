import itertools

import numpy as np
import pandas as pd
from adore.api.schema import *
from adore.optimization.api.evaluator import GraphApiEvaluator


class MDGNCEdgeFailuresCalculator:
    """
    Mixed-discrete GN&C evaluator with explicit edge multiplicity and independent
    edge failures.

    Parallel edges between the same source and target are treated as redundant
    connections. Each edge fails independently with probability p_edge_fail.
    """

    mass = {
        "S": lambda p: 3.0 + 6.0 * p,
        "C": lambda p: 3.0 + 7.0 * p**2,
        "A": lambda p: 3.5 + 6.0 * p,
    }
    conn_mass_penalty = 1.0
    failure_rate = {
        "S": lambda p: 0.00015 - p * 0.0001 * p,
        "C": lambda p: 0.0001 - p * 0.00008 * p**0.5,
        "A": lambda p: 0.0002 - 0.0001 * (2 * (p - 0.55)) ** 2,
    }
    p_edge_fail = 0.0001

    @classmethod
    def calc_mass(cls, sensor_params, computer_params, conns, actuator_params=None, act_conns=None):
        mass = sum(cls.mass["S"](p) for p in sensor_params)
        mass += sum(cls.mass["C"](p) for p in computer_params)
        if actuator_params is not None:
            mass += sum(cls.mass["A"](p) for p in actuator_params)

        n_conn = len(conns)
        if act_conns is not None:
            n_conn += len(act_conns)
        mass += n_conn * cls.conn_mass_penalty
        return mass

    @classmethod
    def calc_failure_rate(cls, sensor_params, computer_params, conns, actuator_params=None, act_conns=None):
        rate = cls.failure_rate
        failure_rates = [
            np.array([rate["S"](p) for p in sensor_params]),
            np.array([rate["C"](p) for p in computer_params]),
        ]
        obj_conns = [conns]
        if actuator_params is not None:
            failure_rates.append(np.array([rate["A"](p) for p in actuator_params]))
            obj_conns.append(act_conns)

        return cls._calc_failure_rate_from_arrays(failure_rates, obj_conns)

    @classmethod
    def _calc_failure_rate_from_arrays(cls, failure_rates, obj_conns):
        conn_matrices = []
        for i, edges in enumerate(obj_conns):
            matrix = np.zeros((len(failure_rates[i]), len(failure_rates[i + 1])), dtype=int)
            for i_src, i_tgt in edges:
                matrix[i_src, i_tgt] += 1
            conn_matrices.append(matrix)

        def _branch_failures(i_rates=0, src_connected_mask=None) -> float:
            calc_downstream = i_rates < len(conn_matrices) - 1
            rates, tgt_rates = failure_rates[i_rates], failure_rates[i_rates + 1]
            conn_mat = conn_matrices[i_rates]

            if src_connected_mask is None:
                src_connected_mask = np.ones((len(rates),), dtype=bool)

            total_rate = 0.0
            for ok_sources in itertools.product(
                *[([False, True] if src_connected_mask[i_conn] else [False]) for i_conn in range(len(rates))]
            ):
                if i_rates > 0 and not any(ok_sources):
                    continue

                ok_sources = np.array(ok_sources, dtype=bool)
                occurrence_prob = rates.copy()
                occurrence_prob[ok_sources] = 1 - occurrence_prob[ok_sources]

                source_prob = 1.0
                for partial_prob in occurrence_prob[src_connected_mask]:
                    source_prob *= partial_prob

                live_edge_counts = conn_mat[ok_sources, :].sum(axis=0)
                edge_up_prob = np.zeros((len(tgt_rates),), dtype=float)
                connected_by_structure = live_edge_counts > 0
                edge_up_prob[connected_by_structure] = 1.0 - cls.p_edge_fail ** live_edge_counts[connected_by_structure]

                total_rate += source_prob * _target_failures(
                    tgt_rates,
                    edge_up_prob,
                    calc_downstream,
                    i_rates,
                )

            return total_rate

        def _target_failures(tgt_rates, edge_up_prob, calc_downstream, i_rates) -> float:
            total_rate = 0.0
            target_options = []
            for prob_connected in edge_up_prob:
                if prob_connected == 0.0:
                    target_options.append([(False, 1.0)])
                else:
                    target_options.append([(False, 1.0 - prob_connected), (True, prob_connected)])

            for connected_targets_with_prob in itertools.product(*target_options):
                connected_targets = np.array([state for state, _ in connected_targets_with_prob], dtype=bool)

                connectivity_prob = 1.0
                for _, partial_prob in connected_targets_with_prob:
                    connectivity_prob *= partial_prob
                if connectivity_prob == 0.0:
                    continue

                tgt_failure_rates = tgt_rates[connected_targets]
                if len(tgt_failure_rates) == 0:
                    total_rate += connectivity_prob
                    continue

                all_tgt_fail_prob = 1.0
                for prob in tgt_failure_rates:
                    all_tgt_fail_prob *= prob
                total_rate += connectivity_prob * all_tgt_fail_prob

                if calc_downstream:
                    total_rate += connectivity_prob * _branch_failures(
                        i_rates=i_rates + 1,
                        src_connected_mask=connected_targets,
                    )

            return total_rate

        return float(np.log10(_branch_failures()))

    @staticmethod
    def plot_results(results_df: pd.DataFrame, evaluator_name: str):
        import matplotlib.pyplot as plt

        obj_cols = [col for col in results_df.columns if col.startswith("OBJ_")]
        pareto_df = results_df.where(results_df.inParetoFront)

        plt.figure(), plt.title(f"MD GNC Edge Failures Results: {evaluator_name}")
        plt.scatter(results_df[obj_cols[0]], results_df[obj_cols[1]], s=5, c="k", label="Architectures")
        plt.scatter(pareto_df[obj_cols[0]], pareto_df[obj_cols[1]], s=20, c="b", label="Pareto front")
        plt.xlabel("System Failure Rate (log_10)"), plt.ylabel("System Mass")
        plt.legend()
        plt.show()


class MDGNCEdgeFailuresEvaluator(GraphApiEvaluator):
    """
    Evaluate the mixed-discrete GNC architecture defined in mdgnc.adore,
    including edge multiplicity and independent edge failures.
    """

    def _evaluate(self, architecture: Architecture, arch_qois: List[ArchQOI], **_) -> Dict[ArchQOI, float]:
        sensor_params = self._get_element_params(architecture, "sensor")
        computer_params = self._get_element_params(architecture, "computer")
        actuator_params = self._get_element_params(architecture, "actuator")
        if len(actuator_params) == 0:
            actuator_params = None

        sensor_computer_conns = self._get_element_connections(architecture, "sensor", "computer")
        computer_actuator_conns = None
        if actuator_params is not None:
            computer_actuator_conns = self._get_element_connections(architecture, "computer", "actuator")

        arch_qoi_map = {}
        for arch_qoi in arch_qois:
            if arch_qoi.ref == "mass":
                arch_qoi_map[arch_qoi] = MDGNCEdgeFailuresCalculator.calc_mass(
                    sensor_params,
                    computer_params,
                    sensor_computer_conns,
                    actuator_params,
                    computer_actuator_conns,
                )
            elif arch_qoi.ref == "failure-rate":
                arch_qoi_map[arch_qoi] = MDGNCEdgeFailuresCalculator.calc_failure_rate(
                    sensor_params,
                    computer_params,
                    sensor_computer_conns,
                    actuator_params,
                    computer_actuator_conns,
                )
        return arch_qoi_map

    @staticmethod
    def _get_element_params(architecture: Architecture, element_ref: str) -> List[float]:
        params = []
        for component in architecture.system.components:
            if component.ref != element_ref:
                continue
            for instance in component.instances:
                if len(instance.qois) == 0:
                    continue
                params.append(float(instance.qois[0].value))
        return params

    @staticmethod
    def _get_element_connections(architecture: Architecture, src_ref: str, tgt_ref: str) -> List[tuple[int, int]]:
        connections = []
        conn_target_idx_map = {}
        for component in architecture.system.components:
            if component.ref == src_ref:
                for instance in component.instances:
                    for out_port_connection in instance.output_ports:
                        for target_id in out_port_connection.target_ids:
                            connections.append((instance.index, target_id))
            elif component.ref == tgt_ref:
                for instance in component.instances:
                    for in_port_connection in instance.input_ports:
                        conn_target_idx_map[in_port_connection.id] = instance.index

        connection_indices = []
        for i_src, target_id in connections:
            if target_id not in conn_target_idx_map:
                raise RuntimeError(f"Target ID not found: {target_id}")
            connection_indices.append((i_src, conn_target_idx_map[target_id]))
        return connection_indices


def get_evaluator(path):
    return MDGNCEdgeFailuresEvaluator.from_file(str(path), save_to_project=True)