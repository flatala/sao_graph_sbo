import numpy as np
from sb_arch_opt.problems.rocket_eval import *
from adore.optimization.api.evaluator import *
from adore.api.schema import *


class AdoreRocketEvaluator(GraphApiEvaluator):
    """
    Evaluator for the rocket optimization problem.
    """

    _head_shape_map = {
        'cone': HeadShape.CONE,
        'semisphere': HeadShape.SPHERE,
        'elliptical': HeadShape.ELLIPTICAL,
    }

    def _evaluate(self, architecture: Architecture, arch_qois: List[ArchQOI], **kwargs) -> Dict[ArchQOI, float]:

        lc_norm_stage_bounds = [(0, 1), (.1, .6), (.2, 1)]
        lc_stage1_bounds = [(15, 30), (15, 40), (15, 40)]

        # Get rocket stages
        n_stages = len(architecture.system.systems)
        prev_stage_length = 0
        stages = []
        for stage_sys in architecture.system.systems:
            assert stage_sys.ref == 'stage'

            assert stage_sys.components[0].ref == 'rocket-body'
            assert stage_sys.components[0].qois[0].ref == 'length'
            length = stage_sys.components[0].qois[0].value

            # Map to non-normalized stage length
            i_stage = len(stages)
            norm_bnd, bnd = lc_norm_stage_bounds[i_stage], lc_stage1_bounds[i_stage]
            length = length*(norm_bnd[1]-norm_bnd[0]) + norm_bnd[0]
            if i_stage == 0:
                stage1_bounds = lc_stage1_bounds[n_stages-1]
                length = length*(stage1_bounds[1]-stage1_bounds[0]) + stage1_bounds[0]
            elif i_stage > 0:
                length *= prev_stage_length
            prev_stage_length = length

            engines = []
            for engine_sys in stage_sys.systems:
                assert engine_sys.ref == 'engine-assembly'
                engine = Engine[engine_sys.components[0].name]
                engines.append(engine)

            stages.append(Stage(engines=engines, length=length))

        # Build rocket
        cone_angle = ellipse_l_ratio = 0
        head_shape = None
        vehicle_props = {}
        for comp in architecture.system.components:
            if comp.ref == 'launch-vehicle':
                for qoi in comp.qois:
                    vehicle_props[qoi.ref] = qoi.value

            else:
                head_shape = self._head_shape_map[comp.ref]
                if head_shape == HeadShape.CONE:
                    assert comp.qois[0].ref == 'cone-angle'
                    cone_angle = comp.qois[0].value
                elif head_shape == HeadShape.ELLIPTICAL:
                    assert comp.qois[0].ref == 'length-ratio'
                    ellipse_l_ratio = comp.qois[0].value

        rocket = Rocket(
            stages=stages,
            head_shape=head_shape,
            cone_angle=cone_angle,
            ellipse_l_ratio=ellipse_l_ratio,
            length_diameter_ratio=vehicle_props['l-to-d-ratio'],
            max_q=vehicle_props['max-q'],
            payload_density=vehicle_props['payload-density'],
            orbit_altitude=vehicle_props['orbit-altitude'],
        )

        # Evaluate and get results
        perf = RocketEvaluator.evaluate(rocket)
        results = {
            'cost': np.log10(perf.cost),
            'payload-mass': np.log10(max(1., perf.payload_mass)),
            'structural-constraint': perf.delta_structural,
            'volume-constraint': perf.delta_payload,
            'delta-delta-v': perf.delta_delta_v,
        }
        return {arch_qoi: results.get(arch_qoi.ref) for arch_qoi in arch_qois}


def get_evaluator():
    return AdoreRocketEvaluator.from_file('Rocket_design_problem.adore')


if __name__ == '__main__':
    evaluator = get_evaluator()

    for _ in range(10):
        arch, dv, _ = evaluator.get_architecture(evaluator.get_random_design_vector())
        obj, con = evaluator.evaluate(arch)
        print(f'DV {dv!r} --> OBJ {obj!r}; CON {con!r}')

    problem = evaluator.get_arch_opt_problem(n_parallel=4)
    problem.print_stats()
    from sb_arch_opt.sampling import HierarchicalSampling
    pop = HierarchicalSampling().do(problem, 10)
    problem.evaluate(pop.get('X'))
