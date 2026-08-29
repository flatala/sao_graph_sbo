import os
import io
import warnings
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from cached_property import cached_property

# Avoid OpenMDAO optional MPI/PETSc import spam in local serial runs.
os.environ["OPENMDAO_REQUIRE_MPI"] = "false"

# Invalid intermediate thermodynamic states are common for failed trial designs.
# We want pymoo/SBArchOpt progress output, not low-level pyCycle/OpenMDAO warnings.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"pycycle\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"openmdao\..*")

import open_turb_arch.evaluation.architecture as el
import open_turb_arch.evaluation.analysis.builder as builder
import open_turb_arch.evaluation.analysis.disciplines as disciplines
from open_turb_arch.evaluation.analysis.balancer import DesignBalancer
from openmdao.core.analysis_error import AnalysisError

from adore.optimization.api.factory_evaluator import *
from adore.graph.api.supplementary import *
from adore.api.schema import *


@dataclass
class PortStub:
    pass


@dataclass
class Connector:
    targets: List[object]


@dataclass
class GearboxData:
    name: str
    gear_ratio: float


def _create_list():
    return []


class SimpleJetEngineEvaluator(ClassFactoryApiEvaluator):
    """
    Class-factory evaluator for the simple jet engine optimization problem.
    """

    opr_bounds = (1.1, 60)
    pr_compressor_perc_bounds = (.1, .9)
    verbose = True

    analysis_problem = builder.AnalysisProblem(design_condition=builder.DesignCondition(
        mach=1e-6,  # Mach number [-]
        alt=0,  # Altitude [ft]
        thrust=150e3,  # Thrust [N]
        turbine_in_temp=1450,  # Turbine inlet temperature [C]
        bleed_offtake=0.5,  # Extraction bleed offtake [kg/s]
        power_offtake=37.5e3,  # Power offtake [W]
        balancer=DesignBalancer(init_turbine_pr=10, init_mass_flow=400, init_extraction_bleed_frac=0.02),
    ))
    max_iter = 30

    @staticmethod
    def get_class_factories() -> List[ClassFactory]:
        port_def = ExternalPortDef(name='ports', auto_match_pattern='*')

        return [
            # Compressor and fan/bypass
            ClassFactory(el=ExternalComponentDef(name='compressor', auto=True), cls=el.Compressor, props=dict(
                name='compressor',
                map=el.CompressorMap.AXI_5,
                pr=ExternalQOIDef(name='pr-param', auto=True),
                mach=.4, eff=.83,
            )),
            ClassFactory(el=ExternalComponentDef(name='fan', auto=True), cls=el.Compressor, props=dict(
                name='fan',
                map=el.CompressorMap.AXI_5,
                pr=ExternalQOIDef(name='fpr', auto=True),
                mach=.4578, eff=.89,
            )),
            ClassFactory(el=ExternalComponentDef(name='splitter', auto=True), cls=el.Splitter, props=dict(
                name='splitter',
                bpr=ExternalQOIDef(name='bpr', auto=True),
                core_mach=.3, bypass_mach=.45,
            )),

            # Core turbomachinery
            ClassFactory(el=ExternalComponentDef(name='burner', auto=True), cls=el.Burner, props=dict(
                name='burner',
                fuel=el.FuelType.JET_A,
                mach=.1, p_loss_frac=.03,
            )),
            ClassFactory(el=ExternalComponentDef(name='turbine', auto=True), cls=el.Turbine, props=dict(
                name='turbine',
                map=el.TurbineMap.LPT_2269,
                mach=.4, eff=.86,
            )),

            # Nozzles
            ClassFactory(el=ExternalComponentDef(name='nozzle', auto=True), cls=el.Nozzle, props=dict(
                name='nozzle_core',
                type=el.NozzleType.CD,
                v_loss_coefficient=.99,
            )),
            ClassFactory(el=ExternalComponentDef(name='bypass-nozzle', auto=True), cls=el.Nozzle, props=dict(
                name='bypass_nozzle',
                type=el.NozzleType.CV,
                fuel_in_air=False,
                v_loss_coefficient=.99,
            )),
            ClassFactory(el=ExternalComponentDef(name='mixed-nozzle', auto=True), cls=el.Nozzle, props=dict(
                name='nozzle_joint',
                type=el.NozzleType.CV,
                fuel_in_air=True,
                v_loss_coefficient=.99,
            )),
            ClassFactory(el=ExternalComponentDef(name='mixer', auto=True), cls=el.Mixer, props=dict(
                name='mixer',
            )),

            # Shaft
            ClassFactory(el=ExternalComponentDef(name='shaft', auto=True), cls=el.Shaft, props=dict(
                name='shaft',
                rpm_design=ExternalQOIDef(name='rpm', auto=True),
                connections=_create_list,
                power_loss=0.,
            )),
            ClassFactory(el=ExternalComponentDef(name='gearbox', auto=True), cls=GearboxData, props=dict(
                name='gearbox',
                gear_ratio=ExternalQOIDef(name='gear-ratio', auto=True),
            )),

            # Offtakes
            ClassFactory(el=ExternalComponentDef(name='generator', auto=True), cls=Connector, props=dict(
                targets=ConnectionValue(conn_target_def=port_def, input_conn=True),
            )),
            ClassFactory(el=ExternalComponentDef(name='bleed-air-duct', auto=True), cls=Connector, props=dict(
                targets=ConnectionValue(conn_target_def=port_def, input_conn=True),
            )),
            ClassFactory(el=port_def, cls=PortStub),
        ]

    @staticmethod
    def get_metrics_factory() -> MetricsFactory:
        return MetricsFactory(metrics={
            'tsfc': ExternalQOIDef(name='tsfc', auto=True),
            'jet_mach': ExternalQOIDef(name='jet-mach', auto=True),
            'pr': ExternalQOIDef(name='pr', auto=True),  # list with length n_shafts
            'pr_perc_sum': ExternalQOIDef(name='pr-perc-sum', auto=True),
            'diameter': ExternalQOIDef(name='diameter', auto=True),
            'length': ExternalQOIDef(name='length', auto=True),
            'noise': ExternalQOIDef(name='noise-level', auto=True),
            'nox': ExternalQOIDef(name='nox', auto=True),
            'weight': ExternalQOIDef(name='weight', auto=True),
        })

    def _evaluate(self, architecture: Architecture, arch_qois: List[ArchQOI], **kwargs) -> Dict[ArchQOI, float]:
        engine_arch, pr_constraints = self._build_engine_arch(architecture)
        metrics = self._evaluate_engine_arch(engine_arch)

        metrics.update(pr_constraints)
        return self.process_results(architecture, arch_qois, metrics)

    def _build_engine_arch(self, architecture: Architecture):
        # Get elements by factory name
        object_map: Dict[str, list] = defaultdict(list)
        for factory in self._class_factories:
            name = factory.el.name
            for engine_arch_el in self.instantiate(architecture, factories=name):
                object_map[name].append(engine_arch_el)

        # Order turbo-machinery to start with high pressure (inside to outside)
        n = len(object_map['shaft'])
        names = ['hp', 'ip', 'lp']
        if n == 2:
            names.pop(1)
        elif n == 1:
            names = ['lp']
        for key in ['compressor', 'shaft', 'turbine']:
            for i, obj in enumerate(object_map[key]):
                if i > 0:
                    obj.name = f'{obj.name[:4]}_{names[i]}' if key not in 'shaft' else f'{obj.name}_{names[i]}'

        pr_constraints = self._correct_pr(object_map)
        self._correct_turbo_mach(object_map)
        self._parse_fan_gearbox(object_map)
        self._apply_offtakes(object_map)
        self._correct_nozzle_type(object_map)

        self._connect_airflow(architecture, object_map)
        self._connect_shafts(architecture, object_map)

        return self._get_engine_arch(object_map), pr_constraints

    def _evaluate_engine_arch(self, engine_arch: el.TurbofanArchitecture):
        cycle_builder = builder.CycleBuilder(engine_arch, self.analysis_problem, max_iter=self.max_iter)
        om_problem = cycle_builder.get_problem()
        # cycle_builder.view_n2(om_problem)

        try:
            if self.verbose:
                cycle_builder.run(om_problem, print_solver=True)
            else:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    cycle_builder.run(om_problem, print_solver=False)
        except (AnalysisError, ValueError, FloatingPointError, RuntimeError):
            return {}
        # cycle_builder.print_results(om_problem)

        ops_metrics = cycle_builder.get_metrics(om_problem)[self.analysis_problem.design_condition]
        return {
            'tsfc': ops_metrics.tsfc,
            'jet_mach': ops_metrics.mach_jet,
            'diameter': disciplines.Diameter(ops_metrics, engine_arch).diameter_calculation()[0],
            'length': disciplines.Length(ops_metrics, engine_arch).length_calculation()[0],
            'noise': disciplines.Noise(ops_metrics, engine_arch).noise_calculation(),
            'nox': disciplines.NOx(ops_metrics).NOx_calculation(),
            'weight': disciplines.Weight(ops_metrics, engine_arch).weight_calculation()[0],
        }

    @staticmethod
    def _get_engine_arch(object_map):
        arch_els = defaultdict(list)
        for key, objects in object_map.items():
            for o in objects:
                if isinstance(o, el.ArchElement):
                    o.name = o.name.replace(' ', '_')
                    arch_els[key].append(o)

        arch_els['compressor'] = arch_els['compressor'][::-1]

        elements = []
        for key in ['inlet', 'fan', 'splitter', 'compressor', 'burner', 'turbine', 'gearbox']:
            elements += arch_els.get(key, [])

        for objects in arch_els.values():
            for o in objects:
                if o not in elements:
                    elements.append(o)

        return el.TurbofanArchitecture(elements=elements)

    def _connect_airflow(self, architecture: Architecture, object_map):
        air_graph = self.resolve_sup_dsg(self.airflow_sup_dsg, architecture)
        traversed = {}

        def _trace_airflow(node: SupElement):
            if node in traversed:
                return traversed[node]

            # Special case: connector
            if node.ref == 'connector':
                next_els = air_graph.next(node)
                assert len(next_els) == 1
                return _trace_airflow(next_els[0])

            # Get engine object associated to node
            engine_arch_el = self._get_engine_el(object_map, node.ref)
            assert engine_arch_el is not None
            traversed[node] = engine_arch_el

            # Get connection targets
            next_els = air_graph.next(node)

            # Special case: nozzle
            if isinstance(engine_arch_el, el.Nozzle):
                assert len(next_els) == 0
                return engine_arch_el

            # Special case: splitter (has 2 outputs)
            if isinstance(engine_arch_el, el.Splitter):
                assert len(next_els) == 2
                if next_els[0].name.lower().startswith('compressor'):
                    target_core, target_bypass = next_els
                else:
                    target_core, target_bypass = next_els[::-1]

                engine_arch_el.target_core = _trace_airflow(target_core)
                engine_arch_el.target_bypass = _trace_airflow(target_bypass)

                return engine_arch_el

            # Next trace step
            assert len(next_els) == 1
            target_el = _trace_airflow(next_els[0])
            if hasattr(engine_arch_el, 'target'):
                engine_arch_el.target = target_el

            return engine_arch_el

        # Trace airflow from inlet
        inlet = air_graph.get_by_ref('inlet')
        assert inlet is not None
        inlet_el = el.Inlet(
            name='inlet',
            mach=.6, p_recovery=1,
        )
        object_map['inlet'].append(inlet_el)
        _trace_airflow(inlet)

        # Set mixer sources
        mixer = air_graph.get_by_ref('mixer')
        if mixer is not None:
            mixer_el = self._get_engine_el(object_map, mixer.ref)
            assert isinstance(mixer_el, el.Mixer)

            sources = air_graph.prev(mixer)
            assert len(sources) == 2
            if sources[0].ref.startswith('splitter'):
                core_src, bypass_src = sources[::-1]
            else:
                core_src, bypass_src = sources
            core_src = air_graph.prev(core_src)[0]  # Skip the turbine connector

            for i, source in enumerate([core_src, bypass_src]):
                source_el = self._get_engine_el(object_map, source.ref)
                assert isinstance(source_el, (el.Turbine, el.Splitter))
                source_el.flow_out = f'Fl_I{i+1}'
                setattr(mixer_el, f'source_{i+1}', source_el)

    def _connect_shafts(self, architecture: Architecture, object_map):
        shaft_graph = self.resolve_sup_dsg(self.shafts_sup_dsg, architecture)

        shaft: el.Shaft
        for i, shaft in enumerate(object_map['shaft']):
            shaft_sup_el = shaft_graph.get_by_ref(f'shaft_{i}')
            assert shaft_sup_el is not None

            for connected_sup_el in shaft_graph.next(shaft_sup_el):
                connected_el = self._get_engine_el(object_map, connected_sup_el.ref)
                if connected_el is None:
                    continue
                if isinstance(connected_el, (el.Compressor, el.Turbine)):
                    shaft.connections.insert(0, connected_el)  # Gearbox should be last if present

    @staticmethod
    def _get_engine_el(object_map, ref) -> Optional[el.ArchElement]:
        if ref is None:
            return
        engine_arch_els = object_map.get(ref.split('_')[0], [])
        if len(engine_arch_els) == 0:
            return

        el_idx = int(ref.split('_')[1]) if '_' in ref else 0
        engine_arch_el: el.ArchElement = engine_arch_els[el_idx]
        return engine_arch_el

    def _correct_pr(self, object_map):
        compressors: List[el.Compressor] = object_map['compressor']
        assert 1 <= len(compressors) <= 3

        # Overall Pressure Ratio is determined by first PR param (normalized)
        opr = compressors[0].pr*(self.opr_bounds[1]-self.opr_bounds[0]) + self.opr_bounds[0]

        # Reduce by Fan PR
        fans: List[el.Compressor] = object_map.get('fan', [])
        if len(fans) == 1:
            opr = opr / fans[0].pr

        if len(compressors) == 1:
            compressors[0].pr = opr
            return {
                'pr': [opr],
                'pr_perc_sum': 0,
            }

        # Get requested percentages for subsequent compressors
        perc_bnd = self.pr_compressor_perc_bounds
        pr_perc = np.zeros((len(compressors),))
        for i, compressor in list(enumerate(compressors))[1:]:
            pr_perc[i] = compressor.pr*(perc_bnd[1]-perc_bnd[0]) + perc_bnd[0]
        if sum(pr_perc) >= 1:
            pr_perc[1:] = 1/3
        pr_perc[0] = 1-np.sum(pr_perc[1:])

        # Convert to pressure ratio (ratios are multiplicative)
        pr = 10**(np.log10(opr)*pr_perc)

        # # Like open_turb_arch.architecting.turbofan.shafts_number::ShaftChoice.modify_architecture (lines 121 to 128)
        # if len(compressors) == 2:
        #     pr_base = np.sqrt(opr / (pr_perc[1] - pr_perc[1]**2))
        # elif len(compressors) == 3:
        #     pr_base = (opr / (pr_perc[1]*pr_perc[2] - pr_perc[1]**2*pr_perc[2] - pr_perc[1]*pr_perc[2]**2)) ** (1/3)
        # else:
        #     raise NotImplementedError
        # pr = pr_base*pr_perc

        # Set pressure ratios
        assert np.abs(np.prod(pr) - opr) < 1e-3
        for i, compressor in enumerate(compressors):
            compressor.pr = pr[i]

        # Get values for constraints
        return {
            'pr': pr,
            'pr_perc_sum': sum(pr_perc[1:]),
        }

    @staticmethod
    def _correct_turbo_mach(object_map):
        # open_turb_arch.architecting.turbofan.shafts_number::ShaftChoice._add_shafts (line 201)
        compressors: List[el.Compressor] = object_map['compressor']
        for i, compressor in enumerate(compressors):
            if i > 0:
                compressor.mach = compressors[i-1].mach*1.15

        # open_turb_arch.architecting.turbofan.shafts_number::ShaftChoice._add_shafts (line 206)
        turbines: List[el.Turbine] = object_map['turbine']
        for i, turbine in enumerate(turbines):
            if i > 0:
                turbine.mach = turbines[i-1].mach*1.15

    @staticmethod
    def _parse_fan_gearbox(object_map):
        shafts: List[el.Shaft] = object_map['shaft']
        lp_core_shaft = shafts[-1]

        gearboxes = object_map.get('gearbox', [])
        if len(gearboxes) == 0:
            if len(object_map.get('fan', [])) > 0:
                fan = object_map['fan'][0]
                lp_core_shaft.connections.append(fan)
            return
        gearbox_data: GearboxData = gearboxes[0]

        fan_shaft = el.Shaft(
            name='fan_shaft',
            rpm_design=lp_core_shaft.rpm_design / gearbox_data.gear_ratio,
            connections=[object_map['fan'][0]],
        )
        object_map['fan-shaft'].append(fan_shaft)

        gearbox = el.Gearbox(
            name='gearbox',
            core_shaft=lp_core_shaft,
            fan_shaft=fan_shaft,
        )
        object_map['gearbox'].append(gearbox)

        fan_shaft.connections.append(gearbox)
        lp_core_shaft.connections.append(gearbox)

    @staticmethod
    def _correct_nozzle_type(object_map):
        if len(object_map.get('fan', [])) > 0 and len(object_map.get('nozzle', [])) > 0:
            nozzle: el.Nozzle = object_map['nozzle'][0]
            nozzle.type = el.NozzleType.CV

    @classmethod
    def _apply_offtakes(cls, object_map):
        # Apply power offtake
        power_offtake = cls.analysis_problem.design_condition.power_offtake
        if power_offtake != 0:
            generator: Connector = object_map['generator'][0]
            assert len(generator.targets) == 1

            power_offtake_shaft = generator.targets[0]
            assert isinstance(power_offtake_shaft, el.Shaft)

            power_offtake_shaft.offtake_shaft = True
            power_offtake_shaft.power_offtake = power_offtake

        # Apply bleed offtake
        bleed_offtake = cls.analysis_problem.design_condition.bleed_offtake
        if bleed_offtake != 0:
            bleed_duct: Connector = object_map['bleed-air-duct'][0]
            assert len(bleed_duct.targets) == 1

            bleed_offtake_compressor = bleed_duct.targets[0]
            assert isinstance(bleed_offtake_compressor, el.Compressor)

            bleed_offtake = el.BleedIntra(
                name='bleed_offtake',
                source=bleed_offtake_compressor,
                bleed_names=['bleed_offtake_atmos'],
                source_frac_w=[.02],
            )
            object_map['bleed'].append(bleed_offtake)
            bleed_offtake_compressor.offtake_bleed = True
            bleed_offtake_compressor.bleed_names.append('bleed_offtake_atmos')

    @cached_property
    def airflow_sup_dsg(self) -> SupplementaryDesignSpaceGraph:
        air_sup_dsg = SupplementaryDesignSpaceGraph(self.project, self.translator.factory)

        # Fan choice
        inlet = SupElement('Inlet', ref='inlet')
        fan = SupElement('Fan', ref='fan')
        splitter = SupElement('Splitter', ref='splitter')
        compressor_connector = SupElement('Compressor Conn', ref='connector')

        fan_choice = SupSelectionChoice(
            name='fan_choice',
            element=inlet,
            option_elements=[compressor_connector, fan],
        )
        air_sup_dsg.add_selection_choice(fan_choice)
        air_sup_dsg.add_edges([
            (fan, splitter),
            (splitter, compressor_connector),
        ])

        fan_el = self._get_adore_comp_by_ref('fan')
        air_sup_dsg.add_mapping(ExistenceMapping(
            sup_choice=fan_choice,
            mapping={SourceElementRef(element_id=fan_el.id): fan},
            none_exist_option=compressor_connector,
        ))

        # Compressor stages
        compressor_el = self._get_adore_comp_by_ref('compressor')
        compressors = [SupElement(f'Compressor {i+1}', ref=f'compressor_{i}') for i in range(3)]
        compressor_choice = SupSelectionChoice(
            name='compressor_choice',
            element=compressor_connector,
            option_elements=compressors,
        )
        air_sup_dsg.add_selection_choice(compressor_choice)

        air_sup_dsg.add_mapping(ExistenceMapping(
            sup_choice=compressor_choice,
            mapping={SourceElementRef(element_id=compressor_el.id, component_idx=i): compressors[i]
                     for i in reversed(list(range(len(compressors))))},
            none_exist_option=compressors[0],
        ))

        air_sup_dsg.add_edges([(compressors[i+1], compressors[i]) for i in range(len(compressors)-1)])

        burner = SupElement('Burner', ref='burner')
        turbines = [SupElement(f'Turbine {i+1}', ref=f'turbine_{i}') for i in range(3)]
        air_sup_dsg.add_edges([
            (compressors[0], burner),
            (burner, turbines[0]),
        ])

        # Turbine stages
        turbine_connector = SupElement('Turbine Conn', ref='connector')
        turbine_el = self._get_adore_comp_by_ref('turbine')
        for i in range(len(turbines)-1):
            turbine_choice = SupSelectionChoice(
                name=f'turbine_choice_{i}',
                element=turbines[i],
                option_elements=[turbine_connector, turbines[i+1]],
            )
            air_sup_dsg.add_selection_choice(turbine_choice)

            air_sup_dsg.add_mapping(ExistenceMapping(
                sup_choice=turbine_choice,
                mapping={SourceElementRef(element_id=turbine_el.id, component_idx=i+1): turbines[i+1]},
                none_exist_option=turbine_connector,
            ))

        air_sup_dsg.add_edge(turbines[-1], turbine_connector)

        # Nozzle choice
        core_nozzle = SupElement('Core Nozzle', ref='nozzle')
        bypass_nozzle = SupElement('Bypass Nozzle', ref='bypass-nozzle')

        mixed_nozzle = SupElement('Mixed Nozzle', ref='mixed-nozzle')
        mixer = SupElement('Mixer', ref='mixer')
        air_sup_dsg.add_edge(mixer, mixed_nozzle)
        mixer_el = self._get_adore_comp_by_ref('mixer')

        core_mixer_choice = SupSelectionChoice(
            name='core_mixer_choice',
            element=turbine_connector,
            option_elements=[core_nozzle, mixer],
        )
        air_sup_dsg.add_selection_choice(core_mixer_choice)

        air_sup_dsg.add_mapping(ExistenceMapping(
            sup_choice=core_mixer_choice,
            mapping={SourceElementRef(element_id=mixer_el.id): mixer},
            none_exist_option=core_nozzle,
        ))

        bypass_mixer_choice = SupSelectionChoice(
            name='bypass_mixer_choice',
            element=splitter,
            option_elements=[bypass_nozzle, mixer],
        )
        air_sup_dsg.add_selection_choice(bypass_mixer_choice)

        air_sup_dsg.add_mapping(ExistenceMapping(
            sup_choice=bypass_mixer_choice,
            mapping={SourceElementRef(element_id=mixer_el.id): mixer},
            none_exist_option=bypass_nozzle,
        ))

        return air_sup_dsg

    @cached_property
    def shafts_sup_dsg(self) -> SupplementaryDesignSpaceGraph:
        shaft_sup_dsg = SupplementaryDesignSpaceGraph(self.project, self.translator.factory)

        # Shaft nr choice
        shaft_els = [SupElement(f'Shaft {i+1}', ref=f'shaft_{i}') for i in range(3)]
        shaft_nr_choice = SupSelectionChoice(
            name='shaft_nr',
            element=SupElement('start'),
            option_elements=shaft_els,
        )
        shaft_sup_dsg.add_selection_choice(shaft_nr_choice)

        # Ensure that if shaft 3 is select, also shaft 2 is selected, etc.
        for i, shaft_el in reversed(list(enumerate(shaft_els))):
            if i > 0:
                shaft_sup_dsg.add_edge(shaft_el, shaft_els[i-1])

        # Map to nr of shaft instances
        shaft_adore_comp = self._get_adore_comp_by_ref('shaft')
        shaft_sup_dsg.add_mapping(ExistenceMapping(
            sup_choice=shaft_nr_choice,
            # Map in reverse because if works like else-if's: select shaft 3; else if shaft 2; etc.
            mapping={SourceElementRef(element_id=shaft_adore_comp.id, component_idx=i): shaft_els[i]
                     for i in reversed(list(range(len(shaft_els))))},
            none_exist_option=shaft_els[0],
        ))

        # Connected turbo-machinery
        for i, shaft_el in enumerate(shaft_els):
            shaft_sup_dsg.add_edges([
                (shaft_el, SupElement(f'Compressor {i+1}', ref=f'compressor_{i}')),
                (shaft_el, SupElement(f'Turbine {i+1}', ref=f'turbine_{i}')),
            ])

        return shaft_sup_dsg

    def _get_adore_comp_by_ref(self, ref) -> Component:
        for comp in self.project.design_space.system.components:
            if comp.ref == ref:
                return comp
        raise ValueError(f'Component not found: {ref}')

    @staticmethod
    def get_dv_from_simple_ota(x: np.ndarray):
        # Fan [0], Shaft [3], Gearbox [10], Nozzle [12], Offtakes [13]
        i_dv_map = {
            0: lambda v: 1-v,  # Fan, no fan
            10: lambda v: 1-v,  # Gearbox, no gearbox
            12: lambda v: v,  # Separated, mixed nozzle
            3: lambda v: v,  # Nr of shafts
            14: lambda v: v,  # Bleed offtake
            13: lambda v: v,  # Power offtake
            1: lambda v: v,  # BPR
            2: lambda v: v,  # FPR
            11: lambda v: v,  # Gear ratio
            4: lambda v: ((v-1.1)/(60-1.1)),  # Compr 1, PR param
            5: lambda v: ((v-.1)/(.9-.1)),  # Compr 2, PR param
            6: lambda v: ((v-.1)/(.9-.1)),  # Compr 3, PR param
            7: lambda v: v,  # Shaft 1 RPM
            8: lambda v: v,  # Shaft 2 RPM
            9: lambda v: v,  # Shaft 3 RPM
        }
        i_dv = np.array(list(i_dv_map.keys()))

        x_adore = x*0
        for i, xi in enumerate(x):
            x_adore[i, :] = [i_dv_map[j](xi[j]) for j in i_dv]

        return x_adore

    def validate_simple_ota(self, x: np.ndarray, execute=False):
        from sb_arch_opt.problems.turbofan_arch import SimpleTurbofanArch
        x_adore = self.get_dv_from_simple_ota(x)

        adore_engine_arch = []
        for xi in x_adore:
            architecture, _, _ = self.get_architecture(xi)
            engine_arch, _ = self._build_engine_arch(architecture)
            adore_engine_arch.append((architecture, engine_arch))
        self._arch_cache = {}
        self._inst_cache = {}

        ota_problem = SimpleTurbofanArch()
        # ota_problem._problem.verbose = True

        ota_engine_arch = []
        for xi in x:
            ota_arch, _ = ota_problem._problem.generate_architecture(ota_problem._convert_x(xi))
            ota_engine_arch.append(ota_arch)

        def _el_repr(el__: el.ArchElement):
            import copy
            el_print: el.ArchElement = copy.deepcopy(el__)
            # el_print.name = '*'
            if hasattr(el_print, 'connections'):
                el_print.connections = []
            if hasattr(el_print, 'target'):
                el_print.target = 'TGT'
            if hasattr(el_print, 'fan_shaft'):
                el_print.fan_shaft = el_print.fan_shaft.name
            if hasattr(el_print, 'core_shaft'):
                el_print.core_shaft = el_print.core_shaft.name
            return repr(el_print)

            # import re
            # r = repr(el__)
            # r = re.sub(r"name='[^']*'", 'name', r)
            # return r

        for i, engine_arch in enumerate(ota_engine_arch):
            ota_els = {_el_repr(el_) for el_ in engine_arch.elements}
            adore_els = {_el_repr(el_) for el_ in adore_engine_arch[i][1].elements}

            if len(ota_els ^ adore_els) > 0:
                print(f'DIFF {i}: {x[i, :]}')
                print(f'  OTA EXTRA ({len(ota_els-adore_els)}):')
                for el_ in (ota_els - adore_els):
                    print(f'    {el_}')
                print(f'  ADORE EXTRA ({len(adore_els-ota_els)}):')
                for el_ in (adore_els - ota_els):
                    print(f'    {el_}')
                print(f'  OTA ALL ({len(ota_els)}):')
                for el_ in ota_els:
                    print(f'    {el_}')

        if execute:
            import concurrent.futures
            from tornado.concurrent import DummyExecutor
            with concurrent.futures.ProcessPoolExecutor() as executor:
            # with DummyExecutor() as executor:
                for i in range(len(ota_engine_arch)):
                    futures = [
                        executor.submit(self.evaluate, adore_engine_arch[i][0]),
                        executor.submit(ota_problem._problem.evaluate, ota_problem._convert_x(x[i, :])),
                    ]
                    concurrent.futures.wait(futures)
                    print(f'ADORE RES: {futures[0].result()}')
                    print(f'OTA RES  : {futures[1].result()[1:3]}')


class SimpleJetEngineSurrogateEvaluator(SimpleJetEngineEvaluator):
    """
    Simple engine evaluator that uses the SBArchOpt random-forest model instead of running OpenMDAO.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from sb_arch_opt.problems.turbofan_arch import SimpleTurbofanArchModel
        self._surrogate_problem = SimpleTurbofanArchModel(train=True)

    def _evaluate(self, architecture: Architecture, arch_qois: List[ArchQOI], **kwargs) -> Dict[ArchQOI, float]:
        _, pr_constraints = self._build_engine_arch(architecture)

        design_vector, _ = self.get_design_vector(architecture)
        x_ota = self.get_simple_ota_dv_from_adore(np.array([design_vector], dtype=float))
        out = self._surrogate_problem.evaluate(x_ota, return_as_dictionary=True)

        raw_constraints = out['G'][0, :] / self._surrogate_problem._con_factors + self._surrogate_problem._con_offsets
        metrics = {
            'tsfc': out['F'][0, 0],
            'jet_mach': raw_constraints[0],
        }
        metrics.update(pr_constraints)
        return self.process_results(architecture, arch_qois, metrics)

    @staticmethod
    def get_simple_ota_dv_from_adore(x_adore: np.ndarray):
        x_ota = np.zeros_like(x_adore)
        x_ota[:, 0] = 1 - x_adore[:, 0]  # Fan
        x_ota[:, 10] = 1 - x_adore[:, 1]  # Gearbox
        x_ota[:, 12] = x_adore[:, 2]  # Mixed nozzle
        x_ota[:, 3] = x_adore[:, 3]  # Nr of shafts
        x_ota[:, 14] = x_adore[:, 4]  # Bleed offtake
        x_ota[:, 13] = x_adore[:, 5]  # Power offtake
        x_ota[:, 1] = x_adore[:, 6]  # BPR
        x_ota[:, 2] = x_adore[:, 7]  # FPR
        x_ota[:, 11] = x_adore[:, 8]  # Gear ratio
        x_ota[:, 4] = x_adore[:, 9]*(60-1.1) + 1.1  # Compressor 1 pressure ratio
        x_ota[:, 5] = x_adore[:, 10]*(.9-.1) + .1  # Compressor 2 pressure ratio fraction
        x_ota[:, 6] = x_adore[:, 11]*(.9-.1) + .1  # Compressor 3 pressure ratio fraction
        x_ota[:, 7] = x_adore[:, 12]  # Shaft 1 RPM
        x_ota[:, 8] = x_adore[:, 13]  # Shaft 2 RPM
        x_ota[:, 9] = x_adore[:, 14]  # Shaft 3 RPM
        return x_ota


def get_evaluator():
    engine_cfe = SimpleJetEngineEvaluator.from_file('simple_jet_engine.adore')
    engine_cfe.update_external_database()
    return engine_cfe


def get_surrogate_evaluator():
    engine_cfe = SimpleJetEngineSurrogateEvaluator.from_file('simple_jet_engine.adore')
    engine_cfe.update_external_database()
    return engine_cfe


if __name__ == '__main__':
    evaluator = get_evaluator()
    evaluator.to_file('simple_jet_engine_linked.adore')

    evaluator.shafts_sup_dsg.export_drawio('sup_dsg_shafts.drawio')
    evaluator.airflow_sup_dsg.export_drawio('sup_dsg_airflow.drawio')

    # from sb_arch_opt.problems.turbofan_arch import SimpleTurbofanArch
    # from sb_arch_opt.algo.pymoo_interface.storage_restart import load_from_previous_results
    # p = SimpleTurbofanArch()
    # # evaluator.validate_simple_ota(p.pareto_set(), execute=True)
    # doe_pop = load_from_previous_results(p, 'ota_doe')
    # is_failed = np.any(np.isinf(doe_pop.get('F')), axis=1)
    # evaluator.validate_simple_ota(doe_pop.get('X')[~is_failed, :][26:, :], execute=True)

    for _ in range(10):
        arch, dv, _ = evaluator.get_architecture(evaluator.get_random_design_vector())
        obj, con = evaluator.evaluate(arch)
        print(f'DV {dv!r} --> OBJ {obj!r}; CON {con!r}')

    # problem = evaluator.get_arch_opt_problem(n_parallel=4)
    # assert problem.get_n_valid_discrete() == 70
    # x_all, _ = problem.all_discrete_x
    # assert x_all is not None and x_all.shape == (70, problem.n_var)
    #
    # from sb_arch_opt.sampling import HierarchicalSampling
    # pop = HierarchicalSampling().do(problem, 10)
    # problem.evaluate(pop.get('X'))
