import os

from vivarium.core.process import Composite
from vivarium.core.composition import (
    compose_experiment,
    COMPOSITE_OUT_DIR,
    FACTORY_KEY
)
from vivarium.library.dict_utils import deep_merge
from vivarium.library.units import units

# processes
from vivarium_multibody.processes.multibody_physics import (
    Multibody,
)
from vivarium_multibody.processes.diffusion_field import (
    DiffusionField,
)
from vivarium_multibody.processes.derive_colony_shape import ColonyShapeDeriver
from vivarium_multibody.composites.grow_divide import GrowDivide


# plots
from vivarium.plots.agents_multigen import plot_agents_multigen
from vivarium_multibody.plots.snapshots import (
    format_snapshot_data,
    plot_snapshots,
)

NAME = 'lattice_environment'


# make a configuration dictionary for the Lattice compartment
def make_lattice_config(
        time_step=None,
        jitter_force=None,
        bounds=None,
        n_bins=None,
        depth=None,
        concentrations=None,
        molecules=None,
        diffusion=None,
        keep_fields_emit=None,
        set_config=None,
        parallel=None,
):
    config = {'multibody': {}, 'diffusion': {}}

    if time_step:
        config['multibody']['time_step'] = time_step
        config['diffusion']['time_step'] = time_step
    if bounds:
        config['multibody']['bounds'] = bounds
        config['diffusion']['bounds'] = bounds
        config['diffusion']['n_bins'] = bounds
    if n_bins:
        config['diffusion']['n_bins'] = n_bins
    if jitter_force:
        config['multibody']['jitter_force'] = jitter_force
    if depth:
        config['diffusion']['depth'] = depth
    if diffusion:
        config['diffusion']['diffusion'] = diffusion
    if concentrations:
        config['diffusion']['gradient'] = {
            'type': 'uniform',
            'molecules': concentrations}
        molecules = list(concentrations.keys())
        config['diffusion']['molecules'] = molecules
    elif molecules:
        # molecules are a list, assume uniform concentrations of 1
        config['diffusion']['molecules'] = molecules
    if keep_fields_emit:
        # by default no fields are emitted
        config['diffusion']['_schema'] = {
            'fields': {
                field_id: {
                    '_emit': False}
                for field_id in molecules
                if field_id not in keep_fields_emit}}
    if parallel:
        config['diffusion']['_parallel'] = True
        config['multibody']['_parallel'] = True
    if set_config:
        config = deep_merge(config, set_config)

    return config



class Lattice(Composite):
    """
    Lattice:  A two-dimensional lattice environmental model with multibody physics and diffusing molecular fields.
    """

    name = NAME
    defaults = {
        # To exclude a process, from the compartment, set its
        # configuration dictionary to None, e.g. colony_mass_deriver
        'multibody': {
            'bounds': [10, 10],
            'size': [10, 10],
        },
        'diffusion': {
            'molecules': ['glc'],
            'n_bins': [10, 10],
            'size': [10, 10],
            'depth': 3000.0,
            'diffusion': 1e-2,
        },
        'colony_shape_deriver': None,
    }

    def generate_processes(self, config):
        processes = {
            'multibody': Multibody(config['multibody']),
            'diffusion': DiffusionField(config['diffusion'])
        }
        colony_shape_config = config['colony_shape_deriver']
        if colony_shape_config is not None:
            processes['colony_shape_deriver'] = ColonyShapeDeriver(
                colony_shape_config)
        return processes

    def generate_topology(self, config):
        topology = {
            'multibody': {
                'agents': ('agents',),
            },
            'diffusion': {
                'agents': ('agents',),
                'fields': ('fields',),
                'dimensions': ('dimensions',),
            },
            'colony_shape_deriver': {
                'colony_global': ('colony_global',),
                'agents': ('agents',),
            }
        }
        return {
            process: process_topology
            for process, process_topology in topology.items()
            if config[process] is not None
        }


def test_lattice(
        config=None,
        n_agents=1,
        total_time=1000
):
    # configure the compartment
    lattice_config = config or make_lattice_config()

    # declare the hierarchy
    agent_ids = [str(agent_id) for agent_id in range(n_agents)]
    hierarchy = {
        FACTORY_KEY: {
            'type': Lattice,
            'config': lattice_config},
        'agents': {
            agent_id: {
                FACTORY_KEY: {
                    'type': GrowDivide,
                    'config': {
                        'agent_id': agent_id,
                        'growth': {
                            'growth_rate': 0.05,  # 0.006 very fast growth
                            'default_growth_noise': 1e-3,
                        },
                        'divide_condition': {
                            'threshold': 3000 * units.fg
                        },
                        '_schema': {}
                    }}
            } for agent_id in agent_ids
        }}

    # configure experiment with helper function compose_experiment()
    initial_state = {
        'agents': {
            agent_id: {
                'boundary': {'mass': 1500 * units.fg}
            } for agent_id in agent_ids
        }}
    experiment_settings = {
        'initial_state': initial_state,
        'experiment_id': 'spatial_environment'}
    spatial_experiment = compose_experiment(
        hierarchy=hierarchy,
        settings=experiment_settings)

    # run the simulation
    spatial_experiment.update(total_time)
    data = spatial_experiment.emitter.get_data_unitless()
    return data


def main():
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    os.makedirs(out_dir, exist_ok=True)

    bounds = [25, 25]
    config = make_lattice_config(
        bounds=bounds,
    )
    data = test_lattice(
        config=config,
        n_agents=1,
        total_time=4000)

    plot_settings = {}
    plot_agents_multigen(data, plot_settings, out_dir)

    agents, fields = format_snapshot_data(data)
    plot_snapshots(
        bounds, agents=agents, fields=fields, out_dir=out_dir)

if __name__ == '__main__':
    main()
