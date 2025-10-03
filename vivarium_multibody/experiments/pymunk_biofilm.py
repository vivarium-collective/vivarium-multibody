#This script is a similar simulation structure to the mother machine


import os
import random

from vivarium.core.composition import simulate_experiment, Composite
from vivarium.core.engine import Engine
from vivarium.library.units import units
from vivarium.plots.agents_multigen import plot_agents_multigen

from vivarium_multibody.composites.grow_divide import GrowDivide
from vivarium_multibody.composites.lattice import Lattice
# plotting
from vivarium_multibody.plots.snapshots import (
    plot_snapshots, format_snapshot_data)
from vivarium_multibody.plots.snapshots_video import make_video
from vivarium_multibody.processes.derive_globals import volume_from_length
from vivarium_multibody.processes.multibody_physics import DEFAULT_BOUNDS, PI, Multibody


def cellbody_config(config):
    width = 1
    length = 2
    volume = volume_from_length(length, width)
    agent_ids = config['agent_ids']
    bounds = config.get('bounds', DEFAULT_BOUNDS)

    #place agents on surface/cell in box randomly - not mother machine channels
    init_agents = {}
    for agent_id in agent_ids:
        x = 10
        y = width/2
        init_agents[agent_id] = {
            'boundary': {
                'location': [x, y],
                'angle': PI,
                'volume': volume,
                'length': length,  # agent length
                'width': width,
                'mass': 1339 * units.fg,
                'thrust':0,
                'torque':0
            }
        }
    return init_agents


def env_configs(n_agents=1, bounds=None):
    if bounds is None:
        bounds = [20, 20]
    n_bins = [60,60]
    growth_rate = 0.05
    timestep = 1.0

    agent_ids = [str(i) for i in range(n_agents)]

    growth_division_config = {
        'agent_path': ('agents',),
        'global_path': ('global',),
        'growth': {
            'default_growth_rate': growth_rate}
    }

    #inital agent config
    body_config = {
        'bounds': bounds,
        'agent_ids': agent_ids,
    }
    init_agents = cellbody_config(body_config)

    multibody_config = {
        'agent_shape':'circle',
        'bounds': bounds,
        'jitter_force': 1e-4,
        'mother_machine': False,
        'time_step': timestep,
        'animate': False,
        'physics_dt': timestep,
    }


    #diffusion field
    diffusion_config = {
        'time_step': timestep,  # diffusion update dt
        'molecules': ['glc'],  # glucose to simulate
        'gradient': {
            'type': 'gaussian',
            'molecules': {
                'glc': {
                    'center': [10, 2.0],  # gaussian center in domain coordinates
                    'deviation': 5},  # std deviation
            }},
        'diffusion': 0.05,  # diffusion coefficient
        'n_bins': n_bins,  # grid resolution
        'size': bounds}  # size of the grid

    return {
        'agent_ids': agent_ids,
        'growth_division_config': growth_division_config,
        'environment': {
            'bounds': bounds,
            'diffusion': diffusion_config,
        },
        'multibody_config': multibody_config,
        'initial_state': {
            'agents': init_agents,
        },
    }


def run_biofilm_sim(time=20, out_dir='out'):
    config = env_configs()
    agent_ids = config['agent_ids']

    #environmental setup - build environment composite
    environment = Lattice(config['environment']).generate({})
    env_comp = Composite(environment)

    for agent_id in agent_ids:
        agent_comp = GrowDivide(config['growth_division_config']).generate({'agent_id': agent_id})
        env_comp.merge(composite=agent_comp, path=('agents', agent_id))

    #multibody physics process plug in - plugging this in causes cells to go all over the place
    #TODO: cells rendering and making sure cells are on screen, accurately, don't overlay each other
        #Uncomment the line below and they're in frame not all over the place
    multibody_proc = Multibody(config['multibody_config'])
    env_comp['processes']['multibody'] = multibody_proc
    env_comp['topology']['multibody'] = {
        'agents': ('agents',)}  # connect the process's "agents" port to your store at ('agents',)

    #vivarium engine
    experiment = Engine(
        processes=env_comp['processes'],
        topology=env_comp['topology'],
        initial_state=config['initial_state'],
        progress_bar=True,
    )

    #simulate
    settings = {'total_time': time, 'return_raw_data': True, 'emit_step': 5}
    data = simulate_experiment(experiment=experiment, settings=settings)

    #plots - add more features like maybe make color assigned to []
    plot_settings = {'agents_key': 'agents'}
    plot_agents_multigen(
        data,
        plot_settings,
        out_dir=out_dir)

    agents, fields = format_snapshot_data(data)
    bounds = config['environment']['bounds']

    plot_snapshots(
        bounds=config['environment']['bounds'],
        agents=agents,
        fields=fields,
        n_snapshots=10,
        #include_fields=True,
        out_dir=out_dir,
        #phylogeny_names=True,
        filename=f"cells_in_box",
    )


    times = sorted(fields.keys())
    num_frames = len(times)
    #vid_steps = max(1, num_frames // 120)
    vid_steps = num_frames

    make_video(
        data,
        bounds,
        plot_type='fields',
        step=vid_steps,
        out_dir=out_dir,
        filename=f"cells_in_movie"
    )

    #should make a plot for how much glucose overtime
    plot_agents_multigen(data, plot_settings, out_dir=out_dir)


if __name__ == '__main__':
    out_dir = os.path.join('out', 'experiments', 'biofilm_cells')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_biofilm_sim(20, out_dir)
