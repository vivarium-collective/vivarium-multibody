import os
import math
import random

from vivarium.core.engine import Engine
from vivarium.core.composition import simulate_experiment, Composite
from vivarium.plots.agents_multigen import plot_agents_multigen

from vivarium_multibody.composites.lattice import Lattice
from vivarium_multibody.composites.grow_divide import GrowDivide
from vivarium_multibody.processes.derive_globals import volume_from_length
from vivarium_multibody.processes.multibody_physics import DEFAULT_BOUNDS, PI, random_body_position

# plotting
from vivarium_multibody.plots.snapshots import (
    plot_snapshots, format_snapshot_data)
from vivarium_multibody.plots.snapshots_video import make_video


#from vivarium_multibody.processes import *


def cellbody_config(config):
    width = 2
    length = 4
    volume = volume_from_length(length, width)
    agent_ids = config['agent_ids']
    bounds = config.get('bounds', DEFAULT_BOUNDS)

    #place agents on surface/cell in box randomly - not mother machine channels
    init_agents = {}
    for agent_id in agent_ids:
        x = random.uniform(0.0, bounds[0])
        y = random.uniform(0.0, bounds[1])
        init_agents[agent_id] = {
            'boundary': {
                'location': [x, y],
                'angle': PI / 2,  # 90 degrees meaning all cells oriented vertically
                'volume': volume,
                'length': length,  # agent length
                'width': width
            }
        }
    return init_agents


def env_configs(n_agents=4):
    bounds = [20.0, 20.0]
    n_bins = [8, 8]
    growth_rate = 0.05
    timestep = 120

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

    #diffusion field
    diffusion_config = {
        'time_step': timestep,  # diffusion update dt
        'molecules': ['glc'],  # glucose to simulate
        'gradient': {
            'type': 'gaussian',
            'molecules': {
                'glc': {
                    'center': [1.0, 1.0],  # gaussian center in domain coordinates
                    'deviation': 3},  # std deviation
            }},
        'diffusion': 4e-3,  # diffusion coefficient
        'n_bins': n_bins,  # grid resolution
        'size': bounds}  # size of the grid

    return {
        'agent_ids': agent_ids,
        'growth_division_config': growth_division_config,
        'environment': {
            'bounds': bounds,
            'diffusion': diffusion_config,
        },
        'initial_state': {
            'agents': init_agents,
        },
    }


def run_biofilm_sim(time=7000, out_dir='out'):
    config = env_configs()
    agent_ids = config['agent_ids']

    #environmental setup - build environment composite
    environment = Lattice(config['environment']).generate({})
    env_comp = Composite(environment)

    for agent_id in agent_ids:
        agent_comp = GrowDivide(config['growth_division_config']).generate({'agent_id': agent_id})
        env_comp.merge(composite=agent_comp, path=('agents', agent_id))

    #vivarium engine
    experiment = Engine(
        processes=env_comp['processes'],
        topology=env_comp['topology'],
        initial_state=config['initial_state'],
        progress_bar=True,
    )

    #simulate
    settings = {'total_time': time, 'return_raw_data': True}
    data = simulate_experiment(experiment=experiment, settings=settings)

    #plots
    plot_settings = {'agents_key': 'agents'}
    plot_agents_multigen(data, plot_settings, out_dir=out_dir)

    agents, fields = format_snapshot_data(data)
    bounds = config['environment']['bounds']
    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=40,
        out_dir=out_dir,
        filename=f"cells_in_box"
    )

    make_video(
        data,
        bounds,
        plot_type='fields',
        step=100,
        out_dir=out_dir,
        filename=f"cells_in_movie"
    )

    print("Starting simulate...")
    settings = {'total_time': time, 'return_raw_data': True}
    data = simulate_experiment(experiment=experiment, settings=settings)
    print("Simulate done.")


if __name__ == '__main__':
    out_dir = os.path.join('out', 'experiments', 'biofilm_cells')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_biofilm_sim(7000, out_dir)
