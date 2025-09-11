# This file is my commentary/exploration of the mother_machine and multibody processes
    # I went through line by line to understand and comment how this file works and is executed to possibly use it to simulate biofilms
    # Some changes such as agent numbers, variables, timepoints, etc. where changed to explore their outcome on the simulation result

import os
import math
import random

from vivarium.core.engine import Engine
from vivarium.core.composition import simulate_experiment
from vivarium.plots.agents_multigen import plot_agents_multigen

# composites - python classes
from vivarium_multibody.composites.lattice import Lattice
from vivarium_multibody.composites.grow_divide import GrowDivide
from vivarium_multibody.processes.derive_globals import volume_from_length
from vivarium_multibody.processes.multibody_physics import DEFAULT_BOUNDS, PI

# plotting
from vivarium_multibody.plots.snapshots import (
    plot_snapshots, format_snapshot_data)
from vivarium_multibody.plots.snapshots_video import make_video
#from vivarium_multibody.processes import *



#This function gets the initial agent body locations given the mother machine set up
# figures out how many slots fit in the channel, randomly assign agent to slots, return their starting properties
def mother_machine_cellbody_config(config):
    width = 1
    length = 4
    volume = volume_from_length(length, width) #function that gets the cell length from the volume (derive_globals.py)

    agent_ids = config['agent_ids']
    boundaries = config.get('bounds', DEFAULT_BOUNDS)
    channel_space = config.get('channel_space', 1.0)
    n_agents = len(agent_ids)

    # possible locations shuffled for index-in
    # math.floor a math function defined by the C standard
    # returns the floor of x as an integral, this returns the largest integer less than or equal to x, whole numbers
    # boundaries[0] = total width of the channel (x-axis)
    # channel_space = space between agents
    # so boundaries[0]/channel_space = how many agents could fit across the channel

    n_space = math.floor(boundaries[0]/channel_space) #ensures full slots for agents, not partials (cant have half an agent)

    #assert conditions = don't place more agents than available slots (n_space) in the mother machine, otherwise gives message
    assert n_agents < n_space, 'more agents than mother machine spaces'

    # x*channel_space - channel_space / 2  --> place agents location at the center of the slot;
        # channel_space = 1, x position is 0.5, so coordinates is (0.5, 0.01)
                # change the x but the y-coordinate remains the same --> channel's depth or height offset
    # each location is 2D coordinate
    possible_locations = [
        [x*channel_space - channel_space / 2, 0.01]
        for x in range(1, n_space) #integers from 1 up to n_space-1, each x is a slot index across the channel
    ]
    # Randomly rearrange the list of coordinates in place
    # Ensures agents don't go left to right based on x-axis, fill random slots
    random.shuffle(possible_locations)

    # This code creates the initial state for all agents, looping over each agent ID, assigns the agent a position, angle, size
       # returns a dictionary to initialize the simulation's agent stores
        # One big dictionary of all agents, keyed by their ID
    initial_agents = {
        agent_id: {
            'boundary': {
                'location': possible_locations[index],
                'angle': PI/2, #90 degrees meaning all cells oriented vertically
                'volume': volume,
                'length': length, #agent length
                'width': width}}
        for index, agent_id in enumerate(agent_ids)} #goes through list of agent_id in enumerate agent_ids, gives index (pos in list) and agent_id
    return initial_agents #agent dictionary can then be used for initial state in agent stores

def get_mother_config(n_agents=4):
    bounds = [20,20]
    n_bins = [10, 10]
    channel_height = 0.5 * bounds[1]
    channel_space = 1.3
    space_thickness = 0.4
    growth_rate = 0.0006
    time_step = 60

    agent_ids = [str(agent_id) for agent_id in range(n_agents)]

    #growth and division agent configs
    growth_division_config = {
        'agent_path': ('..', '..', 'agents'),
        'global_path': ('global',),
        'growth': {
            'default_growth_rate': growth_rate
        }
    }

    #environment configs

    #multibody config
    #agent mechanics
    multibody_config = {
        'time_step': time_step,
        'jitter_force': 1e-2, #small force to avoid overlapping
        'mother_machine': {
            'space_thickness':space_thickness, #wall thickness
            'channel_height': channel_height, #channel-y size
            'channel_space': channel_space}, #spacing between slots
        'bounds': bounds} #domain size (x,y)

    #inital agent geometry/placement
    #initial_agent is state into agents store at time 0
    body_config = {
        'bounds': bounds,
        'channel_space': channel_space,
        'channel_height': channel_height,
        'agent_ids': agent_ids}
    initial_agents = mother_machine_cellbody_config(body_config)

    #diffusion
    #environment molecule grid of diffusion
    # creates a 2D diffusion grid with a gaussian initial profile
    #bins sets the grid size, size and bounds is the physical extent
    diffusion_config = {
        'time_step': time_step, #diffuion update dt
        'molecules': ['glc'], #glucose to simulate
        'gradient': {
            'type': 'gaussian',
            'molecules': {
                'glc': {
                    'center': [1.0,1.0], #gaussian center in domain coordinates
                    'deviation':3}, #std deviation
            }},
        'diffusion': 4e-3, #diffusion coefficient
        'n_bins': n_bins, #grid resolution
        'size': bounds} #size of the grid

    #return a single config bundle for composite
        #packages initial_state and environmental configs into one dictionary so then the vivarium composer can consume
    return {
        'initial_state':{
            'agents': initial_agents},
        'agent_ids': agent_ids,
        'growth_division_config': growth_division_config,
        'environment': {
            'multibody': multibody_config,
            'diffusion': diffusion_config}}

#function to run the mother machine simulation and all the configs
#run for 5 timepoints, results to out folder
def run_mother_machine(time=5, out_dir='out'):
    #calls the function that builds the full config dictionary for the experiment
    #sets up environment and the 4 agents (initial_state, agent_ids, growth_division_config,environment)
    config = get_mother_config(n_agents=4)

    #extract agent ids, if missing default to 0
    agent_ids = config.get('agent_ids', ['0'])

    #get the environment composite
    #Lattice --> 2D lattice environment model with multibody physics and diffusing molecular fields
        #a composer (generate composites), a "blueprint" that knows how to build composites
            #defaults define configs for 2 processes: multibody (physics/agents) and diffusion (molecular field)
        #generate_processes instantiates these 2 processes
        #generate_topology maps their ports to stores
            #multibody --> ('agents',)
            #diffusion --> ('agents,), ('fields',), ('dimensions',)

    #instantiaing config options from config dictionary
        #if config has an environment key use it, otherwise return empty {}
    environment = Lattice(config.get('environment', {}))

    #every composer has a .generate(config --> convenience method, .generate({}) calls
        #generate_processes
        #generate_topology
    #bundles these into a composite object so now can be run in an experiment
    #after this the composite is the actual thing usable and able to simulate
    composite = environment.generate({})

    #add agents to growth and division
    #this chunk is the full simulation loop - builds agents, runs simulation and plots results
        #GrowDivide is a composer for growth and division process
        #for each agent_id, build a composite for that agent and then insert the agent's composite under the agents store, keyed by agents ID
        #this attaches growth/division behavior to each agent in the simulation
    growth_division = GrowDivide(config.get('growth_division_config', {}))
    for agent_id in agent_ids:
        agent = growth_division.generate({'agent_id': agent_id})
        composite.merge(composite=agent, path=('agents', agent_id))

    #build the experiment - vivarium engine needs processes, topology, initial_state, progress_bar shows runtime progress
        #this packages everything into a runnable simulation
    experiment = Engine(**{
        'processes': composite['processes'],
        'topology': composite['topology'],
        'initial_state': config.get('initial_state', {}),
        'progress_bar': True,
    })

    #run the simulation
    #runs for time units, return_raw_data --> collects raw trajectory of variables into data, so all results stored in data
    settings = {
        'total_time': time,
        'return_raw_data': True,}
    data = simulate_experiment(experiment, settings)

    #creates agent plots lineage over multiple generations and save to out
    plot_settings = {
        'agents_key': 'agents'}
    plot_agents_multigen(data, plot_settings, out_dir=out_dir)

    #snapshot plot
    # environment snapshots
        #format_snapshot_data = extracts agent + field states at different times
    agents,fields = format_snapshot_data(data)
    bounds = config['environment']['multibody']['bounds'] #simulation domain size
    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields, #concentration fields (glucose)
        n_snapshots=4, #show 4 timepoints
        out_dir=out_dir,
        filename=f"mother_machine_simulation")

    #make video
        #data = simulation output from simulate_experiment
        #plot_type renders the diffusion fields
        #step = 100 --> includes every 100th timestep as a video frame
    make_video(
        data,
        bounds,
        plot_type='fields',
        step=100,
        out_dir=out_dir,
        filename=f"mother_machine_simulation",
    )

#script entry point (to have it run as a program)
    #if __name__ = __main__ ---> this runs the block when the file is executed directly
    #creates out_dir if it doesn't exist
    #run simulation for 7000 units and plot into out_dir

if __name__ == '__main__':
    out_dir = os.path.join('out', 'experiments', 'mother_machine_simulation')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_mother_machine(7000, out_dir)


