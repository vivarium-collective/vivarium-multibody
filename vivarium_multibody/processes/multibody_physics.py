"""
==========================
Multibody physics process
==========================
"""

import os
import sys
import argparse

import random
import math

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# vivarium imports
from vivarium.library.units import units, Quantity
from vivarium.core.process import Process
from vivarium.core.composition import (
    process_in_experiment,
    simulate_experiment,
    PROCESS_OUT_DIR,
)

# vivarium-cell imports
from vivarium_multibody.processes.derive_globals import volume_from_length
from vivarium_multibody.library.pymunk_multibody import PymunkMultibody
from vivarium_multibody.plots.snapshots import (
    plot_snapshots,
    format_snapshot_data,
)


NAME = 'multibody'

DEFAULT_BOUNDS = [40, 40]
DEFAULT_LENGTH_UNIT = units.um
DEFAULT_MASS_UNIT = units.fg
DEFAULT_VELOCITY_UNIT = units.um / units.s
DEFAULT_VOLUME_UNIT = DEFAULT_LENGTH_UNIT ** 3

# constants
PI = math.pi



def random_body_position(body):
    # pick a random point along the boundary
    width, length = body.dimensions
    if random.randint(0, 1) == 0:
        # force along ends
        if random.randint(0, 1) == 0:
            # force on the left end
            location = (random.uniform(0, width), 0)
        else:
            # force on the right end
            location = (random.uniform(0, width), length)
    else:
        # force along length
        if random.randint(0, 1) == 0:
            # force on the bottom end
            location = (0, random.uniform(0, length))
        else:
            # force on the top end
            location = (width, random.uniform(0, length))
    return location

#this is a divider schema
def daughter_locations(value, state):
    parent_length = state['length']
    parent_angle = state['angle']
    pos_ratios = [-0.25, 0.25]
    daughter_locations = []
    for daughter in range(2):
        dx = parent_length * pos_ratios[daughter] * math.cos(parent_angle)
        dy = parent_length * pos_ratios[daughter] * math.sin(parent_angle)
        location = [value[0] + dx, value[1] + dy]
        daughter_locations.append(location)
    return daughter_locations



class Multibody(Process):
    """Simulates collisions and forces between agent bodies with a multi-body physics engine.

    :term:`Ports`:
    * ``agents``: The store containing all agent sub-compartments. Each agent in
      this store has values for location, angle, length, width, mass, thrust, and torque.

    Arguments:
        initial_parameters(dict): Accepts the following configuration keys:

        * **jitter_force**: force applied to random positions along agent
          bodies to mimic thermal fluctuations. Produces Brownian motion.
        * **agent_shape** (:py:class:`str`): agents can take the shapes
          ``rectangle``, ``segment``, or ``circle``.
        * **bounds** (:py:class:`list`): size of the environment in
          micrometers, with ``[x, y]``.
        * **mother_machine** (:py:class:`bool`): if set to ``True``, mother
          machine barriers are introduced.
        * ***animate*** (:py:class:`bool`): interactive matplotlib option to
          animate multibody. To run with animation turned on set True, and use
          the TKAgg matplotlib backend:

          .. code-block:: console

              $ MPLBACKEND=TKAgg python vivarium/processes/snapshots.py

    Notes:
        * rotational diffusion in liquid medium with viscosity = 1 mPa.s: :math:`Dr = 3.5 \pm0.3 rad^{2}/s`
          (Saragosti, et al. 2012. Modeling E. coli tumbles by rotational diffusion.)
        * translational diffusion in liquid medium with viscosity = 1 mPa.s: :math:`Dt = 100 um^{2}/s`
          (Saragosti, et al. 2012. Modeling E. coli tumbles by rotational diffusion.)
    """

    name = NAME
    defaults = {
        'jitter_force': 1e-4,  # pN
        'agent_shape': 'segment',
        'bounds': DEFAULT_BOUNDS,
        'length_unit': DEFAULT_LENGTH_UNIT,
        'mass_unit': DEFAULT_MASS_UNIT,
        'velocity_unit': DEFAULT_VELOCITY_UNIT,
        'boundary_key': 'boundary',
        'mother_machine': False,
        'animate': False,
        'time_step': 1.0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # multibody parameters
        jitter_force = self.parameters['jitter_force']
        self.agent_shape = self.parameters['agent_shape']
        self.bounds = self.parameters['bounds']
        self.mother_machine = self.parameters['mother_machine']

        # units
        self.length_unit = self.parameters['length_unit']
        self.mass_unit = self.parameters['mass_unit']
        self.velocity_unit = self.parameters['velocity_unit']

        # make the multibody object
        if self.mother_machine:
            assert isinstance(self.mother_machine, dict), \
                'mother_machine must be a dictionary with keys ' \
                'spacer_thickness, channel_height, channel_space'
        multibody_config = {
            'agent_shape': self.agent_shape,
            'jitter_force': jitter_force,
            'bounds': self.bounds,
            'barriers': self.mother_machine,
            'physics_dt': self.parameters['time_step'] / 10,
        }
        self.physics = PymunkMultibody(multibody_config)

        # interactive plot for visualization
        self.animate = self.parameters['animate']
        if self.animate:
            plt.ion()
            self.ax = plt.gca()
            self.ax.set_aspect('equal')


    def ports_schema(self):
        glob_schema = {
            '*': {
                self.parameters['boundary_key']: {
                    'location': {
                        '_emit': True,
                        '_default': [0.5 * bound for bound in self.bounds],
                        '_updater': 'set',
                        '_divider': {
                            'divider': daughter_locations,
                            'topology': {
                                'length': ('..', 'length',),
                                'angle': ('..', 'angle',)}}},
                    'length': {
                        '_emit': True,
                        '_default': 2.0},
                    'width': {
                        '_emit': True,
                        '_default': 1.0},
                    'angle': {
                        '_emit': True,
                        '_default': 0.0,
                        '_updater': 'set'},
                    'mass': {
                        '_emit': True,
                        '_default': 1339 * units.fg},
                    'thrust': {
                        '_default': 0.0,
                        '_updater': 'set'},
                    'torque': {
                        '_default': 0.0,
                        '_updater': 'set'},
                }
            }
        }
        schema = {'agents': glob_schema}

        return schema

    def next_update(self, timestep, states):
        agents = states['agents']

        # animate before update
        if self.animate:
            self.animate_frame(agents)

        # update multibody with new agents
        agents = self.bodies_remove_units(agents)
        self.physics.update_bodies(agents)

        # run simulation
        self.physics.run(timestep)

        # get new agent positions
        agent_positions = self.physics.get_body_positions()
        update = {'agents': agent_positions}

        # for mother machine configurations, remove agents above the channel height
        if self.mother_machine:
            channel_height = self.mother_machine['channel_height']
            delete_agents = []
            for agent_id, position in agent_positions.items():
                location = position['boundary']['location']
                y_loc = location[1]
                if y_loc > channel_height:
                    # cell has moved past the channels
                    delete_agents.append(agent_id)
            if delete_agents:
                update['agents'] = {
                    agent_id: position
                    for agent_id, position in agent_positions.items()
                    if agent_id not in delete_agents}

                update['agents']['_delete'] = [
                    agent_id for agent_id in delete_agents]

        return update

    def bodies_remove_units(self, bodies):
        for bodies_id, specs in bodies.items():
            bodies[bodies_id]['boundary'] = self.boundary_remove_units(specs['boundary'])
        return bodies

    def boundary_remove_units(self, boundary):
        if isinstance(boundary['mass'], Quantity):
            boundary['mass'] = boundary['mass'].to(self.mass_unit).magnitude
        if isinstance(boundary['location'], Quantity):
            boundary['location'] = [loc.to(self.length_unit).magnitude for loc in boundary['location']]
        # if isinstance(boundary['width'], Quantity):
        #     boundary['width'] = boundary['width'].to(self.length_unit).magnitude
        # if isinstance(boundary['length'], Quantity):
        #     boundary['length'] = boundary['length'].to(self.length_unit).magnitude
        return boundary

    ## matplotlib interactive plot
    def animate_frame(self, agents):
        plt.cla()
        for agent_id, data in agents.items():
            # location, orientation, length
            data = data['boundary']
            x_center = data['location'][0]
            y_center = data['location'][1]
            angle = data['angle'] / PI * 180 + 90  # rotate 90 degrees to match field
            length = data['length']
            width = data['width']

            # get bottom left position
            x_offset = (width / 2)
            y_offset = (length / 2)
            theta_rad = math.radians(angle)
            dx = x_offset * math.cos(theta_rad) - y_offset * math.sin(theta_rad)
            dy = x_offset * math.sin(theta_rad) + y_offset * math.cos(theta_rad)

            x = x_center - dx
            y = y_center - dy

            if self.agent_shape == 'rectangle' or self.agent_shape == 'segment':
                # Create a rectangle
                rect = patches.Rectangle((x, y), width, length, angle=angle, linewidth=1, edgecolor='b')
                self.ax.add_patch(rect)

            elif self.agent_shape == 'circle':
                # Create a circle
                circle = patches.Circle((x, y), width, linewidth=1, edgecolor='b')
                self.ax.add_patch(circle)

        plt.xlim([0, self.bounds[0]])
        plt.ylim([0, self.bounds[1]])
        plt.draw()
        plt.pause(0.01)


# configs
def make_random_position(bounds):
    return [
        np.random.uniform(0, bounds[0]),
        np.random.uniform(0, bounds[1])]


def single_agent_config(config):
    # cell dimensions
    width = 1.0
    length = 2.0
    volume = volume_from_length(length, width)
    bounds = config.get('bounds', DEFAULT_BOUNDS)
    location = config.get('location')
    if location:
        location = [loc * bounds[n] for n, loc in enumerate(location)]
    else:
        location = make_random_position(bounds)

    return {
        'boundary': {
            'location': location,
            'angle': np.random.uniform(0, 2 * PI),
            'volume': volume,
            'length': length,
            'width': width,
            'mass': 1339 * units.fg,
            'thrust': 0,
            'torque': 0}}


def agent_body_config(config):
    agent_ids = config['agent_ids']
    agent_config = {
        agent_id: single_agent_config(config)
        for agent_id in agent_ids}
    return {'agents': agent_config}


default_gd_config = {'bounds': DEFAULT_BOUNDS}
default_gd_config.update(agent_body_config({
    'bounds': DEFAULT_BOUNDS,
    'agent_ids': ['1', '2']}))


class InvokeUpdate(object):
    def __init__(self, update):
        self.update = update
    def get(self, timeout=0):
        return self.update

