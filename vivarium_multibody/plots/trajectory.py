import itertools
import os

import numpy as np
from matplotlib import pyplot as plt, lines as mlines
from matplotlib.collections import LineCollection

from vivarium_multibody.plots.snapshots import DEFAULT_BOUNDS, PI


def get_agent_trajectories(agents, times):
    trajectories = {}
    for agent_id, series in agents.items():
        time_indices = series['boundary']['location']['time_index']
        series_times = [times[time_index] for time_index in time_indices]

        positions = series['boundary']['location']['value']
        angles = series['boundary']['angle']['value']
        series_values = [[x, y, theta] for ((x, y), theta) in zip(positions, angles)]

        trajectories[agent_id] = {
            'time': series_times,
            'value': series_values,
        }
    return trajectories


def get_agent_type_colors(agent_ids):
    """ get colors for each agent id by agent type
    Assumes that agents of the same type share the beginning
    of their name, followed by '_x' with x as a single number
    TODO -- make this more general for more digits and other comparisons"""
    agent_type_colors = {}
    agent_types = {}
    for agent1, agent2 in itertools.combinations(agent_ids, 2):
        if agent1[0:-2] == agent2[0:-2]:
            agent_type = agent1[0:-2]
            if agent_type not in agent_type_colors:
                color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(agent_type_colors)]
                agent_type_colors[agent_type] = color
            else:
                color = agent_type_colors[agent_type]
            agent_types[agent1] = agent_type
            agent_types[agent2] = agent_type
    for agent in agent_ids:
        if agent not in agent_types:
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(agent_type_colors)]
            agent_type_colors[agent] = color
            agent_types[agent] = agent

    return agent_types, agent_type_colors


def plot_agent_trajectory(agent_timeseries, config, out_dir='out', filename='trajectory'):

    # trajectory plot settings
    legend_fontsize = 18
    markersize = 25

    bounds = config.get('bounds', DEFAULT_BOUNDS)
    field = config.get('field')
    rotate_90 = config.get('rotate_90', False)

    # get agents
    times = np.array(agent_timeseries['time'])
    agents = agent_timeseries['agents']
    agent_types, agent_type_colors = get_agent_type_colors(list(agents.keys()))

    if rotate_90:
        field = rotate_field_90(field)
        for agent_id, series in agents.items():
            agents[agent_id] = rotate_agent_series_90(series, bounds)
        bounds = rotate_bounds_90(bounds)

    # get each agent's trajectory
    trajectories = get_agent_trajectories(agents, times)

    # initialize a spatial figure
    fig, ax = initialize_spatial_figure(bounds, legend_fontsize)

    # move x axis to top
    ax.tick_params(labelbottom=False,labeltop=True,bottom=False,top=True)
    ax.xaxis.set_label_coords(0.5, 1.12)

    if field is not None:
        field = np.transpose(field)
        shape = field.shape
        im = plt.imshow(field,
                        origin='lower',
                        extent=[0, shape[1], 0, shape[0]],
                        cmap='Greys')
        # colorbar for field concentrations
        cbar = plt.colorbar(im, pad=0.02, aspect=50, shrink=0.7)
        cbar.set_label('concentration', rotation=270, labelpad=20)

    for agent_id, trajectory_data in trajectories.items():
        agent_trajectory = trajectory_data['value']

        # convert trajectory to 2D array
        locations_array = np.array(agent_trajectory)
        x_coord = locations_array[:, 0]
        y_coord = locations_array[:, 1]

        # get agent type and color
        agent_type = agent_types[agent_id]
        agent_color = agent_type_colors[agent_type]

        # plot line
        ax.plot(x_coord, y_coord, linewidth=2, color=agent_color, label=agent_type)
        ax.plot(x_coord[0], y_coord[0],
                 color=(0.0, 0.8, 0.0), marker='.', markersize=markersize)  # starting point
        ax.plot(x_coord[-1], y_coord[-1],
                 color='r', marker='.', markersize=markersize)  # ending point

    # create legend for agent types
    agent_labels = [
        mlines.Line2D([], [], color=agent_color, linewidth=2, label=agent_type)
        for agent_type, agent_color in agent_type_colors.items()]
    agent_legend = plt.legend(
        title='agent type', handles=agent_labels, loc='upper center',
        bbox_to_anchor=(0.3, 0.0), ncol=2, prop={'size': legend_fontsize})
    ax.add_artist(agent_legend)

    # create a legend for start/end markers
    start = mlines.Line2D([], [],
            color=(0.0, 0.8, 0.0), marker='.', markersize=markersize, linestyle='None', label='start')
    end = mlines.Line2D([], [],
            color='r', marker='.', markersize=markersize, linestyle='None', label='end')
    marker_legend = plt.legend(
        title='trajectory', handles=[start, end], loc='upper center',
        bbox_to_anchor=(0.7, 0.0), ncol=2, prop={'size': legend_fontsize})
    ax.add_artist(marker_legend)

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


def plot_temporal_trajectory(
        agent_timeseries,
        config,
        out_dir='out',
        filename='temporal'):

    bounds = config.get('bounds', DEFAULT_BOUNDS)
    field = config.get('field')
    rotate_90 = config.get('rotate_90', False)

    # get agents
    times = np.array(agent_timeseries['time'])
    agents = agent_timeseries['agents']

    if rotate_90:
        field = rotate_field_90(field)
        for agent_id, series in agents.items():
            agents[agent_id] = rotate_agent_series_90(series, bounds)
        bounds = rotate_bounds_90(bounds)

    # get each agent's trajectory
    trajectories = get_agent_trajectories(agents, times)

    # initialize a spatial figure
    fig, ax = initialize_spatial_figure(bounds)

    if field is not None:
        field = np.transpose(field)
        shape = field.shape
        im = plt.imshow(field,
                        origin='lower',
                        extent=[0, shape[1], 0, shape[0]],
                        cmap='Greys'
                        )

    for agent_id, trajectory_data in trajectories.items():
        agent_trajectory = trajectory_data['value']

        # convert trajectory to 2D array
        locations_array = np.array(agent_trajectory)
        x_coord = locations_array[:, 0]
        y_coord = locations_array[:, 1]

        # make multi-colored trajectory
        points = np.array([x_coord, y_coord]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap('cool'))
        lc.set_array(times)
        lc.set_linewidth(6)

        # plot line
        line = plt.gca().add_collection(lc)

    # color bar
    cbar = plt.colorbar(line, ticks=[times[0], times[-1]], aspect=90, shrink=0.4)
    cbar.set_label('time (s)', rotation=270)

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


def rotate_bounds_90(bounds):
    return [bounds[1], bounds[0]]


def rotate_field_90(field):
    return np.rot90(field, 3)  # rotate 3 times for 270


def rotate_agent_series_90(series, bounds):
    location_series = series['boundary']['location']
    angle_series = series['boundary']['angle']

    if isinstance(location_series, dict):
        # this ran with time_indexed_timeseries_from_data
        series['boundary']['location']['value'] = [[y, bounds[0] - x] for [x, y] in location_series['value']]
        series['boundary']['angle']['value'] = [theta + PI / 2 for theta in angle_series['value']]
    else:
        series['boundary']['location'] = [[y, bounds[0] - x] for [x, y] in location_series]
        series['boundary']['angle'] = [theta + PI / 2 for theta in angle_series]
    return series


def initialize_spatial_figure(bounds, fontsize=18):

    x_length = bounds[0]
    y_length = bounds[1]

    # set up figure
    n_ticks = 4
    plot_buffer = 0.02
    buffer = plot_buffer * min(bounds)
    min_edge = min(x_length, y_length)
    x_scale = x_length/min_edge
    y_scale = y_length/min_edge

    # make the figure
    fig = plt.figure(figsize=(8*x_scale, 8*y_scale))
    plt.rcParams.update({'font.size': fontsize, "font.family": "Times New Roman"})

    plt.xlim((0-buffer, x_length+buffer))
    plt.ylim((0-buffer, y_length+buffer))
    plt.xlabel(u'\u03bcm')
    plt.ylabel(u'\u03bcm')

    # specify the number of ticks for each edge
    [x_bins, y_bins] = [int(n_ticks * edge / min_edge) for edge in [x_length, y_length]]
    plt.locator_params(axis='y', nbins=y_bins)
    plt.locator_params(axis='x', nbins=x_bins)
    ax = plt.gca()

    return fig, ax