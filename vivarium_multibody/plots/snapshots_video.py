import os
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt

from vivarium.core.composition import TEST_OUT_DIR

from vivarium_multibody.composites.lattice import test_lattice
from vivarium_multibody.plots.snapshots import (
    make_snapshots_figure,
    make_tags_figure,
    format_snapshot_data,
    get_field_range,
    get_agent_colors,
    get_tag_ranges
)



def make_snapshot_function(
        data,
        bounds,
        **kwargs):
    multibody_agents, multibody_fields = format_snapshot_data(data)

    # make the snapshot plot function
    time_vec = list(multibody_agents.keys())

    # get fields and agent colors
    multibody_field_range = get_field_range(multibody_fields, time_vec)
    multibody_agent_colors = get_agent_colors(multibody_agents)

    def plot_single_snapshot(t_index):
        time_indices = np.array([t_index])
        snapshot_time = [time_vec[t_index]]
        fig = make_snapshots_figure(
            time_indices=time_indices,
            snapshot_times=snapshot_time,
            agents=multibody_agents,
            agent_colors=multibody_agent_colors,
            fields=multibody_fields,
            field_range=multibody_field_range,
            n_snapshots=1,
            bounds=bounds,
            default_font_size=12,
            plot_width=7,
            show_timeline=False,
            scale_bar_length=0,
            **kwargs)
        return fig

    return plot_single_snapshot, time_vec


def make_tags_function(
        data,
        bounds,
        tagged_molecules=None,
        tag_colors=None,
        convert_to_concs=False,
        **kwargs
):
    tag_colors = tag_colors or {}
    multibody_agents, multibody_fields = format_snapshot_data(data)

    # make the snapshot plot function
    time_vec = list(multibody_agents.keys())
    time_indices = np.array(range(1, len(time_vec)))

    # get agent colors, and ranges
    tag_ranges, tag_colors = get_tag_ranges(
        agents=multibody_agents,
        tagged_molecules=tagged_molecules,
        time_indices=time_indices,
        convert_to_concs=convert_to_concs,
        tag_colors=tag_colors)

    # make the function for a single snapshot
    def plot_single_tags(t_index):
        time_index = np.array([t_index])
        snapshot_time = [time_vec[t_index]]
        fig = make_tags_figure(
            time_indices=time_index,
            snapshot_times=snapshot_time,
            agents=multibody_agents,
            tagged_molecules=tagged_molecules,
            convert_to_concs=convert_to_concs,
            tag_ranges=tag_ranges,
            tag_colors=tag_colors,
            n_snapshots=1,
            bounds=bounds,
            default_font_size=12,
            plot_width=7,
            show_timeline=False,
            scale_bar_length=0,
            show_colorbar=False,
            **kwargs)
        return fig

    return plot_single_tags, time_vec


def make_video(
        data,
        bounds,
        type='fields',
        step=1,
        out_dir='out',
        filename='snapshot_vid',
        **kwargs
):

    # make images directory, remove if existing
    out_file = os.path.join(out_dir, f'{filename}.mp4')
    images_dir = os.path.join(out_dir, f'_images_{type}')
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)

    # get the single snapshots function
    if type == 'fields':
        snapshot_fun, time_vec = make_snapshot_function(
            data,
            bounds,
            **kwargs)
    elif type == 'tags':
        snapshot_fun, time_vec = make_tags_function(
            data,
            bounds,
            **kwargs)

    # make the individual snapshot figures
    img_paths = []
    for t_index in range(0, len(time_vec) - 1, step):
        fig = snapshot_fun(t_index)
        fig_path = os.path.join(images_dir, f"img{t_index}.jpg")
        img_paths.append(fig_path)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()

    # make the video
    img_array = []
    for img_file in img_paths:
        img = cv2.imread(img_file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    # delete image folder
    shutil.rmtree(images_dir)


# def make_interactive(data, bounds):
#     plot_single_snapshot = make_snapshot_function(data, bounds)
#
#     interactive_plot = interactive(
#         plot_single_snapshot,
#         t_index=widgets.IntSlider(min=0, max=time_index_range, step=2, value=0))


def main(total_time=2000, step=60):
    out_dir = os.path.join(TEST_OUT_DIR, 'snapshots_video')
    os.makedirs(out_dir, exist_ok=True)

    # GrowDivide agents
    bounds = [25, 25]
    n_bins = [20, 20]
    initial_field = np.zeros((n_bins[0], n_bins[1]))
    initial_field[:, -1] = 100
    data = test_lattice(
        n_agents=3,
        total_time=total_time,
        growth_noise=1e-3,
        bounds=bounds,
        n_bins=n_bins,
        initial_field=initial_field)

    # make snapshot video
    make_video(
        data,
        bounds,
        type='fields',
        step=step,
        out_dir=out_dir,
        filename=f"snapshots")

    # make tags video
    tagged_molecules = [('boundary', 'mass',)]
    make_video(
        data,
        bounds,
        type='tags',
        step=step,
        out_dir=out_dir,
        filename=f"tags",
        tagged_molecules=tagged_molecules,
        background_color='white')




if __name__ == '__main__':
    main(6000)

