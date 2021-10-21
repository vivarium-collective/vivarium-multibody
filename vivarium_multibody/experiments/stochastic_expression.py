import os
import numpy as np

from vivarium.core.composer import Composer
from vivarium.core.process import Process, Deriver
from vivarium.library.units import units, remove_units
from vivarium.core.registry import process_registry
from vivarium.core.composition import composite_in_experiment

# add imported division processes
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.growth_rate import GrowthRate

from vivarium_multibody.processes.derive_globals import length_from_volume
from vivarium_multibody.composites.lattice import (
    Lattice, make_lattice_config)
from vivarium_multibody.plots.snapshots import (
    plot_snapshots, plot_tags, make_tags_figure, format_snapshot_data)
from vivarium_multibody.plots.snapshots_video import make_video


TIMESTEP = 10
MW = {'C': 1e8 * units.g / units.mol}


class StochasticTx(Process):
    defaults = {'ktsc': 1e0, 'kdeg': 1e-3}

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.ktsc = self.parameters['ktsc']
        self.kdeg = self.parameters['kdeg']
        self.stoichiometry = np.array([[0, 1], [0, -1]])
        self.time_left = None
        self.event = None
        # initialize the next timestep
        initial_state = self.initial_state()
        self.calculate_timestep(initial_state)

    def initial_state(self, config=None):
        return {
            'DNA': {'G': 1.0},
            'mRNA': {'C': 1.0}}

    def ports_schema(self):
        return {
            'DNA': {
                'G': {
                    '_default': 1.0,
                    '_emit': True}},
            'mRNA': {
                'C': {
                    '_default': 1.0,
                    '_emit': True}}}

    def calculate_timestep(self, states):
        # retrieve the state values
        g = states['DNA']['G']
        c = states['mRNA']['C']
        array_state = np.array([g, c])

        # Calculate propensities
        propensities = [
            self.ktsc * array_state[0], self.kdeg * array_state[1]]
        prop_sum = sum(propensities)

        # The wait time is distributed exponentially
        self.calculated_timestep = np.random.exponential(scale=prop_sum)
        return self.calculated_timestep

    def next_reaction(self, x):
        """get the next reaction and return a new state"""

        propensities = [self.ktsc * x[0], self.kdeg * x[1]]
        prop_sum = sum(propensities)

        # Choose the next reaction
        r_rxn = np.random.uniform()
        i = 0
        for i, _ in enumerate(propensities):
            if r_rxn < propensities[i] / prop_sum:
                # This means propensity i fires
                break
        x += self.stoichiometry[i]
        return x

    def next_update(self, timestep, states):

        if self.time_left is not None:
            if timestep >= self.time_left:
                event = self.event
                self.event = None
                self.time_left = None
                return event

            self.time_left -= timestep
            return {}

        # retrieve the state values, put them in array
        g = states['DNA']['G']
        c = states['mRNA']['C']
        array_state = np.array([g, c])

        # calculate the next reaction
        new_state = self.next_reaction(array_state)

        # get delta mRNA
        c1 = new_state[1]
        d_c = c1 - c
        update = {
            'mRNA': {
                'C': d_c}}

        if self.calculated_timestep > timestep:
            # didn't get all of our time, store the event for later
            self.time_left = self.calculated_timestep - timestep
            self.event = update
            return {}

        # return an update
        return {
            'mRNA': {
                'C': d_c}}


class Tl(Process):
    defaults = {'ktrl': 5e-4, 'kdeg': 5e-5}

    def ports_schema(self):
        return {
            'mRNA': {
                'C': {
                    '_default': 100 * units.mg / units.mL,
                    '_divider': 'split',
                    '_emit': True}},
            'Protein': {
                'X': {
                    '_default': 200 * units.mg / units.mL,
                    '_divider': 'split',
                    '_emit': True}}}

    def next_update(self, timestep, states):
        C = states['mRNA']['C']
        X = states['Protein']['X']
        dX = (self.parameters['ktrl'] * C - self.parameters['kdeg'] * X) * timestep
        return {
            'Protein': {
                'X': dX}}


class LengthFromVolume(Deriver):
    defaults = {'width': 1.}  # um
    def ports_schema(self):
        return {
            'global': {
                'volume': {'_default': 1 * units.fL},
                'length': {'_default': 2., '_updater': 'set'},
            }}
    def next_update(self, timestep, states):
        volume = states['global']['volume']
        length = length_from_volume(volume.magnitude, self.parameters['width'])
        return {
            'global': {
                'length': length}}


class TxTlDivision(Composer):
    defaults = {
        'time_step': TIMESTEP,
        'stochastic_Tx': {},
        'Tl': {},
        'concs': {
            'molecular_weights': {
                'C': 1e8 * units.g / units.mol}},
        'growth': {
            'time_step': 1,
            'default_growth_rate': 0.0005,
            'default_growth_noise': 0.001,
            'variables': ['volume']},
        'agent_id': np.random.randint(0, 100),
        'divide_condition': {
            'threshold': 2.5 * units.fL},
        'agents_path': ('..', '..', 'agents',),
        'boundary_path': ('boundary',),
        'daughter_path': tuple(),
        '_schema': {
            'concs': {
                'input': {'C': {'_divider': 'binomial'}},
                'output': {'C': {'_divider': 'set'}},
            }}}

    def generate_processes(self, config):
        counts_to_concentration = process_registry.access('counts_to_concentration')
        division_config = dict(
            daughter_path=config['daughter_path'],
            agent_id=config['agent_id'],
            composer=self)
        time_step_config = {'time_step': config['time_step']}
        return {
            'stochastic_Tx': StochasticTx({**config['stochastic_Tx'], **time_step_config}),
            'Tl': Tl({**config['Tl'], **time_step_config}),
            'concs': counts_to_concentration(config['concs']),
            'growth': GrowthRate({**config['growth'], **time_step_config}),
            'divide_condition': DivideCondition(config['divide_condition']),
            'shape': LengthFromVolume(),
            'division': MetaDivision(division_config)}

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        agents_path = config['agents_path']
        return {
            'stochastic_Tx': {
                'DNA': ('DNA',),
                'mRNA': ('RNA_counts',)},
            'Tl': {
                'mRNA': ('RNA',),
                'Protein': ('Protein',)},
            'concs': {
                'global': boundary_path,
                'input': ('RNA_counts',),
                'output': ('RNA',)},
            'growth': {
                'variables': boundary_path,
                'rates': ('rates',)},
            'divide_condition': {
                'variable': boundary_path + ('volume',),
                'divide': boundary_path + ('divide',)},
            'shape': {
                'global': boundary_path,
            },
            'division': {
                'global': boundary_path,
                'agents': agents_path}}


def run_sim(
        total_time=1000,
        bounds=[10, 10],
        time_step=10.,
):
    agent_id = '0'
    diffusion_rate = 0.001
    bins = [10, 10]
    depth = 10

    # make the lattice environment
    lattice_config = make_lattice_config(
        n_bins=bins,
        bounds=bounds,
        depth=depth,
        diffusion=diffusion_rate,
        time_step=time_step,
        jitter_force=1e-5,
    )
    lattice_composer = Lattice(lattice_config)
    lattice_composite = lattice_composer.generate()

    # make a txtl composite, embedded under an agents store
    txtl_composer = TxTlDivision({'time_step': time_step})
    agent = txtl_composer.generate({'agent_id': agent_id})
    lattice_composite.merge(composite=agent, path=('agents', agent_id))

    initial_state = {
        'agents': {
            agent_id: {
                'global': {'volume': 1.2 * units.fL},
                'DNA': {'G': 1},
                # 'RNA': {'C': 5 * units.mg / units.mL},
                'RNA_counts': {'C': 1},
                'Protein': {'X': 50 * units.mg / units.mL}}}}

    sim_settings = {
        'progress_bar': False,
        'initial_state': initial_state,
    }
    experiment = composite_in_experiment(lattice_composite, sim_settings)
    experiment.update(total_time)
    data = experiment.emitter.get_data_deserialized()
    data = remove_units(data)
    return data


def main():
    total_time = 7000
    bounds = [15, 15]
    data = run_sim(
        total_time=total_time,
        bounds=bounds,
    )

    out_dir = 'out/stochastic_expression'
    os.makedirs(out_dir, exist_ok=True)

    # format the data for plot_snapshots
    agents, fields = format_snapshot_data(data)

    # save snapshots figure
    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=4,
        out_dir=out_dir,
        filename='snapshots'
    )

    # tags plot
    n_snapshots = 5
    time_vec = list(agents.keys())
    time_indices = np.round(np.linspace(0, len(time_vec) - 1, n_snapshots)).astype(int)
    snapshot_times = [time_vec[i] for i in time_indices]
    tagged_molecules = [
        ('RNA', 'C'),
        ('Protein', 'X'),
    ]
    make_tags_figure(
        agents=agents,
        bounds=bounds,
        n_snapshots=n_snapshots,
        time_indices=time_indices,
        snapshot_times=snapshot_times,
        convert_to_concs=False,
        tagged_molecules=tagged_molecules,
        out_dir=out_dir,
        filename='tags'
    )

    # make snapshot video
    tagged_molecules = [
        ('RNA', 'C'),  # use RNA_counts?
        ('Protein', 'X'),
    ]
    make_video(
        data,
        bounds,
        plot_type='tags',
        step=100,  # render every nth snapshot
        tagged_molecules=tagged_molecules,
        out_dir=out_dir,
        filename=f'snapshots_video',
    )

    div_number = 0
    for t in data.keys():
        if len(data[t]['agents'].keys()) > div_number:
            print(f"time {t}: agent_ids {list(data[t]['agents'].keys())}")
            div_number += 1


if __name__ == '__main__':
    main()