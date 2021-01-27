import os

from vivarium.library.units import units
from vivarium.core.process import Composite
from vivarium.core.composition import (
    compartment_in_experiment,
    COMPOSITE_OUT_DIR,
)
from vivarium.plots.agents_multigen import plot_agents_multigen

# processes
from vivarium.processes.growth_rate import GrowthRate
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.meta_division import MetaDivision


NAME = 'grow_divide'



class GrowDivide(Composite):

    defaults = {
        'growth': {
            'variables': ['mass'],
        },
        'divide_condition': {
            'threshold': 2000.0},
        'boundary_path': ('boundary',),
        'agents_path': ('..', '..', 'agents',),
        'daughter_path': tuple(),
        '_schema': {
            'growth': {'variables': {'mass': {'_divider': 'split'}}}
        }
    }

    def __init__(self, config):
        super(GrowDivide, self).__init__(config)

    def generate_processes(self, config):
        # division config
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']
        division_config = dict(
            config.get('division', {}),
            daughter_path=daughter_path,
            agent_id=agent_id,
            generator=self)

        return {
            'growth': GrowthRate(config['growth']),
            'divide_condition': DivideCondition(config['divide_condition']),
            'division': MetaDivision(division_config)}

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        agents_path = config['agents_path']
        return {
            'growth': {
                'variables': ('internal',),
                'rates': ('rates',),
            },
            'divide_condition': {
                'variable': ('internal', 'mass',),
                'divide': boundary_path + ('divide',)
            },
            'division': {
                'global': boundary_path,
                'agents': agents_path},
            }




def test_grow_divide(total_time = 2000):

    agent_id = '0'
    composite = GrowDivide({
        'agent_id': agent_id,
        'growth': {
            'growth_rate': 0.006,  # very fast growth
            'default_growth_noise': 1e-3,
        }
    })

    initial_state = {
        'agents': {
            agent_id: {
                'internal': {
                    'mass': 1000}
            }}}

    settings = {
        'initial_state': initial_state,
        'outer_path': ('agents', agent_id),
        'experiment_id': 'division'}
    experiment = compartment_in_experiment(
        composite,
        settings=settings)

    experiment.update(total_time)
    output = experiment.emitter.get_data()

    # assert
    # external starts at 1, goes down until death, and then back up
    # internal does the inverse
    assert list(output[0.0]['agents'].keys()) == [agent_id]
    assert agent_id not in list(output[total_time]['agents'].keys())
    assert len(output[0.0]['agents']) == 1
    assert len(output[total_time]['agents']) > 1

    return output


def main():
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output = test_grow_divide(8000)

    plot_settings = {}
    plot_agents_multigen(output, plot_settings, out_dir)


if __name__ == '__main__':
    main()
