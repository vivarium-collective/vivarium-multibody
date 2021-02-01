from vivarium.core.process import Process


class Exchange(Process):
    """ Exchange
    A minimal exchange process that moves molecules between two ports
    """
    defaults = {
        'molecules': [],
        'uptake_rate': {},
        'secrete_rate': {},
        'default_uptake_rate': 1e-3,
        'default_secrete_rate': 1e-5,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.uptake_rate = self.parameters['uptake_rate']
        self.secrete_rate = self.parameters['secrete_rate']

        for mol_id in self.parameters['molecules']:
            if mol_id not in self.uptake_rate:
                self.uptake_rate[mol_id] = self.parameters['default_uptake_rate']
            if mol_id not in self.secrete_rate:
                self.secrete_rate[mol_id] = self.parameters['default_secrete_rate']

    def ports_schema(self):
        return {
            'external': {
                mol_id: {
                    '_default': 0.0,
                    '_emit': True}
                for mol_id in self.parameters['molecules']},
            'internal': {
                mol_id: {
                    '_default': 0.0,
                    '_emit': True}
                for mol_id in self.parameters['molecules']}}

    def next_update(self, timestep, states):
        external_molecules = states['external']
        internal_molecules = states['internal']

        delta_in = {
            mol_id: mol_ex * self.uptake_rate[mol_id] -
                    internal_molecules[mol_id] * self.secrete_rate[mol_id] * timestep
            for mol_id, mol_ex in external_molecules.items()}



        return {
            'internal': delta_in,
            'external': {
                mol_id: -delta
                for mol_id, delta in delta_in.items()}}
