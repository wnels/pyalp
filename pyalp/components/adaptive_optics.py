import numpy as np

#==============================================================================
#==============================================================================
class spatial_light_modulator:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid, element_count, element_length):
        self.grid = grid
        self.element_count = element_count
        self.element_length = element_length
        self.reset()

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam):
        beam.x_field = beam.x_field * np.exp(1j * self.get_phase_grid())
        return beam

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def update_phase(self, phase_perturbations):
        self.phases += phase_perturbations

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def reset(self):
        self.phases = np.zeros((self.element_count, self.element_count))

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_phase_grid(self, phases=None):
        if phases is None:
            phases = self.phases

        expand = np.round(self.element_length / self.grid.x_delta)
        phase_grid = np.repeat(phases, expand, axis=0)
        phase_grid = np.repeat(phase_grid, expand, axis=1)
        pad = self.grid.count - phase_grid.shape[0]
        pad0 = int(np.floor(pad / 2.0))
        pad1 = int(np.ceil(pad / 2.0))
        padding = ((pad0, pad1), (pad0, pad1))
        phase_grid = np.pad(phase_grid, padding, 'constant')
        return phase_grid

#==============================================================================
#==============================================================================
class spatial_light_modulator_spgd(spatial_light_modulator):
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(
        self,
        grid,
        element_count,
        element_length,
        perturbation_mag,
        learning_rate):

        super().__init__(grid, element_count, element_length)
        self.perturbation_mag = perturbation_mag
        self.learning_rate = learning_rate
        self.perturbations = np.zeros((self.element_count, self.element_count))

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate_plus(self, beam):
        beam.x_field = beam.x_field * np.exp(1j * self.get_phase_grid_plus())

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate_minus(self, beam):
        beam.x_field = beam.x_field * np.exp(1j * self.get_phase_grid_minus())

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def new_perturbation(self):
        size = (self.element_count, self.element_count)
        self.perturbations = np.random.randint(-1, 2, size).astype(np.float64)
        self.perturbations *= self.perturbation_mag

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_phase_grid_plus(self):
        return self.get_phase_grid(self.phases + self.perturbations)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_phase_grid_minus(self):
        return self.get_phase_grid(self.phases - self.perturbations)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def update_phase(self, plus, minus):
        self.phases += \
            self.learning_rate * \
            self.perturbations * \
            (plus - minus) / \
            self.perturbation_mag**2
