import numpy as np

#==============================================================================
#==============================================================================
class SpatialLightModulator:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid, element_count, element_length):
        self.grid = grid
        self.element_count = element_count
        self.element_length = element_length
        self.phase = np.zeros(self.get_parameter_count())

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam):
        beam.x_field = beam.x_field * np.exp(1j * self.get_phase_grid())
        return beam

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def update_parameters(self, parameters):
        self.phases = np.reshape(
            parameters,
            (self.element_count, self.element_count))

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_parameter_count(self):
        return self.element_count**2

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
class StochasticParallelGradientDescent:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(
        self,
        parameter_count,
        perturbation_mag,
        learning_rate):

        self.parameters = np.zeros(parameter_count)
        self.perturbations = np.zeros(parameter_count)
        self.perturbation_mag = perturbation_mag
        self.learning_rate = learning_rate

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_positive_perturbation(self):
        return self.parameters + self.perturbations

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_negative_perturbation(self):
        return self.parameters - self.perturbations

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def new_perturbation(self):
        self.perturbations = np.random.randint(-1, 2, self.parameters.size)
        self.perturbations = self.perturbations.astype(np.float64)
        self.perturbations *= self.perturbation_mag

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def update_parameters(self, plus, minus):
        self.parameters += \
            self.learning_rate * \
            self.perturbations * \
            (plus - minus) / \
            self.perturbation_mag**2
