import numpy as np
from scipy.fft import fft2, fftshift

class kolmogorov_turbulence:
    def __init__(self, cn2, grid):
        self.cn2 = cn2
        self.grid = grid
        self.spectrum = 0.033 * self.cn2 * np.power(
            self.grid.k_matrix,
            -11.0/3.0)

    def get_phase_screen(self, distance):
        noise = \
            np.random.standard_normal((self.grid.count, self.grid.count)) + \
            1j * np.random.standard_normal((self.grid.count, self.grid.count))

        phase = \
            self.grid.k_delta * \
            np.sqrt(2 * np.pi * distance) * \
            fft2(fftshift(noise * np.sqrt(self.spectrum)))

        return np.real(phase)