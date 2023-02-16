import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

#==============================================================================
#==============================================================================
class Gaussian:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid, spot_size, wavelength, radius, focus=np.inf):
        self.grid = grid
        self.wavelength = wavelength
        self.spot_size = spot_size
        self.focus = focus

        self.x_field = np.exp(
            -np.square(self.grid.r_matrix) / np.square(self.spot_size)
            -0.5 * 1j * self.get_wavenumber() *
            np.square(self.grid.r_matrix) / self.focus)

        self.x_field = self.x_field * (self.grid.r_matrix < radius)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, distance):
        fresnel_phase = np.exp(
            -0.5j *
            np.square(self.grid.k_matrix) *
            distance /
            self.get_wavenumber())

        k_field = fftshift(fft2(self.x_field))
        k_field *= fresnel_phase
        self.x_field = ifft2(ifftshift(k_field))

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def distort(self, amplitude=1, phase=0):
        distortion = amplitude * np.exp(1j * self.get_wavenumber() * phase)
        self.x_field = self.x_field * distortion

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def phase_conjugate(self):
        magnitude = np.abs(self.x_field)
        phase = np.angle(self.x_field)
        self.x_field = magnitude * np.exp(-1j * phase)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_intensity(self):
        return np.square(np.abs(self.x_field))

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_wavenumber(self):
        return 2.0 * np.pi / self.wavelength

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_on_axis_intensity(self):
        return np.abs(self.get_on_axis_field())**2

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def get_on_axis_field(self):
        return self.x_field[self.grid.count // 2, self.grid.count // 2]