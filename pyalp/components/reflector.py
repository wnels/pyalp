import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from pyalp.components import spatial_filter

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_reflector(grid, reflector_type, radius=None):
    if reflector_type.lower() == "mirror":
        return mirror(grid, radius)
    elif reflector_type.lower() == "rough":
        return rough(grid)
    elif reflector_type.lower() == "cornercube":
        return cornercube(grid, radius)
    else:
        raise Exception(f"reflector type {reflector_type} not supported")

#==============================================================================
#==============================================================================
class mirror:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid, radius):
        self.aperture = spatial_filter.tophat(grid, radius)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam):
        self.aperture.propagate(beam)

#==============================================================================
#==============================================================================
class rough:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid):
        self.phase = np.random.uniform(0, 2 * np.pi, (grid.count, grid.count))
        self.k_filter = grid.r_matrix < (grid.count / 8 * grid.x_delta)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam):
        beam.x_field = beam.x_field * self.get_phasor()

        beam.x_field =\
            ifft2(ifftshift(fftshift(fft2(beam.x_field)) * self.k_filter))

    def get_phasor(self):
        return np.exp(1j * self.phase)

#==============================================================================
#==============================================================================
class cornercube:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid, radius):
        self.aperture = spatial_filter.tophat(grid, radius)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam):
        self.aperture.propagate(beam)
        beam.x_field = np.rot90(beam.x_field, 2)
