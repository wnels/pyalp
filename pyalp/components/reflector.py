from abc import ABC, abstractmethod

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from pyalp.beams import beams
from pyalp.components import spatial_filter
from pyalp.domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_reflector(grid: grids.Grid2D, reflector_type: str, radius: bool=None):
    if reflector_type.lower() == "mirror":
        return Mirror(grid, radius)
    elif reflector_type.lower() == "rough":
        return Rough(grid)
    elif reflector_type.lower() == "cornercube":
        return Cornercube(grid, radius)
    else:
        raise Exception(f"reflector type {reflector_type} not supported")

#==============================================================================
#==============================================================================
class Reflector(ABC):
    @abstractmethod
    def propagate(self, beam: beams.Gaussian):
        pass

#==============================================================================
#==============================================================================
class Mirror:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid: grids.Grid2D, radius: float):
        self.aperture = spatial_filter.Tophat(grid, radius)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam: beams.Gaussian):
        self.aperture.propagate(beam)

#==============================================================================
#==============================================================================
class Rough:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid: grids.Grid2D):
        self.phase = np.random.uniform(0, 2 * np.pi, (grid.count, grid.count))
        self.k_filter = grid.r_matrix < (grid.count / 8 * grid.x_delta)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam: beams.Gaussian):
        beam.x_field = beam.x_field * self.get_phasor()

        beam.x_field =\
            ifft2(ifftshift(fftshift(fft2(beam.x_field)) * self.k_filter))

    def get_phasor(self):
        return np.exp(1j * self.phase)

#==============================================================================
#==============================================================================
class Cornercube:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid: grids.Grid2D, radius: float):
        self.aperture = spatial_filter.Tophat(grid, radius)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam: beams.Gaussian):
        self.aperture.propagate(beam)
        beam.x_field = np.rot90(beam.x_field, 2)
