import numpy as np

from pyalp.beams import beams
from pyalp.domain import grids

#==============================================================================
#==============================================================================
class Gaussian:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(
      self,
      grid: grids.Grid2D,
      spot_size: float,
      radius: float,
      focus: float=np.inf,
      wavenumber: float=0):
        self.spot_size = spot_size
        self.grid = grid
        self.focus = focus

        self.filter = np.exp(
            -np.square(self.grid.r_matrix) / np.square(self.spot_size)
            -0.5 * 1j * wavenumber * np.square(self.grid.r_matrix) / self.focus)

        self.filter = self.filter * (grid.r_matrix < radius)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam: beams.Gaussian):
        beam.x_field = self.filter * beam.x_field

#==============================================================================
#==============================================================================
class Tophat:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid: grids.Grid2D, radius: float):

        self.filter = grid.r_matrix < radius

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam: grids.Grid2D):
        beam.x_field = self.filter * beam.x_field
