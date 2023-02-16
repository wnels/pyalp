from scipy.fft import fft2, fftshift

from pyalp.beams import beams
from pyalp.domain import grids

#==============================================================================
#==============================================================================
class ThinLens:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, focal_length: float):
        self.focal_length = focal_length

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def focus(self, beam: beams.Gaussian):
        x_field = fftshift(fft2(beam.x_field))
        x_delta = beam.grid.k_delta * self.focal_length / beam.get_wavenumber()
        grid = grids.Grid2D(x_delta, beam.grid.count)
        beam.x_field = x_field
        beam.grid = grid
