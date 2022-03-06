import numpy as np
from scipy.fft import fft2, fftshift

import sys
sys.path.append("..")
from domain import grids

#==============================================================================
#==============================================================================
class thin_lens:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, focal_length):
        self.focal_length = focal_length

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def focus(self, beam):
        x_field = fftshift(fft2(beam.x_field))
        x_delta = beam.grid.k_delta * self.focal_length / beam.get_wavenumber()
        grid = grids.grid_2d(x_delta, beam.grid.count)
        beam.x_field = x_field
        beam.grid = grid
        return beam
