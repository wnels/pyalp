import numpy as np

from pyalp.beams import beams

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def power_in_bucket(beam: beams.Gaussian, radius: float):
    bucket = (beam.grid.r_matrix < radius)
    pib = \
        np.sum(beam.get_intensity() * bucket) / \
        np.sum(beam.get_intensity())
    return pib

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
def get_on_axis_field(beam: beams.Gaussian):
    return beam.x_field[beam.grid.count // 2, beam.grid.count // 2]

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
def get_on_axis_intensity(beam: beams.Gaussian):
    return np.abs(get_on_axis_field())**2
