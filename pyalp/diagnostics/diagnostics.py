import numpy as np

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def power_in_bucket(beam, radius):
    bucket = (beam.grid.r_matrix < radius)
    pib = \
        np.sum(beam.get_intensity() * bucket) / \
        np.sum(beam.get_intensity())
    return pib

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
def get_on_axis_field(beam):
    return beam.x_field[beam.grid.count // 2, beam.grid.count // 2]

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
def get_on_axis_intensity(beam):
    return np.abs(get_on_axis_field())**2
