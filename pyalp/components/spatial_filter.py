import numpy as np

#==============================================================================
#==============================================================================
class gaussian:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid, spot_size, radius, focus=np.inf, wavenumber=0):
        self.spot_size = spot_size
        self.grid = grid
        self.focus = focus

        self.filter = np.exp(
            -np.square(self.grid.r_matrix) / np.square(self.spot_size)
            -0.5 * 1j * wavenumber * np.square(self.grid.r_matrix) / self.focus)

        self.filter = self.filter * (grid.r_matrix < radius)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam):
        beam.x_field = self.filter * beam.x_field

#==============================================================================
#==============================================================================
class tophat:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, grid, radius):

        self.filter = grid.r_matrix < radius

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def propagate(self, beam):
        beam.x_field = self.filter * beam.x_field
