import numpy as np

#==============================================================================
#==============================================================================
class Grid2D:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, x_delta: float, count: int):
        self.count = count
        self.x_delta = x_delta
        self.k_delta = 2 * np.pi / (self.count * self.x_delta)

        x_min = -self.count * self.x_delta / 2
        x_max = (self.count * self.x_delta) / 2

        k_min = -self.count * self.k_delta / 2
        k_max = self.count * self.k_delta / 2

        if self.count % 2 == 0:
            x_max -= self.x_delta
            k_max -= self.k_delta

        self.x_vector = np.linspace(x_min, x_max, self.count)
        self.k_vector = np.linspace(k_min, k_max, self.count)

        self.x_matrix, self.y_matrix = np.meshgrid(
            self.x_vector,
            self.x_vector)

        self.kx_matrix, self.ky_matrix = np.meshgrid(
            self.k_vector,
            self.k_vector)

        self.r_matrix = np.sqrt(
            np.square(self.x_matrix) +
            np.square(self.y_matrix))

        self.k_matrix = np.sqrt(
            np.square(self.kx_matrix) +
            np.square(self.ky_matrix))