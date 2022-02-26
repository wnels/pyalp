import numpy as np

#==============================================================================
#==============================================================================
class grid_2d:
    def __init__(self, x_delta, count):
        self.count = count

        self.x_vector = np.linspace(
            -(count * x_delta) / 2,
            (count * x_delta) / 2,
            count)

        self.x_matrix, self.y_matrix = np.meshgrid(
            self.x_vector,
            self.x_vector)

        self.r_matrix = np.sqrt(
            np.square(self.x_matrix) +
            np.square(self.y_matrix))

        self.k_delta = 2 * np.pi / (count * x_delta)

        self.k_vector = np.linspace(
            -count * self.k_delta / 2,
            count * self.k_delta / 2,
            count)

        self.kx_matrix, self.ky_matrix = np.meshgrid(
            self.k_vector,
            self.k_vector)

        self.k_matrix = np.sqrt(
            np.square(self.kx_matrix) +
            np.square(self.ky_matrix))