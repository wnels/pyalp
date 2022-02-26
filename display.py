import matplotlib.pyplot as plt
import numpy as np

def display_norm(matrix):
    matrix -= matrix.min()
    matrix /= matrix.max()
    matrix *= 255
    matrix = matrix.astype(np.uint8)
    plt.figure()
    plt.imshow(matrix)
    plt.show()