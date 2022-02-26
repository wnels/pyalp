import numpy as np

import beams
import display
import grids
import turbulence

spotsize = 0.05
wavelength = 1e-6
grid = grids.grid_2d(x_delta=5e-4, count=2048)
beam = beams.laser_beam(grid, spotsize, wavelength)
intensity1 = beam.get_intensity()
rayleigh_length = np.pi * np.square(spotsize) / wavelength
beam.propagate(rayleigh_length)
intensity2 = beam.get_intensity()

print(intensity2.max())

turb = turbulence.kolmogorov_turbulence(cn2=1e-14, grid=grid)
phase_screen = turb.get_phase_screen(100)
display.display_norm(phase_screen)

display.display_norm(intensity1)
display.display_norm(intensity2)
display.display_norm(intensity1 - intensity2)
print('DONE')


