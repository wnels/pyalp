from beams import beams
from components import atmosphere, phase_screen
from diagnostics import display
from domain import grids

grid = grids.grid_2d(x_delta=2.5e-4, count=2048)
beam = beams.gaussian(grid, spot_size=0.05, wavelength=1e-6)
turb = phase_screen.kolmogorov(grid, cn2=1e-14)
channel = atmosphere.channel(turb, distance=3000, phase_screen_count=10)
channel.forward(beam, progress_bar=True)
display.plot2d(beam.get_intensity(), grid.x_vector, title='z = 3 km')
display.plot1d(beam.get_intensity(), grid.x_vector, title='z = 3 km, y = 0')