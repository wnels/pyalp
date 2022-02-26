from atmosphere import turbulence, engine
from beams import beams
from diagnostics import display
from domain import grids

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
    grid = grids.grid_2d(x_delta=5e-4, count=2048)
    beam = beams.laser_beam(grid, spot_size=0.05, wavelength=1e-6)
    turb = turbulence.kolmogorov_turbulence(cn2=1e-14, grid=grid)
    channel = engine.atm_channel(turb, distance=1000, phase_screen_count=10)

    channel.forward(beam, progress_bar=True)
    intensity = beam.get_intensity()
    display.display_norm(intensity)
