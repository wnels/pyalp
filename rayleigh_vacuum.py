import argparse
import numpy as np
import yaml

from beams import beams
from diagnostics import display
from domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def rayleigh_vacuum_experiment(config_path):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    beam = beams.laser_beam(grid, **config['beam'])

    rayleigh_length = 0.5 * beam.get_wavenumber() * beam.spot_size**2

    intensity0 = beam.get_intensity()
    beam.propagate(rayleigh_length)
    intensity1 = beam.get_intensity()

    display.plot1d(
        [intensity0, intensity1],
        grid.x_vector,
        legend=['transmitter', 'rayleigh distance'])

    return intensity0, intensity1

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='single pass atmospheric channel simulation')

    parser.add_argument(
        'config_path',
        metavar='config_path',
        type=str,
        help='path to config file')

    args = parser.parse_args()
    intensity0, intensity1 = rayleigh_vacuum_experiment(args.config_path)
    intensity_ratio = intensity1.max() / intensity0.max()

    assert np.isclose(intensity_ratio, 0.5, atol=0.005), \
        'On-axis intensity at rayleigh distance is not 1/2 transmitted' \
        ' intensity. There is an error in beam propagation.'
