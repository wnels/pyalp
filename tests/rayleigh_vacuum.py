import argparse
import numpy as np
import yaml

from pyalp.beams import guassian
from pyalp.diagnostics import display
from pyalp.domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def rayleigh_vacuum_experiment(config_path, write_plot=False):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    beam = guassian.gaussian(grid, **config['beam'])

    rayleigh_length = 0.5 * beam.get_wavenumber() * beam.spot_size**2

    intensity0 = beam.get_intensity()
    beam.propagate(rayleigh_length)
    intensity1 = beam.get_intensity()

    if write_plot:
        display.plot1d(
            [intensity0, intensity1],
            grid.x_vector,
            file='rayleigh_vacuum_test.png',
            legend=['transmitted beam', 'rayleigh distance'])

    return intensity0, intensity1

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

    desc = \
        'Test vacuum propagation by esnuring that the on-axis intensity ' \
        'falls to 1/2 the original on-axis intensity at the rayleigh distance'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        'config_path',
        metavar='config_path',
        type=str,
        help='path to config file')

    parser.add_argument(
        '--plot',
        action='store_true',
        help='write plot to file for debugging')

    args = parser.parse_args()

    intensity0, intensity1 = rayleigh_vacuum_experiment(
        args.config_path,
        args.plot)

    intensity_ratio = intensity1.max() / intensity0.max()

    assert np.isclose(intensity_ratio, 0.5, atol=0.005), \
        'On-axis intensity at rayleigh distance is not 1/2 transmitted' \
        ' intensity. There is an error in beam propagation.'
