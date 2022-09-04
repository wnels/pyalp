import argparse
import numpy as np
import yaml

from pyalp.beams import gaussian
from pyalp.diagnostics import display
from pyalp.domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def strehl_experiment(config_path):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    beam = guassian.gaussian(grid, **config['beam'])

    intensity0 = beam.get_intensity()
    beam.propagate(config['beam']['focus'])
    intensity1 = beam.get_intensity()

    display.plot1d(
        [intensity0, intensity1],
        grid.x_vector,
        file='strehl.png',
        legend=['transmitted', 'target'])

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
    intensity0, intensity1 = strehl_experiment(args.config_path)
    intensity_ratio = intensity1.max() / intensity0.max()
