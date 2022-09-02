import argparse
import numpy as np
import tqdm
import yaml

from beams import beams
from components import atmosphere, lens, phase_screen
from diagnostics import display
from domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def double_pass_experiment(config_path, instances):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    imaging_lens = lens.thin_lens(**config['lens'])

    avg_intensity = np.zeros_like(grid.x_matrix)
    for index in tqdm.tqdm(range(instances)):
        beam = beams.laser_beam(grid, **config['beam'])
        turb = phase_screen.kolmogorov(grid, **config['turbulence']['kolmogorov'])
        channel = atmosphere.channel(turb, **config['turbulence']['atmosphere'])

        channel.forward(beam)
        channel.backward(beam)
        beam = imaging_lens.focus(beam)

        intensity = beam.get_intensity()
        avg_intensity += intensity

    display.plot2d(avg_intensity, grid.x_vector)
    display.plot1d(avg_intensity, grid.x_vector)

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

    parser.add_argument(
        '--instances',
        type=int,
        required=True,
        help='number of independent turbulent channels to average')

    args = parser.parse_args()
    double_pass_experiment(args.config_path, args.instances)
