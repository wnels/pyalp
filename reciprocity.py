import argparse
import numpy as np
import tqdm
import yaml

from beams import beams
from components import atmosphere, lens, phase_screen, spatial_filter
from diagnostics import display
from domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def reciprocity_experiment(config_path, instances):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    imaging_lens = lens.thin_lens(**config['lens'])
    gauss_filter = spatial_filter.gaussian(grid, config['beam']['spot_size'])

    for index in tqdm.tqdm(range(instances)):
        beam = beams.laser_beam(grid, **config['beam'])
        turb = phase_screen.kolmogorov(grid, **config['turbulence']['kolmogorov'])
        channel = atmosphere.channel(turb, **config['turbulence']['atmosphere'])

        channel.forward(beam)
        target = np.abs(np.sum(np.square(beam.x_field)))

        channel.backward(beam)
        beam.distort(gauss_filter.filter)
        beam = imaging_lens.focus(beam)

        on_axis_intensity = beam.get_on_axis_intensity()
        reciever = np.sqrt(on_axis_intensity)
        print('done')



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
    reciprocity_experiment(args.config_path, args.instances)
