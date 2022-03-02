import argparse
import yaml

from atmosphere import turbulence, engine
from beams import beams
from diagnostics import display
from domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def single_pass_experiment(config_path):

    with open(args.config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    beam = beams.laser_beam(grid, **config['beam'])
    turb = turbulence.kolmogorov(grid, **config['turbulence']['kolmogorov'])
    channel = engine.atm_channel(turb, **config['turbulence']['atmosphere'])

    channel.forward(beam, progress_bar=True)
    intensity = beam.get_intensity()
    display.display_norm(intensity)
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

    args = parser.parse_args()
    single_pass_experiment(args.config_path)
