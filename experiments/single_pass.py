import argparse
import yaml

from pyalp.beams import guassian
from pyalp.components import atmosphere, phase_screen
from pyalp.diagnostics import display
from pyalp.domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def single_pass_experiment(config_path):

    with open(args.config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    beam = guassian.gaussian(grid, **config['beam'])
    turb = phase_screen.kolmogorov(grid, **config['turbulence']['kolmogorov'])
    channel = atmosphere.channel(turb, **config['turbulence']['atmosphere'])

    channel.forward(beam, progress_bar=True)
    intensity = beam.get_intensity()

    display.plot2d(
        intensity,
        grid.x_vector,
        file='single_pass.png')

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
