import argparse
import numpy as np
import yaml

from pyalp.beams import beams
from pyalp.components import atmosphere, phase_screen
from pyalp.diagnostics import display
from pyalp.domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def double_pass_experiment(config_path, write_plot):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    beam = beams.gaussian(grid, **config['beam'])
    turb = phase_screen.kolmogorov(grid, **config['turbulence']['kolmogorov'])
    channel = atmosphere.channel(turb, **config['turbulence']['atmosphere'])

    intensity0 = beam.get_intensity()

    channel.forward(beam, progress_bar=True)
    beam.phase_conjugate()
    channel.backward(beam, progress_bar=True)

    intensity1 = beam.get_intensity()

    if write_plot:
        display.plot2d(
            intensity1,
            grid.x_vector,
            file='phase_conjugation_test.png')

    return intensity0, intensity1

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

    desc = 'test if phase conjugation at the target results in original beam'

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

    intensity0, intensity1 = double_pass_experiment(
        args.config_path,
        args.plot)

    max_difference = np.abs(intensity1 - intensity0).max()

    assert np.isclose(max_difference, 0.0, atol=0.00001), \
        'Phase conjugated double pass does not match original intensity' \
        ' profile.'

