import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import yaml

from pyalp.beams import guassian
from pyalp.components import atmosphere, lens, phase_screen, reflector, spatial_filter
from pyalp.diagnostics import display
from pyalp.domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def reciprocity_experiment(config_path, instances, save_interval=10):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    imaging_lens = lens.thin_lens(**config['lens'])
    beam = guassian.gaussian(grid, **config['beam'])

    gauss_filter = spatial_filter.gaussian(
        grid,
        config['beam']['spot_size'],
        config['beam']['radius'],
        config['beam']['focus'],
        beam.get_wavenumber())

    target_values = np.zeros(instances, dtype=np.complex_)
    detector_values = np.zeros(instances, dtype=np.complex_)

    for index in tqdm.tqdm(range(instances)):

        turb = phase_screen.kolmogorov(grid, **config['turbulence']['kolmogorov'])
        channel = atmosphere.channel(turb, **config['turbulence']['atmosphere'])
        target = reflector.rough(grid)

        beam = guassian.gaussian(grid, **config['beam'])
        channel.forward(beam)
        target_values[index] = np.sum(np.square(beam.x_field) * target.get_phasor())
        target.propagate(beam)
        channel.backward(beam)
        gauss_filter.propagate(beam)
        imaging_lens.focus(beam)
        detector_values[index] = beam.get_on_axis_field()

    return target_values, detector_values

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
    target_values, detector_values = reciprocity_experiment(
        args.config_path,
        args.instances)

    assert np.all(np.isclose(target_values, detector_values, rtol=1e-2)), \
        'Optical setup is not reciprocal'
