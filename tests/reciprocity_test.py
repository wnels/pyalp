import argparse
import numpy as np
import tqdm
import yaml

from pyalp.beams import beams
from pyalp.components import atmosphere, lens, phase_screen, spatial_filter
from pyalp.domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def reciprocity_experiment(config_path, instances, save_interval=10):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    imaging_lens = lens.thin_lens(**config['lens'])
    beam = beams.gaussian(grid, **config['beam'])

    gauss_filter = spatial_filter.gaussian(
        grid,
        config['beam']['spot_size'],
        config['beam']['radius'],
        config['beam']['focus'],
        beam.get_wavenumber())

    target_values = np.zeros(instances, dtype=np.complex_)
    detector_values = np.zeros(instances, dtype=np.complex_)

    for index in tqdm.tqdm(range(instances)):

        turb = phase_screen.kolmogorov(
            grid,
            **config['turbulence']['kolmogorov'])

        channel = atmosphere.channel(
            turb,
            **config['turbulence']['atmosphere'])

        beam = beams.gaussian(grid, **config['beam'])

        channel.forward(beam)
        target_values[index] = np.sum(np.square(beam.x_field))
        channel.backward(beam)
        gauss_filter.propagate(beam)
        imaging_lens.focus(beam)

        detector_values[index] = beam.get_on_axis_field()

    return target_values, detector_values

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

    desc = \
        'test if the integral of target field squared  multipled by the ' \
        'interaction with target equals the on-axis field at the detector'

    parser = argparse.ArgumentParser(description=desc)

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

    assert np.all(np.isclose(target_values, detector_values)), \
        'Optical setup is not reciprocal'
