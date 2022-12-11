import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import yaml

from pyalp.beams import beams
from pyalp.components import atmosphere, lens, phase_screen, reflector, spatial_filter, adaptive_optics
from pyalp.diagnostics import display
from pyalp.domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def save(target_intensity, slm, metric, index):
    np.save(f'data/target_{index}.npy', target_intensity)
    np.save(f'data/slm_{index}.npy', slm)
    np.save(f'data/J_{index}.npy', metric)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def reciprocity_experiment(config_path, instances, save_interval):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.grid_2d(**config['grid'])
    imaging_lens = lens.thin_lens(**config['lens'])
    turb = phase_screen.kolmogorov(grid, **config['turbulence']['kolmogorov'])
    channel = atmosphere.channel(turb, **config['turbulence']['atmosphere'])
    target = reflector.get_reflector(grid, **config['reflector'])
    beam = beams.gaussian(grid, **config['beam'])

    gauss_filter = spatial_filter.gaussian(
        grid,
        config['beam']['spot_size'],
        config['beam']['radius'],
        config['beam']['focus'],
        beam.get_wavenumber())

    slm = adaptive_optics.spatial_light_modulator_spgd(
        grid,
        **config['spatial_light_modulator'])

    detector_values = []
    detector_plus_values = []
    detector_minus_values = []

    channel.forward(beam)

    for index in tqdm.tqdm(range(instances)):
        slm.new_perturbation()

        beam = beams.gaussian(grid, **config['beam'])
        slm.propagate_plus(beam)
        channel.forward(beam)
        target.propagate(beam)
        channel.backward(beam)
        slm.propagate_plus(beam)
        gauss_filter.propagate(beam)
        imaging_lens.focus(beam)
        detector_plus = np.sqrt(beam.get_on_axis_intensity())

        beam = beams.gaussian(grid, **config['beam'])
        slm.propagate_minus(beam)
        channel.forward(beam)
        target.propagate(beam)
        channel.backward(beam)
        slm.propagate_minus(beam)
        gauss_filter.propagate(beam)
        imaging_lens.focus(beam)
        detector_minus = np.sqrt(beam.get_on_axis_intensity())

        beam = beams.gaussian(grid, **config['beam'])
        slm.update_phase(detector_plus, detector_minus)
        slm.propagate(beam)
        channel.forward(beam)
        target_intensity = beam.get_intensity()
        target.propagate(beam)
        channel.backward(beam)
        slm.propagate(beam)
        gauss_filter.propagate(beam)
        imaging_lens.focus(beam)
        detector_value = np.sqrt(beam.get_on_axis_intensity())

        detector_plus_values.append(detector_plus)
        detector_minus_values.append(detector_minus)
        detector_values.append(detector_value)

        metrics = np.array([
            detector_values,
            detector_plus_values,
            detector_minus_values])

        if save_interval > 0 and index % save_interval == 0:
            save(
                target_intensity,
                slm.get_phase_grid(),
                metrics,
                index)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

    desc = \
        'focus a beam through atmospheric turbulence using a spatial light ' \
        'modulator (SLM) and stochastic parallel gradient descent (SPGD)'

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

    parser.add_argument(
        '--save-interval',
        type=int,
        default=0,
        help='iteration interval to save results')

    args = parser.parse_args()

    reciprocity_experiment(
        args.config_path,
        args.instances,
        args.save_interval)
