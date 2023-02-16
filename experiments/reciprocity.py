import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import yaml

from pyalp.beams import beams
from pyalp.components import atmosphere, lens, phase_screen, reflector, spatial_filter, adaptive_optics
from pyalp.diagnostics import display
from pyalp.domain import grids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def save(
  target_intensity: np.ndarray,
  slm: np.ndarray,
  metric: np.ndarray,
  index: int):
    np.save(f'data/target_{index}.npy', target_intensity)
    np.save(f'data/slm_{index}.npy', slm)
    np.save(f'data/J_{index}.npy', metric)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def double_pass(
  beam: beams.Gaussian,
  slm: adaptive_optics.SpatialLightModulator,
  channel: atmosphere.Channel,
  target: reflector.Reflector,
  gauss_filter: spatial_filter.Gaussian,
  imaging_lens: lens.ThinLens):
    slm.propagate(beam)
    channel.forward(beam)
    target_intensity = beam.get_intensity()
    target.propagate(beam)
    channel.backward(beam)
    slm.propagate(beam)
    gauss_filter.propagate(beam)
    imaging_lens.focus(beam)
    return np.sqrt(beam.get_on_axis_intensity()), target_intensity

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def reciprocity_experiment(
  config_path: str,
  instances: int,
  save_dir: str,
  save_interval: int):

    with open(config_path) as file_stream:
        config = yaml.safe_load(file_stream)

    grid = grids.Grid2D(**config['grid'])
    imaging_lens = lens.ThinLens(**config['lens'])
    turb = phase_screen.Kolmogorov(grid, **config['turbulence']['kolmogorov'])
    channel = atmosphere.Channel(turb, **config['turbulence']['atmosphere'])
    target = reflector.get_reflector(grid, **config['reflector'])
    beam = beams.Gaussian(grid, **config['beam'])

    gauss_filter = spatial_filter.Gaussian(
        grid,
        config['beam']['spot_size'],
        config['beam']['radius'],
        np.inf,
        beam.get_wavenumber())

    slm = adaptive_optics.SpatialLightModulator(
        grid,
        **config['spatial_light_modulator'])

    spgd = adaptive_optics.StochasticParallelGradientDescent(
        parameter_count=slm.get_parameter_count(),
        **config['stochastic_parallel_gradient_descent'])

    detector_values = []
    detector_plus_values = []
    detector_minus_values = []

    channel.forward(beam)

    optical_config = {
        "slm": slm,
        "channel": channel,
        "target": target,
        "gauss_filter": gauss_filter,
        "imaging_lens": imaging_lens}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'x_vector.npy'), grid.x_vector)

    for index in tqdm.tqdm(range(instances)):
        spgd.new_perturbation()

        optical_config['beam'] = beams.Gaussian(grid, **config['beam'])
        slm.update_parameters(spgd.get_positive_perturbation())
        detector_plus, _ = double_pass(**optical_config)

        optical_config['beam'] = beams.Gaussian(grid, **config['beam'])
        slm.update_parameters(spgd.get_negative_perturbation())
        detector_minus, _ = double_pass(**optical_config)

        spgd.update_parameters(detector_plus, detector_minus)
        optical_config['beam'] = beams.Gaussian(grid, **config['beam'])
        slm.update_parameters(spgd.parameters)
        detector_value, target_intensity = double_pass(**optical_config)

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
        '--save-dir',
        type=str,
        default=None,
        help='directory to save results')

    parser.add_argument(
        '--save-interval',
        type=int,
        default=0,
        help='iteration interval to save results')

    args = parser.parse_args()

    reciprocity_experiment(
        args.config_path,
        args.instances,
        args.save_dir,
        args.save_interval)
