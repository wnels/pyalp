import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import yaml

from beams import beams
from components import atmosphere, lens, phase_screen, spatial_filter, adaptive_optics
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
    turb = phase_screen.kolmogorov(grid, **config['turbulence']['kolmogorov'])
    channel = atmosphere.channel(turb, **config['turbulence']['atmosphere'])
    slm = adaptive_optics.spatial_light_modulator(
        grid,
        **config['spatial_light_modulator'])

    beam = beams.gaussian(grid, **config['beam'])
    channel.forward(beam)
    np.save('data/original.npy', beam.get_intensity())
    display.plot2d(beam.get_intensity(), grid.x_vector, file='data/original.png')

    receiver_values = []
    receiver_plus_values = []
    receiver_minus_values = []

    for index in tqdm.tqdm(range(instances)):
        slm.new_perturbation()

        beam = beams.gaussian(grid, **config['beam'])
        beam = slm.propagate_plus(beam)
        channel.forward(beam)
        channel.backward(beam)
        beam = slm.propagate_plus(beam)
        beam.distort(gauss_filter.filter)
        beam = imaging_lens.focus(beam)
        receiver_plus = np.sqrt(beam.get_on_axis_intensity())

        beam = beams.gaussian(grid, **config['beam'])
        beam = slm.propagate_minus(beam)
        channel.forward(beam)
        channel.backward(beam)
        beam = slm.propagate_minus(beam)
        beam.distort(gauss_filter.filter)
        beam = imaging_lens.focus(beam)
        receiver_minus = np.sqrt(beam.get_on_axis_intensity())

        beam = beams.gaussian(grid, **config['beam'])
        slm.update_phase(receiver_plus, receiver_minus)
        beam = slm.propagate(beam)
        channel.forward(beam)
        np.save(f'data/corrected_{index}.npy', beam.get_intensity())
        display.plot2d(beam.get_intensity(), grid.x_vector, file='data/corrected.png')
        channel.backward(beam)
        beam = slm.propagate(beam)
        beam.distort(gauss_filter.filter)
        beam = imaging_lens.focus(beam)
        np.save(f'data/reciever_{index}.npy', beam.get_intensity())
        receiver = np.sqrt(beam.get_on_axis_intensity())

        receiver_plus_values.append(receiver_plus)
        receiver_minus_values.append(receiver_minus)
        receiver_values.append(receiver)
        values = np.array([receiver_values, receiver_plus_values, receiver_minus_values])
        np.save(f'data/J_{index}.npy', values)


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
