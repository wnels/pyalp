import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import PIL.Image
import re
import tqdm

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_x_extent(x_vector, x_limit=0.15):
    idx_min = np.sum(x_vector < -x_limit)
    idx_max = np.sum(x_vector < x_limit)
    x_extent = [-x_limit, x_limit, -x_limit, x_limit]
    return idx_min, idx_max, x_extent

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def extract_number(f):
    s = re.findall("(\d+).npy", f)
    return int(s[0]) if s else -1

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_last_file_idx(dir):
    files = os.listdir(dir)
    max_file = max(files, key=extract_number)
    last_idx = extract_number(max_file)
    return last_idx

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def make_frames(dir):
    x_vector = np.load(os.path.join(dir, "x_vector.npy"))
    idx_min, idx_max, x_extent = get_x_extent(x_vector)

    last_idx = get_last_file_idx(dir)

    last_metric = np.load(os.path.join(dir, f'J_{last_idx}.npy'))
    last_intensity = np.load(os.path.join(dir, f'target_{last_idx}.npy'))
    last_slm = np.load(os.path.join(dir, f'slm_{last_idx}.npy'))

    metric_normalization = np.max(last_metric[0])
    intensity_normalization = np.max(last_intensity)
    phase_min = np.min(last_slm)
    phase_max = np.max(last_slm - phase_min)

    metric = np.squeeze(last_metric[0]) / metric_normalization

    file_idxs = sorted({extract_number(f) for f in os.listdir(dir)})
    file_idxs.remove(-1)

    for count, file_idx in enumerate(file_idxs):

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(4.8, 14))

        ax0.plot(np.arange(metric.shape[0]), metric)
        ax0.scatter([file_idx], [metric[file_idx]])
        ax0.set_xlabel('iteration')
        ax0.set_ylabel('J (normalized)')
        ax0.set_title('Stochastic Parallel Gradient Descent')
        ax0.set_box_aspect(1)
        ax0.grid()

        slm = np.load(f'data/slm_{file_idx}.npy')
        slm = (slm[idx_min:idx_max, idx_min:idx_max] - phase_min) / phase_max
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes('right', size='5%', pad=0.05)
        im1 = ax1.imshow(slm, extent=x_extent, cmap='cool', interpolation='none', vmin=0, vmax=1)
        ax1.set_xlabel('x [meters]')
        ax1.set_ylabel('y [meters]')
        ax1.set_title('Spatial Light Modulator')
        cax = fig.colorbar(im1, cax=cax1, orientation='vertical')
        cax.set_label('phase [degrees]')

        target = np.load(f'data/target_{file_idx}.npy')
        target = target[idx_min:idx_max, idx_min:idx_max] / intensity_normalization
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes('right', size='5%', pad=0.05)
        im2 = ax2.imshow(target, extent=x_extent, vmin=0, vmax=1)
        ax2.set_xlabel('x [meters]')
        ax2.set_ylabel('y [meters]')
        ax2.set_title('Target Plane')
        cax = fig.colorbar(im2, cax=cax2, orientation='vertical')
        cax.set_label('intensity (normalized)')

        plt.tight_layout(pad=2)
        plt.savefig(f'video/frame{count:05}.png')
        plt.close()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def make_gif(dir):

    gif = []
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        gif.append(PIL.Image.open(file_path))

    gif[0].save(
        'temp_result.gif',
        save_all=True,
        optimize=False,
        append_images=gif[1:],
        duration=100,
        loop=0)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

    desc = 'make a GIF of a reciprocity experiment'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        'dir',
        metavar='dir',
        type=str,
        help='directory of data')

    args = parser.parse_args()

    make_frames(args.dir)
    make_gif('video')

