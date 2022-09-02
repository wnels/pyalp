from cmath import phase
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm

from diagnostics import display

x_vector = np.load('x_vector.npy')
x_limit = 0.15
idx_min = np.sum(x_vector < -x_limit)
idx_max = np.sum(x_vector < x_limit)
x_extent = [-x_limit, x_limit, -x_limit, x_limit]

fx_vector = np.load('fx_vector.npy')
fx_limit = 5e-5
fidx_min = np.sum(fx_vector < -fx_limit)
fidx_max = np.sum(fx_vector < fx_limit)
fx_extent = [-fx_limit, fx_limit, -fx_limit, fx_limit]

count = 1000
spgd = np.load(f'data/J_{count-10}.npy')
j = np.squeeze(spgd[0])
j_plus = np.squeeze(spgd[1])
j_minus = np.squeeze(spgd[2])
indices = np.arange(j.shape[0])

intensity_normalization = np.max(np.load(f'data/target_{count-10}.npy'))
phase_normalization = np.max(np.load(f'data/slm_{count-10}.npy'))
metric_normalization = np.max(j)

j = j / np.max(j)

frame_count = 0
#plt.rcParams.update({'font.size': 12})
for index in tqdm.tqdm(range(0, count, 10)):

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.8))
    ax0.plot(indices, j)
    ax0.scatter([index], [j[index]])
    ax0.set_xlabel('iteration')
    ax0.set_ylabel('J (normalized)')
    ax0.set_title('Stochastic Parallel Gradient Descent')
    ax0.set_box_aspect(1)
    ax0.grid()

    slm = np.load(f'data/slm_{index}.npy')
    slm = slm[idx_min:idx_max, idx_min:idx_max]
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    im1 = ax1.imshow(slm, extent=x_extent, cmap='cool', interpolation='none')
    ax1.set_xlabel('x [meters]')
    ax1.set_ylabel('y [meters]')
    ax1.set_title('Spatial Light Modulator')
    cax = fig.colorbar(im1, cax=cax1, orientation='vertical')
    cax.set_label('phase [degrees]')

    target = np.load(f'data/target_{index}.npy')
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
    plt.savefig(f'video/frame{frame_count:05}.png')
    plt.close()
    frame_count += 1