'''
visualize.py -- visualization functions for localizations
and trajectories

'''
# Numerical stuff
import numpy as np 

# Reading / writing trajectories
from scipy import io as sio 

# Dataframes
import pandas as pd 

# I/O
import os
import sys
from . import spazio 

# Plotting
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style = 'ticks')

def wrapup(out_png, dpi = 400, open_result = True):
    ''' Save a plot to PNG '''
    plt.tight_layout()
    plt.savefig(out_png, dpi = dpi)
    plt.close()
    if open_result:
        os.system('open %s' % out_png)

def loc_density(
    locs,
    ax = None,
    upsampling_factor = 20,
    verbose = False,
): 
    kernel_size = 2 * int(upsampling_factor * 0.2) + 1
    middle = kernel_size//2
    radius = kernel_size/3
    r2 = radius**2

    kernel_y, kernel_x = np.mgrid[:kernel_size, :kernel_size]
    kernel = np.exp(-((kernel_x-middle)**2 + (kernel_y-middle)**2) / (2*r2))

    n_locs = sum([trajs[traj_idx][0].shape[0] for traj_idx in range(n_trajs)])

    for loc_idx in range(n_locs):
        y = int(round(positions[loc_idx, 0] * upsampling_factor, 0))
        x = int(round(positions[loc_idx, 1] * upsampling_factor, 0))
        try:
            density[
                y-middle : y+middle+1,
                x-middle : x+middle+1,
            ] += kernel
        except ValueError:
            continue 

        if verbose:
            sys.stdout.write('Finished compiling the densities of %d/%d localizations...\r' % (loc_idx+1, n_locs))
            sys.stdout.flush()
    if ax == None:
        fig, ax = plt.subplots(figsize = (4, 4))
        ax.imshow(
            density[::-1,:],
            cmap='gray',
            vmax=density.mean() + density.std() * vmax_mod,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        wrapup()
    else:
        ax.imshow(
            density[::-1,:],
            cmap='gray',
            vmax=density.mean() + density.std() * vmax_mod,
        )
    
def loc_density_from_trajs(
    trajs,
    ax = None,
    upsampling_factor = 20,
    verbose = False,
):
    locs = spazio.extract_positions_from_trajs(trajs)
    loc_density(locs, ax=ax, upsampling_factor=upsampling_factor,
        verbose=verbose)

def plot_trajectories(
    trajs,
    ax = None,
    cmap = 'viridis',
    cap = 3000,
    verbose = True,
    n_colors = 100,
    color_index = None,
):
    n_trajs = trajs.shape[0]
    if n_trajs > cap:
        trajs = trajs[:cap, :]
        n_trajs = trajs.shape[0]
    else:
        cap = n_trajs 

    if color_index == None:
        color_index = (np.arange(n_trajs) * 33).astype('uint16')

    colors = sns.color_palette(cmap, n_colors)

    if ax == None:
        fig, ax = plt.subplots(figsize = (4, 4))
        finish_plot = True
    else:
        finish_plot = False 

    for traj_idx in range(cap):
        traj = trajs[traj_idx]
        ax.plot(
            traj[0][:,1] * upsampling_factor - 0.5,
            traj[0][:,0] * upsampling_factor - 0.5,
            marker = '.',
            markersize = 2,
            linestyle = '-',
            linewidth = 0.5,
            color = colors[traj_idx % len(colors)],
        )
        if verbose:
            sys.stdout.write('Finished plotting %d/%d trajectories...\r' % (traj_idx + 1, traj_cap))
            sys.stdout.flush()
    ax.set_aspect('equal')
    return ax











