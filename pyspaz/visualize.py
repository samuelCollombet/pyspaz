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
    metadata = {},
    ax = None,
    upsampling_factor = 20,
    kernel_width = 0.5,
    verbose = False,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    convert_to_um = True,
    vmax_mod = 1.0,
    cmap = 'gray',
    out_png = 'default_loc_density_out.png',
): 
    # Get the list of localized positions
    m_keys = list(metadata.keys())
    positions = np.asarray(locs[[y_col, x_col]])
    if convert_to_um and ('pixel_size_um' in m_keys):
        positions = positions * metadata['pixel_size_um']

    # Make the size of the out frame
    if ('N' in m_keys) and ('M' in m_keys):
        if convert_to_um:
            n_up = int(metadata['N'] * metadata['pixel_size_um']) * upsampling_factor
            m_up = int(metadata['M'] * metadata['pixel_size_um']) * upsampling_factor
        else:
            n_up = int(metadata['N']) * upsampling_factor
            m_up = int(metadata['M']) * upsampling_factor
    else:
        if convert_to_um:
            n_up = int(positions[:,0].max() * metadata['pixel_size_um']) * upsampling_factor
            m_up = int(positions[:,1].max() * metadata['pixel_size_um']) * upsampling_factor
        else:
            n_up = int(positions[:,0].max()) * upsampling_factor
            m_up = int(positions[:,1].max()) * upsampling_factor 

    density = np.zeros((n_up, m_up), dtype = 'float64')

    # Determine the size of the Gaussian kernel to use for
    # KDE
    sigma = kernel_width * upsampling_factor
    w = int(6 * sigma)
    if w % 2 == 0: w+=1
    half_w = w // 2
    r2 = sigma ** 2
    kernel_y, kernel_x = np.mgrid[:w, :w]
    kernel = np.exp(-((kernel_x-half_w)**2 + (kernel_y-half_w)**2) / (2*r2))

    n_locs = len(locs)
    for loc_idx in range(n_locs):
        y = int(round(positions[loc_idx, 0] * upsampling_factor, 0))
        x = int(round(positions[loc_idx, 1] * upsampling_factor, 0))
        try:
            # Localization is entirely inside the borders
            density[
                y-half_w : y+half_w+1,
                x-half_w : x+half_w+1,
            ] += kernel
        except ValueError:
            # Localization is close to the edge
            k_y, k_x = np.mgrid[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
            in_y, in_x = ((k_y>=0) & (k_x>=0) & (k_y<n_up) & (k_x<m_up)).nonzero()
            density[k_y.flatten()[in_y], k_x.flatten()[in_x]] = \
                density[k_y.flatten()[in_y], k_x.flatten()[in_x]] + kernel[in_y, in_x]

        if verbose:
            sys.stdout.write('Finished compiling the densities of %d/%d localizations...\r' % (loc_idx+1, n_locs))
            sys.stdout.flush()
    if ax == None:
        fig, ax = plt.subplots(figsize = (4, 4))
        ax.imshow(
            density[::-1,:],
            cmap=cmap,
            vmax=density.mean() + density.std() * vmax_mod,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        wrapup(out_png)
    else:
        ax.imshow(
            density[::-1,:],
            cmap=cmap,
            vmax=density.mean() + density.std() * vmax_mod,
        )
    return density 
    
def loc_density_from_trajs(
    trajs,
    ax = None,
    upsampling_factor = 20,
    verbose = False,
):
    locs = spazio.extract_positions_from_trajs(trajs)
    loc_density(locs, ax=ax, upsampling_factor=upsampling_factor,
        verbose=verbose)

def show_trajectories(
    trajs,
    ax = None,
    cmap = 'viridis',
    cap = 3000,
    verbose = True,
    n_colors = 100,
    color_by = None,
    upsampling_factor = 1,
):
    n_trajs = trajs.shape[0]
    if n_trajs > cap:
        trajs = trajs[:cap, :]
        n_trajs = trajs.shape[0]
    else:
        cap = n_trajs 

    if color_by == None:
        color_index = (np.arange(n_trajs) * 33).astype('uint16') % n_colors
    else:
        color_values = np.array([i[0] for i in trajs[:, color_by]])
        color_bins = np.arange(n_colors) * color_values.max() / (n_colors-1)
        color_index = np.digitize(color_values, bins = color_bins)

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
            color = colors[color_index[traj_idx]],
        )
        if verbose:
            sys.stdout.write('Finished plotting %d/%d trajectories...\r' % (traj_idx + 1, cap))
            sys.stdout.flush()
    ax.set_aspect('equal')


#
# Functions that operate directly on files
#
def plot_tracked_mat(
    tracked_mat_file,
    out_png,
    cmap = 'viridis',
    cap = 3000,
    verbose = True,
    n_colors = 100,
    color_index = None,
):
    spazio.check_file_exists(tracked_mat_file)

    trajs, metadata, traj_cols = spazio.load_trajs(tracked_mat_file)
    kwargs = {'cmap' : cmap, 'cap' : cap, 'verbose' : verbose,
        'n_colors' : n_colors, 'color_index' : color_index}
    ax = plot_trajectories(trajs, ax=None, **kwargs)
    wrapup(out_png)

def loc_density_from_file(
    loc_file,
    out_png,
    upsampling_factor = 20,
    kernel_width = 0.5,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    convert_to_um = True,
    vmax_mod = 1.0,
    verbose = False,
): 
    spazio.check_file_exists(loc_file)
    if 'Tracked.mat' in loc_file:
        trajs, metadata, traj_cols = spazio.load_trajs(loc_file)

    locs, metadata = spazio.load_locs(loc_file)
    density = loc_density(
        locs,
        metadata = metadata,
        ax = None,
        upsampling_factor = upsampling_factor,
        kernel_width = kernel_width,
        y_col = y_col, 
        x_col = x_col,
        convert_to_um = convert_to_um,
        vmax_mod = vmax_mod,
        verbose = verbose,
        out_png = out_png,
    )




