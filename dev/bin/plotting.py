"""
Urban Boundary Layer Observation Data Processing
Script name:    Plotting Functions
Path:           ~/bin/plotting.py
Description:    This script contains functions to plot data.
"""

import calendar
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import copy
import datetime
import numpy as np
import os
import logging
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import xarray as xr

from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from metpy.plots import Hodograph as hodo
from scipy.ndimage.filters import gaussian_filter

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.captureWarnings(True)

# Change font
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# Letter strings for subplots
subplot_letters = ['a', 'b', 'c', 'd']
# Full names for sites
site_names = {'BRON': 'The Bronx', 'QUEE': 'Queens', 'STAT': 'Staten Island'}
# Color cycler list
color_cycler = ['r', 'g', 'b']

def noise_filter(data):
    # Smooths noisy data to provide clean contours    
    v = data.values.copy()
    v[np.isnan(data.values)] = 0
    v = gaussian_filter(v, sigma=0.5, truncate=6)
    
    w = 0*data.values.copy() + 1
    w[np.isnan(data.values)] = 0
    w = gaussian_filter(w, sigma=0.5, truncate=6)
    
    return v/w

def colormaps(param, mode):
    '''
    Defines colormaps for given parameters for anomaly datasets.

    Parameters
    ----------
    param : str
        Data variable from xArray Dataset.
    mode : str
        String describing colormap sequence (e.g., sequential, divergent, etc.).

    Returns
    -------
    None.

    '''

    if mode == 'sequential':
        cs = {'temperature': 'Reds',
              'potential_temperature': 'Reds',
              'virtual_potential_temperature': 'Reds',
              'mixing_ratio': 'Greens',
              'relative_humidity': 'Greens',
              'specific_humidity': 'Greens',
              'vapor_density': 'Greens',
              'pressure': 'cividis',
              'U': 'Blues',
              'u': 'Blues',
              'v': 'Blues',
              'w': 'Blues',
              'ri': 'binary'}
        return cs[param]
    elif mode == 'divergent':
        cs = {'temperature': 'RdBu_r',
              'potential_temperature': 'RdBu_r',
              'virtual_potential_temperature': 'RdBu_r',
              'mixing_ratio': 'BrBG',
              'relative_humidity': 'BrBG',
              'specific_humidity': 'BrBG',
              'vapor_density': 'BrBG',
              'pressure': 'RdBu_r',
              'ri': 'RdBu_r',
              'U': 'RdBu_r',
              'u': 'RdBu_r',
              'v': 'RdBu_r',
              'w': 'RdBu_r'}
        return cs[param]


def site_coords(site):
    '''
    Returns coordinates for a given site.

    Parameters
    ----------
    site : str
        Observation site. Can be either 'BRON', 'MANH', 'QUEE', or 'STAT'.

    Returns
    -------
    tuple
        Coordinates of desired site.

    '''
    site_coords = {'BRON': (40.8729, -73.8945), 'MANH': (40.8200, -73.9493),
                   'QUEE': (40.7366, -73.8201), 'STAT': (40.6021, -74.1504)}

    return site_coords[site]

def colorbar_limits(data):
    '''
    Calculate minimum and maxmimum colorbar values.

    Parameters
    ----------
    data : xArray Dataset
        Dataset with site and parameter already selected.

    Returns
    -------
    vmin : float
        Minimum colorbar value.
    vmax : float
        Maximum colorvar value.

    '''

    # Handle diverging cases
    if data.min() < 0 and data.max() > 0:
        # Limits are +/- 2-sigma
        vmin, vmax = [-2*data.std(), 2*data.std()]
    # Handle other cases
    else:
        # Limits are +/- 2-sigma
        vmin, vmax = [data.mean() - 2*data.std(), data.mean() + 2*data.std()]

    return vmin, vmax


def quiver(data, event=None, site='QUEE', mode='hourly', param='w', limits=None):

    try:
        units = data[param].units    
    except:
        units = ''

    # Number of colorbar levels
    N = 13

    if mode == 'hourly':
        data = data.sel(site=site).groupby('time.hour').mean()
        data = data.transpose("height", "hour")
        data = data.coarsen(height=3, boundary='trim').mean()
        
        cmap = colormaps(param, 'sequential')
        
        norm = None
        
        # If anomaly has both positive and negative data, use a divergent colormap. Else, use sequential.
        sigma = 3
        mean, std = [data[param].mean(), data[param].std()]
        if param == 'w':
            levels = np.linspace(-1, 1, N)
        else:
            levels = np.linspace(mean - sigma*std, 
                                 mean + sigma*std, N)
        values = noise_filter(data[param])
        
        if np.nanmin(values) < 0 and np.nanmax(values) > 0:
            # Define colormap
            cmap = colormaps(param, 'divergent')
            norm = mpl.colors.TwoSlopeNorm(vmin=min(levels), 
                                           vcenter=0.,
                                           vmax=max(levels))
        else:
            # Define colormap
            cmap = colormaps(param, 'sequential')
        

        fig, ax = plt.subplots(dpi=300, figsize=(5, 3))
        im = ax.contourf(data.hour.values, data.height.values,
                         values, norm=norm, 
                         cmap=cmap, levels=levels, extend='both')
        vectors = ax.quiver(data.hour.values, data.height.values,
                            data['u'].values, data['v'].values, 
                            scale=300, cmap=cmap, pivot='mid', 
                            width=0.004, zorder=1)

    elif mode == 'anomaly':
        data = data.sel(site=site)
        data = data.transpose("height", "hour")
        data = data.coarsen(height=3, boundary='trim').mean()

        cmap = colormaps(param, 'sequential')
        if not limits:
            vmin, vmax = colorbar_limits(data[param])
            if data[param].min() < 0 and data[param].max() > 0:
                cmap = colormaps(param, 'divergent')
        else:
            vmin, vmax = limits
            if vmin < 0 and vmax > 0:
                cmap = colormaps(param, 'divergent')

        fig, ax = plt.subplots(dpi=300)
        im = ax.pcontourf(data.hour.values, data.height.values,
                           data[param].values, cmap=cmap, norm=norm)
        vectors = ax.quiver(data.hour.values, data.height.values,
                            data['u'].values, data['v'].values, scale=200, pivot='mid', width=0.004, zorder=1)

    else:
        data = data.sel(site=site).transpose("height", "time")
        data = data.coarsen(height=3, boundary='trim').mean()

        cmap = colormaps(param, 'sequential')
        if not limits:
            vmin, vmax = data[param].min(), data[param].max()
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            if data[param].min() < 0 and data[param].max() > 0:
                cmap = colormaps(param, 'divergent')
                norm = mpl.colors.CenteredNorm(vcenter=0)
        else:
            vmin, vmax = limits
            if vmin < 0 and vmax > 0:
                cmap = colormaps(param, 'divergent')

        fig, ax = plt.subplots(dpi=300, figsize=(5, 3))
        # im = ax.pcolormesh(data.time.values, data.height.values,
        #                    data[param].values, alpha=1, cmap=cmap, vmin=vmin, vmax=vmax)
        im = ax.contourf(data.time.values, data.height.values,
                           data[param].values, levels=16, alpha=1, cmap=cmap, norm=norm)
        vectors = ax.quiver(data.time.values, data.height.values,
                            data['u'].values, data['v'].values, scale=300, cmap=cmap, pivot='mid', width=0.002, zorder=1)

        myFmt = mdates.DateFormatter('%m-%d %H')
        ax.xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate()

    
    ''' Colorbar formatting. '''
    # Set up colorbar axis
    cax = make_axes_locatable(ax).append_axes('right', 
                                              size='3%', 
                                              pad=0.1)
    
    # Define colorbar. Custom notation for specific humidity.
    if param == 'specific_humidity':
        colorbar = fig.colorbar(im, cax=cax, cmap=cmap, extend='both', format='%.1e')
    else:
        colorbar = fig.colorbar(im, cax=cax, cmap=cmap, extend='both', format='%.2f')
    
    # Create colorbar label
    colorbar.set_label('$\mathregular{{{1}}}$'.format(param.replace('_', ' ').title(), units), rotation=270, labelpad=20)
    
    ax.set_ylabel('Height [m]', labelpad=10)

    qk = ax.quiverkey(vectors, X=0.84, Y=1.05, U=10,
                      label='{0} m/s'.format(10), 
                      coordinates='axes', zorder=2, labelpos='E')

    if event:
        ax.set_title('{0}, {1}, {2}, {3}'.format(
            site, mode, param, event), loc='left')
    else:
       #  ax.set_title('{0}, {1}, {2}'.format(site, mode, param), loc='left')
       ax.set_title(' '.format(site, mode, param), loc='left')
    fig.tight_layout()
    plt.show()

    return ax


def multi_plot(data, param):
    fig, axs = plt.subplots(dpi=300, nrows=2, ncols=2,
                            sharex=True, sharey=True)
    sites = data.site.values
    for i, ax in enumerate(fig.axes):
        bounds = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
        cmap = copy.copy(mpl.cm.get_cmap("RdBu"))
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='both')
        norm = colors.TwoSlopeNorm(vmin=min(bounds), vcenter=0.25, vmax=max(bounds))

        if 'time' in list(data.dims.keys()):
            im = ax.pcolormesh(data.time.values, data.height.values, data[param].sel(
                site=sites[i]).values.T, norm=norm, cmap=cmap, shading='auto')
        elif 'hour' in list(data.dims.keys()):
            im = ax.pcolormesh(data.hour.values, data.height.values, data[param].sel(
                site=sites[i]).values.T, norm=norm, cmap=cmap, shading='auto')
        over, under = str(colors.to_hex(cmap(1))), str(colors.to_hex(cmap(0)))
        cmap.set_over('#440154')

        ax.set_title('{0}'.format(sites[i]))
    fig.autofmt_xdate(rotation=30, ha='right')
    fig.subplots_adjust(right=0.85)
    #  fig.suptitle(data.time[0])
    cax = fig.add_axes([1.025, 0.17, 0.02, 0.74])
    colorbar = fig.colorbar(im, cax=cax, extend='both')
    fig.tight_layout()


def cum_wind_rose_grouped(data, site, height, hours=[0, 24], 
                          params=['zeta', 'wind_direction'], title=True, savefig=False):
    '''
    Plot frequency of a parameter over all wind directions for a given site, height, and time range.

    Parameters
    ----------
    data : xArray Dataset
        Dataset with typical information.
    site : str
        Site name.
    height : int
        Height of desired observations.
    hours : list or tuple, optional
        List or tuple with 2 elements, with the first and last hour of the range desired. End non-inclusive. The default is [0, 24].

    '''

    # Filter data by the given parameters
    data = data.sel(site=site, 
                    height=height, 
                    time=((data.time.dt.hour >= hours[0]) &
                          (data.time.dt.hour < hours[-1])))
    # Initialize a DataFrame to hold grouped data
    df = pd.DataFrame()
    # Define the secondary groupby labels
    labels = ['Unstable', 'Slightly unstable',
              'Neutral', 'Slightly stable', 'Stable']
    # Define the groupby bins
    n_bins = 25
    bins = {params[0]: [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
            params[1]: np.linspace(0, 360, num=n_bins, endpoint=True)}
    # Group by primary parameter
    data_grouped = data.groupby_bins(params[1], 
                                     bins=bins[params[1]], 
                                     right=False, 
                                     include_lowest=True)
    # Iterate over each primary grouping
    for key in data_grouped.groups.keys():
        # Initialize a dictionary to hold grouped data
        s = {}
        # If there is data in the group, store it. Else, continue.
        try:
            # Perform secondary grouping
            temp = data_grouped[key].groupby_bins(params[0], 
                                                  bins=bins[params[0]], 
                                                  labels=labels, 
                                                  right=False, 
                                                  include_lowest=True)
            # Iterate over every secondary group
            for subkey in temp.groups.keys():
                # Count number of occurrences in this secondary group
                s[subkey] = len(temp[subkey].unstack().time.values)
                # Attach corresponding primary group
                s[params[1]] = key.left
                # Append to DataFrame
                df = df.append(s, ignore_index=True)
        except:
            for subkey in labels:
                # Count number of occurrences in this secondary group
                s[subkey] = 0
                # Attach corresponding primary group
                s[params[1]] = key.left
                # Append to DataFrame
                df = df.append(s, ignore_index=True)
    
    # Reset the index to primary group.
    df = df.set_index(params[1], drop=True)
    # Consolidate all primary groupings to ensure they are aggregated
    df = df.groupby(df.index).sum()
    # Create column with sum of occurrences per primary group.
    df['sum'] = df.sum(axis=1)
    # Set minimum count to establish valid frequency
    df = df.where(df['sum'] > 20)
    # Divide all occurrence columns by the sum to calculate percentage.
    df = df.iloc[:, :-1].div(df['sum'], axis=0)
    df = df.sort_index()

    # Fill nans with 0, since nan values indicating no occurences, which is equivalent to 0%
    df = df.fillna(0)

    df_ = df.copy()
    # Iterate over columns to find missing secondary group values
    for col in labels:
        # If missing, fill with zeroes
        if col not in df_.columns:
            df_[col] = 0
    # Keep only the columns with labels
    df_ = df_[labels]
    # Convert wind directions to radians
    df_.index = df_.index.array * np.pi/180
    
    # Define the inner radius
    bottom = 1
    # Define the theta values on the r-theta plane
    theta = df_.index.array
    # Define the width of each bar based on number of primary group bins
    width = (2*np.pi) / (n_bins-1)
    
    print(df_)
    
    # Initialize figure
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_subplot(111, polar=True)
    # Reorient the figure
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    summation = []
    cmap = mpl.cm.RdBu
    
    # Iterate over every secondary grouping
    for i in range(0, len(df_.columns)):
        # For the first iteration, the base radius will be the predefined inner radius
        if i == 0:
            summation = [bottom] * len(df_.iloc[:, i])
        # Add from there
        else:
            summation = summation + df_.iloc[:, i-1]
        # Plot it
        if i == 2:
            quotient = 0.5
        else:
            quotient = i/len(df_.columns)
        ax.bar(theta, df_.iloc[:, i], 
               width=width, 
               bottom=summation, 
               label=df_.columns[i], 
               color=cmap(quotient), 
               linewidth=1, edgecolor='k')
        
    ax.grid(True, linestyle=':')
    ax.xaxis.grid(False)
    
    theta_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticklabels(theta_ticks)
    
    # Define and format the radial labels
    ax.set_yticks([1, 1.5, 2],)
    bbox = dict(boxstyle="round", ec='none', fc='none')
    plt.setp(ax.get_yticklabels(), bbox=bbox)
    ax.set_yticklabels(['0%', '50%', '100%'], zorder=999)
    ax.tick_params(axis='y', which='major', pad=30)
    # ax.set_rlabel_position(70)
    
    if title:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.325), 
                  ncol=len(bins[params[0]]), frameon=False)
        # ax.set_title('{0} at {1} m from {2}:00 to {3}:00 LST'.format(site,
        #                                                              height, 
        #                                                              hours[0], 
        #                                                              hours[1]),
        #              y=1.35)
        
    if savefig:
        dirname = '/Users/gabriel/Documents/urban_boundary_layer_obs/turbulence_manuscript/figs'
        filename = 'wind_direction-stability-{0}-s{1}_e{2}-{3}m.png'.format(site,
                                                                        hours[0],
                                                                        hours[1],
                                                                        height)
        plt.savefig(os.path.join(dirname, filename), dpi=300, bbox_inches='tight')


def cum_wind_rose(data, site, height, ax=None):
    
    ''' Note to self: try a non-binned scatter wind rose instead of the wind rose. '''

    data = data.sel(site=site, height=height)

    df = pd.DataFrame()
    param = 'U'
    params = [param, 'wind_direction']
    
    n_bins = 25
    bins = {'U': [0, 5, 10, 20, 50, 100],
            'wind_direction': np.linspace(0, 360, num=n_bins, endpoint=True)}
    
    labels = ['{0} to {1}'.format(bins[param][i], bins[param][i+1]) for i in range(0, len(bins[param])-1)]
    
    # Group data into pre-defined bins
    data_grouped = data.groupby_bins(params[1], bins=bins[params[1]], right=False, include_lowest=True)
    for key in data_grouped.groups.keys():
        s = {}
        temp = data_grouped[key].groupby_bins(params[0], bins=bins[params[0]], labels=labels, right=False, include_lowest=True)
        for subkey in temp.groups.keys():
            s[subkey] = len(temp[subkey].unstack().time.values)
        s['wind_direction'] = key.left
        df = df.append(s, ignore_index=True)
    df = df.set_index('wind_direction', drop=True)
    df['sum'] = df.sum(axis=1)
    # df = df.iloc[:, :-1].div(df['sum'], axis=0)
    df = df.sort_index()

    # Fill nans with 0, since nan values indicating no occurences, which is equivalent to 0%
    df = df.fillna(0)

    df_ = df.copy()
    cols = labels
    for col in cols:
        if col not in df_.columns:
            df_[col] = 0
    df_ = df_[cols]
    df_.index = df_.index.array * np.pi/180
    df_ = df_ / sum(df['sum'])
    bottom = 0.02

    theta = df.index.array * np.pi/180
    width = (2*np.pi) / (n_bins-1)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    summation = []
    cmap = mpl.cm.Blues.from_list('Blues_mod', mpl.cm.Blues(np.linspace(0.2, 1, 5)), N=256)
    # Loop through columns in the DataFrame
    for i in range(0, len(df_.columns)):
        # For initial category, start the bar at some position ('bottom') and make it into a list, with one item per radial bar
        if i == 0:
            summation = [bottom]*len(df_.iloc[:, i])
        # Else, use the values from the bars
        else:
            summation = summation + df_.iloc[:, i-1]
        ax.bar(theta, df_.iloc[:, i], width=width, bottom=summation, label=df_.columns[i], color=cmap(
            i/len(df_.columns)), 
               linewidth=1, edgecolor='k', zorder=10)
        
    ax.set_yticklabels([])
    ax.grid(which='both', color='k', linestyle=':', linewidth=1, axis='x', alpha=0.01, visible=True)
    
    # Create percentages
    tick_vals = 100*df.iloc[:, -1]/df_.iloc[:, :-1].to_numpy().sum()
    num, roundto = 5, 0.05
    radial_ticks = np.linspace(bottom, df_.sum(axis=1).max()+bottom, num)
    radial_ticks = np.append(radial_ticks, radial_ticks[-1] + np.diff(radial_ticks)[0])
    radial_ticks = [roundto * round(s/roundto) for s in radial_ticks]
    radial_ticks = np.linspace(0, 0.2, num)
    ax.set_rticks(radial_ticks)
    
    radial_tick_labels = ax.set_yticklabels(['{0:2d}%'.format(int(s)) for s in 100*radial_ticks], fontsize=8)
    radial_tick_labels[0].set_visible(False)
    ax.set_rlabel_position(67.5)
    
    theta_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticklabels(theta_ticks)
    
    return ax

def wind_rose_comparison(data, site='QUEE', heights=[100], hours=[0, 24]):
    
    ''' Provide side-by-side comparison between datasets. '''

    # Initialize list to store datasets
    datas = []
    for dataset in data:    
        # Filter times based on user input (end non-inclusive)
        dataset = dataset.sel(time=((dataset.time.dt.hour >= hours[0]) & (dataset.time.dt.hour < hours[1])))
        datas.append(dataset)
        
    nrows, ncols, szf = len(heights), len(data), 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(szf*ncols, szf*nrows), dpi=300, subplot_kw={'projection': 'polar'})

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')
        
        ax = cum_wind_rose(datas[j], site, heights[k], ax=ax)
        
        data_type = ['Normal', 'Extreme heat event']

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title('{0}'.format(data_type[j]), pad=15)
        if j == 0:
            ax.set_ylabel('{0} m'.format(heights[k]))
            ax.yaxis.set_label_coords(-0.2,0.5)
            
    # Legend formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    names = ['{0}'.format(s.split(' ')[0]) for s in by_label.keys()]
    # Define colorbar
    bounds = [int(name) for name in names]
    norm = mpl.colors.BoundaryNorm(bounds, 256)
    cax = fig.add_axes([0.25, 1.03, 0.5, 0.02])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Blues'),
                 cax=cax, orientation='horizontal', extend='max')
    cax.xaxis.set_ticks_position('top')
    colorbar.set_label('Wind speed [$\mathregular{{{0}}}$]'.format('m \ s^{-1}'),
                       labelpad=-50)
    
    fig.tight_layout()
    

def mean_profiles(data, sites=['QUEE'], data_std=None, param='U', hours=[0, 6, 12, 18], plot_std=False):
    '''
    Generate vertical profiles at chosen times of day (see Melecio-Vazquez et al, 2018, Figure 2).
    '''

    # Generate hourly data if data is not already averaged
    if 'time' in data.dims:
        data_std = data.groupby('time.hour').std()
        data = data.groupby('time.hour').mean()
        
    print(data)

    fig, axs = plt.subplots(dpi=300, ncols=len(hours), figsize=(6, 2), sharey=True)
    vmin, vmax = [data[param].min(), data[param].max()]
    for j, site in enumerate(sites):
        # Hold quantities to be plotted
        quantities, stds = [], []
        for i, ax in enumerate(fig.axes):
            hour = hours[i]
            temp = data[param].sel(site=site, hour=hour).values
            temp_std = data_std[param].sel(site=site, hour=hour).values
            # if np.isnan(vmin) or np.isnan(vmax):
            #     vmin, vmax = np.nanmin(temp), np.nanmax(temp)
            # elif np.nanmin(temp) < vmin:
            #     vmin = np.nanmin(temp) - np.nanmax(temp_std)
            # elif np.nanmax(temp) > vmax:
            #     vmax = np.nanmax(temp) + np.nanmax(temp_std)

            im = ax.plot(temp, data.height.values,
                         label='{0}, {1}'.format(site, hours[i]),
                         lw=2,
                         c=color_cycler[j])
            if plot_std:
                std = ax.fill_betweenx(data.height.values,
                                       temp - temp_std,
                                       temp + temp_std,
                                       alpha=0.2)
                
            if i == 0:
                axs[i].spines.right.set_visible(False)
                axs[i].spines.top.set_visible(False)
                axs[i].set_ylabel('Height [m]')
            else:
                axs[i].tick_params(top=False,
                               bottom=True,
                               left=False,
                               right=False,
                               labelleft=False,
                               labelbottom=True)
                axs[i].spines.left.set_visible(False)
                axs[i].spines.right.set_visible(False)
                axs[i].spines.top.set_visible(False)
                
            axs[i].set_title(hours[i], pad=20)

            axs[i].set_xlim([vmin, vmax])
            if param == 'w':
                axs[i].plot([0, 0], [0, 1000], lw=1, c=(0, 0, 0, 0.5), zorder=0)
            
        ax.set_ylim([min(data.height.values), max(data.height.values)])
        if param == 'w':
            ax.plot([0, 0], [0, 1000], lw=1, c='k')
            ax.set_ylim([0, 1000])
        
    fig.legend(sites, frameon=False)
        
    return fig


def timeseries(data, sites=['BRON', 'QUEE', 'STAT'], height=0, params=['temperature']):
    '''
    Plot timeseries for averaged data at a given height.
    '''

    if len(data) != 2:
        return None

    norm, hw = data
    n_avg, n_std = norm.groupby('time.hour').mean(), norm.groupby('time.hour').std()
    hw_avg, hw_std = hw.groupby('time.hour').mean(), hw.groupby('time.hour').std()

    # Define cyclic properties
    plt_cycler = cycler(color=['b', 'r', 'g', 'm'])

    nrows, ncols, szf = len(params), len(sites), 1.75
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows), nrows=nrows, ncols=ncols, dpi=300, sharex=True)

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')
        
        if (params[k] not in ['temperature', 'specific_humidity', 'relative_humidity']) and (height == 0):
            height += 100

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title(site_names[sites[j]], fontsize=10)
        unit = norm[params[k]].attrs['units']
        unit_str = r'$\mathregular{{%s}}$' %unit
        if j == 1:
            ax.set_xlabel('Hour [LST]', labelpad=10)
            
        # Format y-axis label 
        if params[k] in ['u', 'v', 'w']:
            ax.set_ylabel('{0} [{1}]'.format(params[k].replace('_', ' '), unit_str), labelpad=10)
        else:
            ax.set_ylabel('{0} [{1}]'.format(params[k].replace('_', ' ').title(), unit_str), labelpad=10)
            
        ax.label_outer()
        # Define formatting cycle for axis
        ax.set_prop_cycle(plt_cycler)
        
        # Main data
        im_n = ax.plot(n_avg.hour, n_avg[params[k]].sel(
            site=sites[j], height=height), label='Normal', lw=2, marker='o', markersize=3)
        im_hw = ax.plot(hw_avg.hour, hw_avg[params[k]].sel(
            site=sites[j], height=height), label='Extreme heat event', lw=2, marker='s', markersize=3)

        # Standard deviations
        im_n_std = ax.fill_between(n_avg.hour,
                                   n_avg[params[k]].sel(
                                       site=sites[j], height=height).values - n_std[params[k]].sel(site=sites[j], height=height),
                                   n_avg[params[k]].sel(
                                       site=sites[j], height=height) + n_std[params[k]].sel(site=sites[j], height=height),
                                   alpha=0.2)
        im_hw_std = ax.fill_between(hw_avg.hour,
                                    hw_avg[params[k]].sel(
                                        site=sites[j], height=height) - hw_std[params[k]].sel(site=sites[j], height=height),
                                    hw_avg[params[k]].sel(
                                        site=sites[j], height=height) + hw_std[params[k]].sel(site=sites[j], height=height),
                                    alpha=0.2)
        ax.set_xlim([0, 23])
        # Define x-tick labels
        ax.set_xticks(np.arange(0, 24, 3))
        n = 2  # Keeps every nth label
        [l.set_visible(False) for (i,l) in 
         enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        
        if params[k] == 'specific_humidity':
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))

    # Figure formatting
    # fig.suptitle('Properties at {0} m'.format(height))
    fig.tight_layout()
    # Legend formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.575, 1.05), fontsize=10)
    try:
        fig.align_ylabels(axs[:, 0])
    except:
        pass


def contour_timeseries_ri(data, site='QUEE'):
    
    # Redefine data with given parameters
    data = data['ri'].sel(site=site)
    param = 'ri'
    # If data is a timeseries, keep time. If it's a groupby, get hours.
    if 'time' in data.dims:
        times = data.time.values
    elif 'hour' in data.dims:
        times = data.hour.values
    
    ''' Eliminate outliers. '''
    # Get mean and standard deviation
    mean, std = [data.mean(), data.std()] 
    # Define sigma level for filtering
    sigma = 4
    # Remove outliers based on sigmas from mean
    # data = data.where((data > (mean - sigma*std)) & (data < (mean + sigma*std)))
    
    ''' Plotting. '''
    # Define units
    try:
        units = data.units    
    except:
        units = ''
    # Define colorbar extension direction(s)
    min_val, max_val, extend = None, None, None
    # If anomaly has both positive and negative data, use a divergent colormap. Else, use sequential.
    if np.nanmin(data) < 0 and np.nanmax(data) > 0:
        # Define colormap
        cmap = colormaps(param, 'divergent')
    else:
        # Define colormap
        cmap = colormaps(param, 'sequential')
    
    
    # Get colorbar limits
    levels = 12
    
    norm = None
    # Custom processing for bulk Richardson number
    if param == 'ri':
        cmap = 'RdBu'
        
        # Smooths noisy data to provide clean contours    
        v = data.values.copy()
        v[np.isnan(data.values)] = 0
        v = gaussian_filter(v, sigma=1, truncate=6)
        
        w = 0*data.values.copy() + 1
        w[np.isnan(data.values)] = 0
        w = gaussian_filter(w, sigma=1, truncate=6)
        
        values = v/w
        
        levels = [-1, -0.5, 0, 0.25, 0.5, 1, 2]
        norm = mpl.colors.BoundaryNorm(levels, 256)
    else:
        norm = colors.TwoSlopeNorm(vmin=np.nanmin(data.values), 
                                   vcenter=0, 
                                   vmax=np.nanmax(data.values))
        values = data.values
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    im = ax.contourf(times, data.height.values, values.T, 
                     norm=norm, cmap=cmap, levels=levels, extend='both')
    if param == 'ri':
        ax.contour(times, data.height, values.T, 
               colors='k', levels=[0.25], linewidths=2.5)

    # Subplot formatting
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    ax.set_ylabel('Height [m]', labelpad=10)
    ax.label_outer()
    
    # Custom y-limits for Richardson plotting
    if param == 'ri':
        ax.set_ylim([100, data.height.max()])

    # Place colorbar at the end of each row
    cax = make_axes_locatable(ax).append_axes('right', 
                                                size='3%', 
                                                pad=0.1)
    colorbar = fig.colorbar(im, cax=cax, extend=extend)
    colorbar_label = colorbar.set_label('$\mathregular{{{1}}}$'.format(param.replace('_', ' ').title(), units), rotation=270, labelpad=20)

    # Figure formatting
    fig.tight_layout()
    
def contour_timeseries_anomaly(data, sites=['BRON', 'QUEE', 'STAT'], params=['temperature']):
    '''
    Plot contour timeseries for averaged data.
    '''

    if len(data) != 2:
        return None

    norm, hw = data
    
    n_avg, n_std = norm.groupby(
        'time.hour').mean(), norm.groupby('time.hour').std('time')
    hw_avg, hw_std = hw.groupby(
        'time.hour').mean(), hw.groupby('time.hour').std('time')

    nrows, ncols, szf = len(params), len(sites), 2
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows*0.9),
                            nrows=nrows, ncols=ncols, dpi=300)

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')

        # Main data
        x, y = n_avg.hour, n_avg.height
        anom = hw_avg[params[k]].sel(
            site=sites[j]).T - n_avg[params[k]].sel(site=sites[j]).T
        
        
        anom = xr.apply_ufunc(
            lambda x, m, s: (x-m)/s,
            hw_avg[params[k]].sel(site=sites[j]).T,
            n_avg[params[k]].sel(site=sites[j]).T,
            n_std[params[k]].sel(site=sites[j]).T)
        
        # print(anom.max())
        
        # Curb potential infs or -infs in the data for wind components
        if params[k] in ['u', 'w', 'U']:
            sigma = 2
            anom = anom.where((anom < sigma) & (anom > -sigma))
        
        # Define colorbar extension direction(s)
        min_val, max_val, extend = None, None, None

        # If anomaly has both positive and negative data, use a divergent colormap. Else, use sequential.
        if np.nanmin(anom) < 0 and np.nanmax(anom) > 0:
            # Define colormap
            cmap = colormaps(params[k], 'divergent')
            # Get the minimum and maximum values from the anomaly dataset
            min_val, max_val = [np.nanmin(np.ma.masked_invalid(anom)), 
                                np.nanmax(np.ma.masked_invalid(anom))]
            # Cap the bounds at 2-sigma to ensure normalization of data
            if min_val < -2:
                min_val = -2
                extend = 'both'
            if max_val > 2:
                max_val = 2
                extend = 'both'
            # Define normalization about 0
            norm = colors.TwoSlopeNorm(vmin=min_val, vcenter=0., vmax=max_val)
        else:
            # Define colormap
            cmap = colormaps(params[k], 'sequential')
            # Get the minimum and maximum values from the anomaly dataset
            min_val, max_val = [np.nanmin(np.ma.masked_invalid(anom)), 
                                np.nanmax(np.ma.masked_invalid(anom))]
            # Cap the bounds between 0 and the extreme value
            norm = colors.Normalize(vmin=min_val, vmax=max_val)
            if min_val >= 0 and max_val >= 0:
                extend = 'max'
            else:
                extend = 'min'
            
        # Get colorbar limits at 2-sigma and levels
        levels = 12

        im = ax.contourf(x, y, anom, cmap=cmap, norm=norm, extend=extend, levels=levels)

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title('{0}'.format(site_names[sites[j]], fontsize=10))
        
        if j == 1:
            ax.set_xlabel('Hour [LST]', labelpad=10)
        
        # Set a double y-axis label with parameter of interest and height
        if params[k] in ['u', 'v', 'w']:
            ax.set_ylabel('{0} \n \n Height [m]'.format(params[k].replace('_', ' ')), labelpad=10)
        else:
            ax.set_ylabel('{0} \n \n Height [m]'.format(params[k].replace('_', ' ').title()), labelpad=10)
        ax.label_outer()
        
        # Define x-tick labels
        ax.set_xticks(np.arange(0, 24, 3))
        n = 2  # Keeps every nth label
        [l.set_visible(False) for (i,l) in 
         enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]

        # Add subplot text identifier
        # ann = ax.annotate(xy=(0, 0), xytext=(0.93, 0.86), text=subplot_letters[j], 
        #             xycoords='axes fraction', fontsize=12, ha='right')
        # ann.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # Place colorbar at the end of each row
        if j == ncols-1:
            cax = make_axes_locatable(ax).append_axes('right', 
                                                      size='5%', 
                                                      pad=0.1)
            colorbar = fig.colorbar(im, cax=cax, extend=extend)
            colorbar_label = colorbar.set_label('$\sigma$', rotation=270, labelpad=20)

    # Figure formatting
    # fig.suptitle('Heat Wave Anomalies')
    fig.tight_layout()
    # Legend formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), frameon=False)
    
def contour_timeseries_normal(data, sites=['BRON', 'QUEE', 'STAT'], params=['temperature'], quiver_params=None):
    '''
    Plot contour timeseries for averaged data.
    '''
    
    mean, std = [data.groupby('time.hour').mean(), 
                 data.groupby('time.hour').std('time')]

    nrows, ncols, szf = len(params), len(sites), 2.5
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows),
                            nrows=nrows, ncols=ncols, dpi=300)
    
    min_val, max_val = [np.nan, np.nan]

    for i, ax in enumerate(fig.axes):
        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')
        # Select values
        values = data[params[k]].sel(site=sites[j]).groupby('time.hour').mean().T
        # Main data
        x, y = mean.hour, mean.height
        # Curb potential infs or -infs in the data for wind components
        if params[k] in ['u', 'w', 'U']:
            sigma = 2
            # values = values.where((std < sigma) & (std > -sigma))
        # Define colorbar extension direction(s)
        min_val, max_val, extend = None, None, None

        # If anomaly has both positive and negative data, use a divergent colormap. Else, use sequential.
        if np.nanmin(values) < 0 and np.nanmax(values) > 0:
            # Define colormap
            cmap = colormaps(params[k], 'divergent')
            # Get the minimum and maximum values from the anomaly dataset
            min_val, max_val = [np.nanmin(np.ma.masked_invalid(values)), 
                                np.nanmax(np.ma.masked_invalid(values))]
            
            # Enter custom bounds
            custom = False
            
            if custom:
                min_val_, max_val_ = -0.55, 1
                # Cap the bounds at 2-sigma to ensure normalization of data
                if min_val < min_val_:
                    min_val = min_val_
                    extend = 'both'
                if max_val > max_val_:
                    max_val = max_val_
                    extend = 'both'
            # Define normalization about 0
            norm = colors.TwoSlopeNorm(vmin=min_val, vcenter=0., vmax=max_val)
        else:
            # Define colormap
            cmap = colormaps(params[k], 'sequential')
            # Get the minimum and maximum values from the anomaly dataset
            min_val, max_val = [np.nanmin(np.ma.masked_invalid(values)), 
                                np.nanmax(np.ma.masked_invalid(values))]
            # Cap the bounds between 0 and the extreme value
            norm = colors.Normalize(vmin=min_val, vmax=max_val)
            if min_val >= 0 and max_val >= 0:
                extend = 'max'
            else:
                extend = 'min'
        
        # Get colorbar limits at 2-sigma and levels. Add the try/except loop for missing data handling.
        scale_param = 0.2 if np.log(max_val) >= 0 else 0.002
        try:
            min_val_r, max_val_r = [round(scale_param * round(float(min_val)/scale_param), 3), round(scale_param * round(float(max_val)/scale_param), 3)]
        except:
            min_val_r, max_val_r = np.nan, np.nan
        
        # Contour levels must be increasing, so nan if they're equal
        if min_val_r == max_val_r:
            min_val_r, max_val_r = np.nan, np.nan
        
        levels = np.linspace(min_val_r, max_val_r, 11)

        im = ax.contourf(x, y, values, cmap=cmap, norm=norm, extend=extend, levels=levels)

        # If quiver option is enabled, add quiver plot to selected parameters
        if params[k] in quiver_params:
            nx, ny, f = 2, 2, 2
            
            quiver_values = data.sel(site=sites[j]).isel(height=slice(None, None, f*ny)).groupby('time.hour').mean()
            
            X, Y = np.meshgrid(quiver_values['hour'], quiver_values['height'])
            
            q = ax.quiver(X.T[::nx], Y.T[::ny], 
                          -np.sin(quiver_values['wind_direction']*np.pi/180)[::nx], 
                          -np.cos(quiver_values['wind_direction']*np.pi/180)[::ny], 
                          pivot='middle', scale=25, 
                          width=0.02)

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title('{0}'.format(site_names[sites[j]], fontsize=10))
        
        ax.set_xlabel('Hour [LST]')
        ax.set_ylabel('Height [m]', labelpad=10)
        ax.label_outer()
        
        ax.set_xticks([0, 10, 20])

        # Add subplot text identifier
        if j == 0:
            ann = ax.annotate(xy=(0, 0), xytext=(0.9, 0.88), text=subplot_letters[k], 
                        xycoords='axes fraction', fontsize=12)
            ann.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Place colorbar at the end of each row
        if j == ncols-1:
            cax = make_axes_locatable(ax).append_axes('right', 
                                                      size='5%', 
                                                      pad=0.1)
            colorbar = fig.colorbar(im, cax=cax, extend=extend)
            colorbar_label = colorbar.set_label('$\mathregular{{{0}}}$'.format(data[params[k]].attrs['units']), rotation=270, labelpad=20)

    # Figure formatting
    # fig.suptitle('Heat Wave Anomalies')
    fig.tight_layout()
    # Legend formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), frameon=False)

def vertical_profiles(data, sites=['BRON', 'QUEE', 'STAT'], hour=[0], params=['temperature']):
    '''
    Plot vertical profile for averaged data at a given time.
    '''

    if len(data) != 2:
        return None

    norm, hw = data
    n_avg, n_std = norm.groupby(
        'time.hour').mean(), norm.groupby('time.hour').std()
    hw_avg, hw_std = hw.groupby(
        'time.hour').mean(), hw.groupby('time.hour').std()

    # Define cyclic properties
    plt_cycler = cycler(color=['b', 'r', 'g', 'm'])

    nrows, ncols, szf = len(params), len(sites), 2.5
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows),
                            nrows=nrows, ncols=ncols, dpi=300, sharey=True)

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title('{0} \n {1}'.format(
                sites[j], params[k].replace('_', ' ').title()))
        else:
            ax.set_title(params[k].replace('_', ' ').title())
        ax.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])
        ax.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])
        # Define formatting cycle for axis
        ax.set_prop_cycle(plt_cycler)

        # Main data
        im_n = ax.plot(n_avg[params[k]].sel(
            site=sites[j], hour=hour), n_avg.height, label='Normal')
        im_hw = ax.plot(hw_avg[params[k]].sel(
            site=sites[j], hour=hour), hw_avg.height, label='Extreme heat event')

        # Standard deviations
        im_n_std = ax.fill_betweenx(n_avg.height, n_avg[params[k]].sel(site=sites[j], hour=hour).values - n_std[params[k]].sel(site=sites[j], hour=hour),
                                    n_avg[params[k]].sel(
                                        site=sites[j], hour=hour) + n_std[params[k]].sel(site=sites[j], hour=hour),
                                    alpha=0.2)
        im_hw_std = ax.fill_betweenx(hw_avg.height,
                                     hw_avg[params[k]].sel(
                                         site=sites[j], hour=hour) - hw_std[params[k]].sel(site=sites[j], hour=hour),
                                     hw_avg[params[k]].sel(
                                         site=sites[j], hour=hour) + hw_std[params[k]].sel(site=sites[j], hour=hour),
                                     alpha=0.2)

    # Figure formatting
    fig.suptitle('Properties at {0}:00 LST'.format(hour))
    fig.tight_layout()
    # Legend formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), frameon=False)

def vertical_profiles_daily_grid(data, sites=['BRON', 'QUEE', 'STAT'], hours=[6, 12, 18], params=['potential_temperature', 'mixing_ratio'], std=True):
    '''
    Plot vertical profile for averaged data at a given time.
    '''

    param = params[0]
    
    # Single data set boolean
    single = False

    if len(data) == 1:
        single = True
    elif len(data) == 0:
        return None

    if single:
        norm = data[0]
    else:
        norm, hw = data
        
    n_avg, n_std = norm.groupby(
        'time.hour').mean(), norm.groupby('time.hour').std()
    if not single:
        hw_avg, hw_std = hw.groupby(
            'time.hour').mean(), hw.groupby('time.hour').std()

    # Define cyclic properties
    # plt_cycler = cycler(color=['b', 'r', 'g', 'm'])
    markers = ['o', 's']

    nrows, ncols, szf = len(sites), len(hours), 1.7
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows),
                            nrows=nrows, ncols=ncols, dpi=300, sharex=True, sharey=True)

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')
        
        # Subplot formatting
        # Use normal data for minimum and heat wave data for maximum
        if not single:
            ax.set_xlim([np.nanmin(n_avg[params[0]]), np.nanmax(hw_avg[params[0]])])
        ax.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])

        if len(params) > 1:
            ax_ = ax.twiny()
            if not single:
                ax_.set_xlim([np.nanmin(n_avg[params[1]]), np.nanmax(hw_avg[params[1]])])
            ax_.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])
            
        mc = 0
        im_n1 = ax.plot(
            n_avg[params[0]].sel(site=sites[k], hour=hours[j]), 
            n_avg.height, 
            label=params[0], 
            marker='o', markevery=4, markersize=4, 
            color='b', zorder=2)
        
        if std:
            im_n1_std = ax.fill_betweenx(
                n_avg.height,
                n_avg[params[0]].sel(site=sites[k], hour=hours[j]) - n_std[params[0]].sel(site=sites[k], hour=hours[j]),
                n_avg[params[0]].sel(site=sites[k], hour=hours[j]) + n_std[params[0]].sel(site=sites[k], hour=hours[j]),
                color='b',
                alpha=0.1, zorder=2)
        
        if not single:
            im_hw1 = ax.plot(
                hw_avg[params[0]].sel(site=sites[k], hour=hours[j]), 
                hw_avg.height, 
                marker='s', markevery=4, markersize=4, 
                label=params[0], 
                color='r', 
                zorder=2)
            
            if std:
                im_hw1_std = ax.fill_betweenx(
                    hw_avg.height,
                    hw_avg[params[0]].sel(site=sites[k], hour=hours[j]) - hw_std[params[0]].sel(site=sites[k], hour=hours[j]),
                    hw_avg[params[0]].sel(site=sites[k], hour=hours[j]) + hw_std[params[0]].sel(site=sites[k], hour=hours[j]),
                    color='r',
                    alpha=0.1, zorder=2)
            
            if len(params) > 1:
                im_n2 = ax_.plot(n_avg[params[1]].sel(site=sites[k], hour=hours[j]), n_avg.height, label=params[1], marker='o', markevery=4, markersize=4, linestyle='dotted', color='steelblue', zorder=2, fillstyle='none')
            
                if std:
                    im_n2_std = ax_.fill_betweenx(
                        n_avg.height,
                        n_avg[params[1]].sel(site=sites[k], hour=hours[j]) - n_std[params[1]].sel(site=sites[k], hour=hours[j]),
                        n_avg[params[1]].sel(site=sites[k], hour=hours[j]) + n_std[params[1]].sel(site=sites[k], hour=hours[j]),
                        color='steelblue',
                        alpha=0.1, zorder=2)
                
            
                if not single:
                    im_hw2 = ax_.plot(hw_avg[params[1]].sel(site=sites[k], hour=hours[j]), hw_avg.height, marker='s', markevery=4, markersize=4, label=params[1], linestyle='dotted', color='firebrick', zorder=2, fillstyle='none')
                    if std:
                        im_hw2_std = ax_.fill_betweenx(
                            hw_avg.height,
                            hw_avg[params[1]].sel(site=sites[k], hour=hours[j]) - hw_std[params[1]].sel(site=sites[k], hour=hours[j]),
                            hw_avg[params[1]].sel(site=sites[k], hour=hours[j]) + hw_std[params[1]].sel(site=sites[k], hour=hours[j]),
                            color='firebrick',
                            alpha=0.1, zorder=2)
        
        # Only plot one y-axis label. 
        # I'm sure there's a better way to do this using 'sharey'
        label_size = 10
        if j == 0 and k == 1:
            ax.set_ylabel('Height [m]', labelpad=15, fontsize=label_size)
            
        # Modify plot formatting based on parameter
        if params[0] == 'U' or params[0] == 'w':
            ax.set_ylim([100, 2500])
        if params[0] == 'w':
            ax.set_xlim([-1, 1])
        
        # Set subplot in-plot text label
        if params[0] == 'potential_temperature':
            if k == 0:
                ax.set_title('{0}:00 LST'.format(hours[j]), fontsize=10)
                
            if j == 0:
                t = ax.text(0.12, 0.87, '{0}'.format(subplot_letters[k]), 
                        transform=ax.transAxes, size=12, ha='right')
                t.set_bbox(dict(facecolor='none', edgecolor='none', zorder=99))
        elif params[0] == 'U':
            if k == 0:
                ax.set_title('{0}:00 LST'.format(hours[j]), fontsize=10)
                
            if j == 0:
                t = ax.text(0.95, 0.05, '{0}'.format(subplot_letters[k]), 
                        transform=ax.transAxes, size=12, ha='right')
                t.set_bbox(dict(facecolor='none', edgecolor='none', zorder=99))
        else:
            if k == 0:
                ax.set_title('{0}:00 LST'.format(hours[j]), fontsize=10)
                
            if j == 0:
                t = ax.text(0.95, 0.87, '{0}'.format(subplot_letters[k]), 
                        transform=ax.transAxes, size=12, ha='right')
                t.set_bbox(dict(facecolor='none', edgecolor='none', zorder=99))


    # Figure formatting
    # Primary axis label
    if params[0] in ['u', 'v', 'w']:
        fig.text(0.56, -0.03, 
                 ('{0} [$\mathregular{{{1}}}$]'.format(
                     params[0].replace('_', ' '), 
                     norm[params[0]].attrs['units'])), 
                 ha='center', size=label_size)
    else:
        fig.text(0.56, -0.03, 
                 ('{0} [$\mathregular{{{1}}}$]'.format(
                     params[0].replace('_', ' ').title(), 
                     norm[params[0]].attrs['units'])), 
                 ha='center', size=label_size)
    # Secondary axis label
    if len(params) > 1:
        fig.text(0.5, 1, 
                 ('{0} [$\mathregular{{{1}}}$]'.format(
                     params[1].replace('_', ' ').title(), 
                     norm[params[1]].attrs['units'])), 
                 ha='center')
        # fig.suptitle('{0}'.format(param.replace('_', ' ').title()))
        fig.tight_layout()
   
    # Legend formatting
    lines = [mpl.lines.Line2D([0], [0], color='b', label='Normal'),
             mpl.lines.Line2D([0], [0], color='r', label='Extreme heat event')]
    
    fig.legend(handles=lines, frameon=False, bbox_to_anchor=(0.55, 1.03), loc='center', ncol=2)
    fig.subplots_adjust(top=0.9)
    fig.tight_layout()
    
def vertical_profiles_daily_row(data, sites=['BRON', 'QUEE', 'STAT'], hours=[6, 12, 18], params=['potential_temperature', 'mixing_ratio'], std=True):
    '''
    Plot vertical profile for averaged data at a given time.
    '''

    param = params[0]
    
    # Single data set boolean
    single = False

    if len(data) == 1:
        single = True
    elif len(data) == 0:
        return None

    if single:
        norm = data
    else:
        norm, hw = data
        
    n_avg, n_std = norm.groupby(
        'time.hour').mean(), norm.groupby('time.hour').std()
    if not single:
        hw_avg, hw_std = hw.groupby(
            'time.hour').mean(), hw.groupby('time.hour').std()

    # Define cyclic properties
    # plt_cycler = cycler(color=['b', 'r', 'g', 'm'])
    markers = ['o', 's']

    nrows, ncols, szf = 1, len(hours), 1.5
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows*1.5),
                            nrows=nrows, ncols=ncols, dpi=300, sharex=True, sharey=True)

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')
        
        # Subplot formatting
        # Use normal data for minimum and heat wave data for maximum
        if not single:
            ax.set_xlim([np.nanmin(n_avg[params[0]]), np.nanmax(hw_avg[params[0]])])
        ax.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])

        if len(params) > 1:
            ax_ = ax.twiny()
            if not single:
                ax_.set_xlim([np.nanmin(n_avg[params[1]]), np.nanmax(hw_avg[params[1]])])
            ax_.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])
            
        # Line styles and colors
        ls = ['dotted', 'dashed', 'dashdot']
        cs = ['tab:orange', 'tab:green', 'tab:purple']
        # Iterate over all sites
        for m, site in enumerate(sites):
            # Control labeling to prevent legend repetition
            if j == 0:
                label = site_names[site]
            else:
                label = '_nolegend_'
            # Plot the nominal dataset
            im_n1 = ax.plot(
                n_avg[params[0]].sel(site=site, hour=hours[j]), 
                n_avg.height, 
                label=label, 
                linestyle=ls[m], lw=2, color=cs[m], zorder=2)
            # Plot standard deviation shading
            if std:
                im_n1_std = ax.fill_betweenx(
                    n_avg.height,
                    n_avg[params[0]].sel(site=site, hour=hours[j]) - n_std[params[0]].sel(site=site, hour=hours[j]),
                    n_avg[params[0]].sel(site=site, hour=hours[j]) + n_std[params[0]].sel(site=site, hour=hours[j]),
                    color=cs[m],
                    alpha=0.1, zorder=2)
            
            if not single:
                im_hw1 = ax.plot(
                    hw_avg[params[0]].sel(site=site, hour=hours[j]), 
                    hw_avg.height, linestyle=ls[m], color=cs[m], 
                    label=label, zorder=2)
                
                if std:
                    im_hw1_std = ax.fill_betweenx(
                        hw_avg.height,
                        hw_avg[params[0]].sel(site=site, hour=hours[j]) - hw_std[params[0]].sel(site=site, hour=hours[j]),
                        hw_avg[params[0]].sel(site=site, hour=hours[j]) + hw_std[params[0]].sel(site=site, hour=hours[j]),
                        color=cs[m],
                        alpha=0.1, zorder=2)
                
                if len(params) > 1:
                    im_n2 = ax_.plot(n_avg[params[1]].sel(site=site, hour=hours[j]), n_avg.height, label=params[1], marker='o', markevery=4, markersize=4, linestyle='dotted', color='steelblue', zorder=2, fillstyle='none')
                
                    if std:
                        im_n2_std = ax_.fill_betweenx(
                            n_avg.height,
                            n_avg[params[1]].sel(site=site, hour=hours[j]) - n_std[params[1]].sel(site=site, hour=hours[j]),
                            n_avg[params[1]].sel(site=site, hour=hours[j]) + n_std[params[1]].sel(site=site, hour=hours[j]),
                            color='steelblue',
                            alpha=0.1, zorder=2)
                    
                
                    if not single:
                        im_hw2 = ax_.plot(hw_avg[params[1]].sel(site=site, hour=hours[j]), hw_avg.height, marker='s', markevery=4, markersize=4, label=params[1], linestyle='dotted', color='firebrick', zorder=2, fillstyle='none')
                        if std:
                            im_hw2_std = ax_.fill_betweenx(
                                hw_avg.height,
                                hw_avg[params[1]].sel(site=site, hour=hours[j]) - hw_std[params[1]].sel(site=site, hour=hours[j]),
                                hw_avg[params[1]].sel(site=site, hour=hours[j]) + hw_std[params[1]].sel(site=site, hour=hours[j]),
                                color='firebrick',
                                alpha=0.1, zorder=2)
        
        # Only plot one y-axis label. 
        # I'm sure there's a better way to do this using 'sharey'
        label_size = 10
        if j == 0 and k == 1:
            ax.set_ylabel('Height [m]', labelpad=15, fontsize=label_size)
            
        # Modify plot formatting based on parameter
        if params[0] == 'U' or params[0] == 'w':
            ax.set_ylim([100, 2500])
        if params[0] == 'w':
            ax.set_xlim([-1, 1])
        
        # Set subplot in-plot text label
        if params[0] == 'potential_temperature':
            if k == 0:
                ax.set_title('{0}:00 LST'.format(hours[j]), fontsize=10)
                
            if j == 0:
                t = ax.text(0.12, 0.87, '{0}'.format(subplot_letters[k]), 
                        transform=ax.transAxes, size=12, ha='right')
                t.set_bbox(dict(facecolor='none', edgecolor='none', zorder=99))
        elif params[0] == 'U':
            if k == 0:
                ax.set_title('{0}:00 LST'.format(hours[j]), fontsize=10)
                
            if j == 0:
                t = ax.text(0.95, 0.05, '{0}'.format(subplot_letters[k]), 
                        transform=ax.transAxes, size=12, ha='right')
                t.set_bbox(dict(facecolor='none', edgecolor='none', zorder=99))
        else:
            if k == 0:
                ax.set_title('{0}:00 LST'.format(hours[j]), fontsize=10)
                
            if j == 0:
                t = ax.text(0.95, 0.87, '{0}'.format(subplot_letters[k]), 
                        transform=ax.transAxes, size=12, ha='right')
                t.set_bbox(dict(facecolor='none', edgecolor='none', zorder=99))


    # Figure formatting
    # Primary axis label
    if params[0] in ['u', 'v', 'w']:
        fig.text(0.56, -0.03, 
                 ('{0} [$\mathregular{{{1}}}$]'.format(
                     params[0].replace('_', ' '), 
                     norm[params[0]].attrs['units'])), 
                 ha='center', size=label_size)
    else:
        fig.text(0.56, -0.03, 
                 ('{0} [$\mathregular{{{1}}}$]'.format(
                     params[0].replace('_', ' ').title(), 
                     norm[params[0]].attrs['units'])), 
                 ha='center', size=label_size)
    # Secondary axis label
    if len(params) > 1:
        fig.text(0.5, 1, 
                 ('{0} [$\mathregular{{{1}}}$]'.format(
                     params[1].replace('_', ' ').title(), 
                     norm[params[1]].attrs['units'])), 
                 ha='center')
        # fig.suptitle('{0}'.format(param.replace('_', ' ').title()))
        fig.tight_layout()
   
    # Legend formatting
    lines = [mpl.lines.Line2D([0], [0], color='b', label='Normal'),
             mpl.lines.Line2D([0], [0], color='r', label='Extreme heat event')]
    
    fig.legend(frameon=False, bbox_to_anchor=(0.55, 1.03), loc='center', ncol=3)
    fig.subplots_adjust(top=0.9)
    fig.tight_layout()

def histogram(datasets, sites=['QUEE'], primary_group='time.hour', secondary_group='wind_direction', height=0):
    '''
    Create histogram of values based on an xArray Dataset, a parameter and desired groupings.
    '''
    
    # Define bins for the parameter to be secondarily grouped by
    if secondary_group == 'zeta':
        bins = [-np.inf, -1, -0.1, 0.1, 1, np.inf]
        cols = ['Highly unstable', 'Unstable', 'Neutral', 'Stable', 'Highly stable'] 
    elif secondary_group == 'wind_direction':
        bins = np.linspace(0, 360, num=5, endpoint=True)
        cols = ['Northeasterly', 'Southeasterly', 'Southwesterly', 'Northwesterly']
        
    # Initialize list of DataFrames
    dfs = [None] * len(datasets) * len(sites)
    # Initialize index for placing each Dataset
    loc = 0
    for m, site in enumerate(sites):
        # Iterate through given datasets
        for n, dataset in enumerate(datasets):
            # Initialize DataFrame with secondary grouping
            df = pd.DataFrame(columns=cols)
            # Intialize iterand for primary grouping
            i = 0
            # Iterate through primary group
            for group, data_ in dataset.sel(site=site, height=height).groupby(primary_group):
                # Intialize empty dictionary to align values with secondary groups
                temp = {key: 0 for key in cols}
                # Iterate through secondary group
                for subgroup, subdata in data_.groupby_bins(secondary_group, bins, include_lowest=True, labels=cols):
                    # Append value to corrresponding dictionary location
                    if len(subdata[secondary_group].values) > 0:
                        temp[subgroup] = len(subdata[secondary_group].values)
                    else:
                        temp[subgroup] = 0
                # Assign to DataFrame
                df.loc[i] = temp
                i += 1
            # Get sum to derive percentages.
            df['sum'] = df.sum(axis=1)
            df = df.iloc[:, :-1].div(df['sum'], axis=0)
            print(np.nansum(df.iloc[12:21]['Southeasterly'] / df.iloc[12:21].sum(axis=1)) / 9)
            df = df * 100
            
            dfs[m + len(sites)*n] = df
    
    ''' Plotting. '''
    nrows, ncols = len(datasets), len(sites)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                           dpi=300, sharex=True, sharey=True)
    if secondary_group != 'wind_direction':
        cmap = 'RdBu'
    else:
        cmap = 'tab20c'
        
    for i, ax_ in enumerate(fig.axes):
        
        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')
        
        im = dfs[i].plot(kind='bar', stacked=True, 
                     cmap=cmap, ax=ax_, width=0.7, label=None, legend=None)
        ax_.set_ylim([0, 100])
        
        # Set site name as axis title for first row
        if k == 0:
            ax_.set_title(site_names[sites[j]], fontsize=10)
        # Set time label as x-axis
        if j == 1:
            ax_.set_xlabel('Hour [LST]', labelpad=10)
            
        # Add subplot text identifier
        if j == 0:
            ann = ax_.annotate(xy=(0, 0), xytext=(0.95, 0.85), 
                               text=subplot_letters[k], xycoords='axes fraction',
                               fontsize=12, ha='right')
            ann.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
        ax_.set_xticklabels(ax_.get_xticklabels(), rotation = 0, ha='center')
        [l.set_visible(False) for (i,l) in enumerate(ax_.xaxis.get_ticklabels()) if i % 3 != 0]
    
    fig.text(-0.02, 0.55, 'Occurence frequency [%]', 
              ha='center', va='center', rotation='vertical')
    fig.legend(cols, ncol = len(cols), bbox_to_anchor=(0.53, 1.04), loc='center', frameon=False, handlelength=0.8)
    fig.tight_layout()

def quality_plot(data, grouped=False, param='temperature'):
    '''
    Plots a 2D pcolormesh with hourly counts of data for a given dataset and parameter.

    Parameters
    ----------
    dataset : xArray Dataset
        The data for which record stats will be shown.
    param : str
        xArray parameter for which stats will be shown.

    '''
    
    # Merge 2 different datasets for combined statistics
    dataset = xr.merge(data)
    # Define parameters for pcolormesh (lidar only)
    lidar_params = ['u', 'v']
    
    # Initialize list to hold hourly-counted datasets
    hourly = []
    # Iterate over all sites in the dataset to get site-specific counts
    for site in dataset.site.values:
        # Skip Manhattan
        if site == 'MANH':
            continue
        else:
            # Handle lidar-specific parameters
            if param in lidar_params:
                # Count number of observations per height level
                temp = dataset[param].sel(site=site).groupby('time.hour').count()
                # Adjust lidar count
                temp[19] = temp[19] * np.nanmean(temp) / np.nanmean(temp[19])
                hourly.append(temp.T)
            else:
                # Count number of observations per height level
                temp = dataset[param].sel(site=site).groupby('time.hour').count()
                # Adjust lidar count
                if param == 'w':
                   temp[19] = temp[19] * np.nanmean(temp) / np.nanmean(temp[19])
                hourly.append(temp.T)
    
    # Plot lidar data
    if param in lidar_params:
        fig, ax = plt.subplots(dpi=300, ncols=len(hourly), sharey=True)
        for i, ax_ in enumerate(ax):
            im = ax_.pcolormesh(hourly[i].hour, hourly[i].height, hourly[i], 
                                edgecolor=(0, 0, 0, 0.2), lw=0.1)
            ax_.set_xlim([0, 23])
            ax_.set_ylim([100, 3000])
            ax_.set_title(site_names[str(hourly[i].site.values)])
            if i == 0:
                ax_.set_ylabel('Height [m]', labelpad=10)
            if i == 1:
                ax_.set_xlabel('Hour [LST]')
        
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([1, 0.14, 0.03, 0.77])
        colorbar = fig.colorbar(im, cax=cbar_ax)
        colorbar_label = colorbar.set_label('Observation count', rotation=270, labelpad=20)
        
    # Plot microwave radiometer data
    else:
        heights = [2000, 1000, 500, 100]
        fig, ax = plt.subplots(dpi=300, nrows=len(heights), ncols=len(hourly), 
                               sharex=True, sharey=True)
        for i, ax_ in enumerate(ax):
            for j, subax_ in enumerate(ax_):
                if i == 0 and j != 1:
                    subax_.set_title(site_names[str(hourly[j].site.values)])
                elif i > 0 and j == 1:
                    subax_.set_title('{0} m'.format(heights[i]))
                elif i == 0 and j == 1:
                    subax_.set_title('{0}, {1} m'.format(site_names[str(hourly[j].site.values)], heights[i]))
                if i == len(heights)-1 and j == 1:
                    subax_.set_xlabel('Hour [LST]')
                im = subax_.bar(hourly[j].hour.values, 
                                hourly[j].sel(height=heights[i]))
        
            fig.text(0, 0.5, 'Observation count', 
                     ha='center', va='center', rotation='vertical')
    fig.tight_layout()
    
    ''' Handle wind direction data. '''
    del fig, ax
    
    # Define minimum and maximum radial ticks
    r_min, r_max = 0, 0
    # Define height and site lists
    heights, sites = [[1000, 100], 
                      [site for site in dataset.site.values if site != 'MANH']]
    nrows, ncols, factor = len(heights), len(sites), 2.5
    fig, axs = plt.subplots(figsize=(factor*ncols, factor*nrows),
                            nrows=nrows, ncols=ncols,
                            subplot_kw={'projection': 'polar'})
    for i, ax in enumerate(fig.axes):
        j = i % ncols
        k = i // len(hourly)
        
        df = dataset.sel(site=sites[j], 
                         height=heights[k]).groupby_bins('wind_direction', np.arange(0, 360, 15)).count()['u'].to_dataframe()
        bins = [item.left for item in df.index]
        df_ = pd.DataFrame(columns=['bins', 'count'])
        df_['bins'] = bins
        df_['count'] = df['u'].values
    
        theta = df_['bins']*np.pi/180
        r = df_['count']
        
        r_max = r.max() if r.max() > r_max else r_max
        r_min = r.min() if r.min() > r_min else r_min
        
        bottom = 300
        ax.grid(True, linestyle=':')
        ax.xaxis.grid(False)
        theta_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        if j == 1 and k == 1:
            ax.set_xticklabels(theta_ticks)
        else:
            ax.set_xticklabels([])
        
        num, roundto = 5, 50
        ax.spines['polar'].set_visible(False)
        
        ax.bar(theta, r, width=0.2, bottom=bottom, zorder=10)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        title_y = 1.1
        if k == 0:
            ax.set_title(site_names[sites[j]], y=title_y)
        if j == 1:
            ax.set_title('{0} m'.format(heights[k]), y=title_y)
        if k == 0 and j == 1:
            ax.set_title('{0}, {1} m'.format(site_names[sites[j]], heights[k]), y=title_y)
        if j == 1 and k == 1:
            ax.set_title('{0} m'.format(heights[k]), y=title_y+0.07)
        
    for i, ax in enumerate(fig.axes):
        
        radial_ticks = np.linspace(bottom, r_max + bottom, num)
        radial_ticks = np.append(radial_ticks, radial_ticks[-1] + np.diff(radial_ticks)[0])
        radial_ticks = [str(roundto * round(s/roundto) - bottom) for s in radial_ticks]
        ax.set_yticklabels(radial_ticks)
        ax.set_rlabel_position(15)
        ax.set_yticks(np.linspace(bottom + r_min, bottom + r_max, 5))[1:]
        # ticks = ax.get_yticklabels()
        # ax.set_yticklabels([str(s.get_text()) + '\n \n' for s in ticks])
        
            
    fig.tight_layout()

def side_by_side_contour(data, dates=None, site='BRON', param='temperature'):
    '''
    Plot contour plot with side-by-side dates.
    '''
    
    if dates and (len(data) == 2):
        data = xr.merge(data)

    if dates:
        nrows, ncols, szf = 1, len(dates), 2.5
    else:
        nrows, ncols, szf = 1, len(data), 2.5
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows),
                            nrows=nrows, ncols=ncols, dpi=300)
    
    extrema = []
    min_val, max_val, extend = None, None, None
    for i, ax in enumerate(fig.axes):
        
        if  dates:
            date_ = [(pd.to_datetime(dates[i]) - datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
                        (pd.to_datetime(dates[i]) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')]
            
            date_ = slice(date_[0], date_[1])
        else:
            date_ = slice('2017-01-01', '2021-09-30')
        
        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')
        # Select values
        if dates:
            values = data[param].sel(site=site, time=date_).groupby('time.hour').mean().T
        else:
            values = data[i][param].sel(site=site, time=date_).groupby('time.hour').mean().T
        # Main data
        x, y = values.hour, values.height
        # Define colorbar extension direction(s)
        extend = None

        # If anomaly has both positive and negative data, use a divergent colormap. Else, use sequential.
        if np.nanmin(values) < 0 and np.nanmax(values) > 0:
            # Define colormap
            cmap = colormaps(param, 'divergent')
            # Get the minimum and maximum values from the anomaly dataset
            min_val, max_val = [np.nanmin(np.ma.masked_invalid(values)), 
                                np.nanmax(np.ma.masked_invalid(values))]
            # Enter custom bounds
            custom = False
            if custom:
                min_val_, max_val_ = 0, 25
                # Cap the bounds at 2-sigma to ensure normalization of data
                if min_val < min_val_:
                    min_val = min_val_
                    extend = 'both'
                if max_val > max_val_:
                    max_val = max_val_
                    extend = 'both'
                else:
                    extend='both'
            # Define normalization about 0
            norm = colors.TwoSlopeNorm(vmin=min_val, vcenter=0., vmax=max_val)
        else:
            # Define colormap
            cmap = colormaps(param, 'sequential')
            # Get the minimum and maximum values from the anomaly dataset
            min_val_, max_val_ = [np.nanmin(np.ma.masked_invalid(values)), 
                                  np.nanmax(np.ma.masked_invalid(values))]
            # Cap the bounds between 0 and the extreme value
            norm = colors.Normalize(vmin=min_val, vmax=max_val)
            if not min_val:
                min_val = min_val_
            if not max_val:
                max_val = max_val_
            if min_val_ < min_val:
                min_val = min_val_
            if max_val_ > max_val:
                max_val = max_val_
            if min_val >= 0 and max_val >= 0:
                extend = 'max'
            else:
                extend = 'min'
         
        extrema.append(min_val)
        extrema.append(max_val)
        # Get colorbar limits at 2-sigma and levels
        level = 0.25
        min_val_r, max_val_r = [round(level * round(float(min(extrema))/level), 2), round(level * round(max(extrema)/level), 2)]
        
        print(extrema, min_val, max_val, min_val_r, max_val_r)
        
        levels = np.linspace(min_val_r, max_val_r, 15)
        
        # Determine colorbar extension, if any
        if (min_val_r > min_val) & (max_val_r < max_val):
            extend = 'both'
        elif min_val < min_val_r:
            extend = 'min'
        elif max_val > max_val_r:
            extend = 'max'
            
        if np.nanmin(values) < 0 and np.nanmax(values) > 0:
            norm = colors.TwoSlopeNorm(vmin=min_val_r, vcenter=0., vmax=max_val_r)
        else:
            norm = colors.Normalize(vmin=min_val_r, vmax=max_val_r)
            
        im = ax.contourf(x, y, values, cmap=cmap, norm=norm, extend=extend, levels=levels)
        
        # Set titles
        if dates:
            ax.set_title(dates[i], fontsize=10)
        # Set custom title for normal/heat wave comparison
        if i == 0:
            ax.set_title('Normal', fontsize=10)
        if i == 1:
            ax.set_title('Extreme heat event', fontsize=10)
        
        if param in ['U', 'w', 'u']:
            ax.set_ylim([100, 2500])
        ax.set_ylabel('Height [m]', labelpad=10)
        ax.label_outer()
        
        # Define x-tick labels
        ax.set_xticks(np.arange(0, 24, 3))
        n = 2  # Keeps every nth label
        [l.set_visible(False) for (i,l) in 
         enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]

        # Place colorbar at the end of each row
        if j == ncols-1:
            cax = make_axes_locatable(ax).append_axes('right', 
                                                      size='5%', 
                                                      pad=0.1)
            colorbar = fig.colorbar(im, cax=cax, extend=extend)
            if dates:
                colorbar_label = colorbar.set_label('$\mathregular{{{0}}}$'.format(data[param].attrs['units']), rotation=270, labelpad=20)
            else:
                colorbar_label = colorbar.set_label('$\mathregular{{{0}}}$'.format(data[i][param].attrs['units']), rotation=270, labelpad=20)

    # Figure formatting
    fig.text(0.5, -0.03, 'Hour [LST]', ha='center', size=10)
    fig.tight_layout()
    # Legend formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), frameon=False)
    
    return min_val, max_val

if __name__ == '__main__':
    print('Run test...')
