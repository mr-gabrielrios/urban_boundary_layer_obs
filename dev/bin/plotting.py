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
import numpy as np
import logging
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import xarray as xr

from adjustText import adjust_text
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from metpy.plots import Hodograph as hodo
from scipy.ndimage.filters import gaussian_filter
from windrose import WindroseAxes

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.captureWarnings(True)

# Change font
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# Letter strings for subplots
subplot_letters = ['a', 'b', 'c', 'd']
# Full names for sites
site_names = {'BRON': 'The Bronx', 'QUEE': 'Queens', 'STAT': 'Staten Island'}

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


def hodograph(data, sites='all', mode='hourly', time=0, single_site='QUEE'):
    '''
    Generates a hodograph figure, either for a site or overlaid on a map, using wind data at each site.

    Parameters
    ----------
    data : xArray Dataset
        DESCRIPTION.
    sites : str, optional
        Choose visualization method, either 'single' for one site or 'all' for all sites. 'all' results in a visualization overlaid on a map. The default is 'single'.
    mode : str, optional
        Choose temporal mode of data plotted. Will typically be hourly if data is grouped, or exact if data is not grouped.
    time : datetime, optional
        Choose exact time at which data will be plotted. This parameter only matters if mode is exact.
    site : str, optional
        Site at which hodograph will be displayed. This parameter only matters if sites is 'single'. The default is 'QUEE'.

    Returns
    -------
    im : plot

    '''

    def wind_vector(mode, site):
        # Select wind vector data at the desired time
        if mode == 'hourly':
            u = data['u'].sel(site=site).isel(hour=time)
            v = data['v'].sel(site=site).isel(hour=time)
        elif mode == 'exact':
            u = data['u'].sel(site=site).sel(time=time)
            v = data['v'].sel(site=site).sel(time=time)
        return u, v

    # Plotting for single site
    if sites == 'single':
        # Get wind vector data
        u, v = wind_vector(mode, single_site)
        # Plot data using the MetPy hodograph feature
        fig, ax = plt.subplots(dpi=300)
        h = hodo(ax=ax, component_range=max(
            np.nanmax(u.values), np.nanmax(v.values)))
        h.add_grid(increment=2)
        im = h.plot_colormapped(u.values, v.values, u.height, marker='o')
        ax.set_title('Site = {0}, Time = {1}'.format(
            single_site, time), loc='left')
        colorbar = plt.colorbar(im)
        colorbar.set_label('Height [m]')
        return im

    # Plotting for all sites
    elif sites == 'all':
        # Get all coordinates
        coords = [site_coords(loc) for loc in data.site.values]
        # Get central coordinate from all sites by averaging over locations
        coord = (np.nanmean([i[0] for i in coords]),
                 np.nanmean([i[1] for i in coords]))
        # Set window extent for Cartopy plot
        lat_extent, lon_extent = [0.5, 0.5]

        # Initialize figure
        fig, ax = plt.subplots(dpi=300, subplot_kw={
                               'projection': ccrs.PlateCarree()})
        # Set figure extent
        ax.set_extent([coord[1] - lon_extent, coord[1]+lon_extent,
                       coord[0] - lat_extent, coord[0]+lat_extent])
        ax.coastlines()
        # Set data scale relative to figure
        scale = 0.025
        for site in data.site.values:
            if site != 'MANH':
                # Get wind vector data
                u, v = wind_vector(mode, site)

                try:
                    # Calculate figure-specific data offset for proper plotting
                    offset_u, offset_v = [
                        u.values[~np.isnan(u)][0], v.values[~np.isnan(v)][0]]
                    # Prepare data for plotting
                    x, y = [site_coords(site)[1] + scale*(u - offset_u),
                            site_coords(site)[0] + scale*(v - offset_v)]
                    U = np.sqrt(u**2 + v**2)
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate(
                        [points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(0, U.max())
                    lc = LineCollection(segments, cmap='Blues', norm=norm)
                    # Set the values used for colormapping
                    lc.set_array(U)
                    lc.set_linewidth(2)
                    line = ax.add_collection(lc)
                except:
                    continue

        sat_img = cimgt.GoogleTiles(style='satellite')
        ax.add_image(sat_img, 10)
        # Plot colorbar
        fig.colorbar(line, ax=ax)
        # Figure metadata
        ax.set_title('Wind data, {0}'.format(time), loc='left')

        fig.tight_layout()

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

        cmap = 'viridis'
        if not limits:
            vmin, vmax = colorbar_limits(data[param])
            if data[param].min() < 0 and data[param].max() > 0:
                cmap = 'RdBu_r'
        else:
            vmin, vmax = limits
            if vmin < 0 and vmax > 0:
                cmap = 'RdBu_r'

        fig, ax = plt.subplots(dpi=300)
        im = ax.pcontourf(data.hour.values, data.height.values,
                           data[param].values, cmap=cmap, vmin=vmin, vmax=vmax)
        vectors = ax.quiver(data.hour.values, data.height.values,
                            data['u'].values, data['v'].values, scale=200, pivot='mid', width=0.004, zorder=1)

    else:
        data = data.sel(site=site).transpose("height", "time")
        data = data.coarsen(height=3, boundary='trim').mean()

        cmap = 'viridis'
        if not limits:
            vmin, vmax = colorbar_limits(data[param])
            if data[param].min() < 0 and data[param].max() > 0:
                cmap = 'RdBu_r'
        else:
            vmin, vmax = limits
            if vmin < 0 and vmax > 0:
                cmap = 'RdBu_r'

        fig, ax = plt.subplots(dpi=300, figsize=(8, 3))
        im = ax.pcolormesh(data.time.values, data.height.values,
                           data[param].values, alpha=0.75, cmap=cmap, vmin=vmin, vmax=vmax)
        vectors = ax.quiver(data.time.values, data.height.values,
                            data['u'].values, data['v'].values, scale=200, cmap=cmap, pivot='mid', width=0.002, zorder=1)

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


def cum_wind_rose_grouped(data, site, height, hours=[0, 24]):
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
    # Define the relevant groupby parameters
    params = ['zeta', 'wind_direction']
    # Define the secondary groupby labels
    labels = ['Highly unstable', 'Unstable', 'Slightly unstable',
              'Slightly stable', 'Stable', 'Highly stable']
    # Define the groupby bins
    n_bins = 25
    bins = {'zeta': [-np.inf, -2, -1, -0.1, 0.1, 1, np.inf],
            'wind_direction': np.linspace(0, 360, num=n_bins, endpoint=True)}
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
                s['wind_direction'] = key.left
                # Append to DataFrame
                df = df.append(s, ignore_index=True)
        except:
            for subkey in labels:
                # Count number of occurrences in this secondary group
                s[subkey] = 0
                # Attach corresponding primary group
                s['wind_direction'] = key.left
                # Append to DataFrame
                df = df.append(s, ignore_index=True)
    
    # Reset the index to primary group.
    df = df.set_index('wind_direction', drop=True)
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
    
    # Initialize figure
    ax = plt.subplot(111, polar=True)
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
        ax.bar(theta, df_.iloc[:, i], 
               width=width, 
               bottom=summation, 
               label=df_.columns[i], 
               color=cmap(i/len(df_.columns)), 
               linewidth=1, edgecolor='k')
        
    ax.grid(True, linestyle=':')
    ax.xaxis.grid(False)
    
    theta_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticklabels(theta_ticks)
    
    ax.set_yticks([1, 1.5, 2],)
    ax.set_yticklabels(['0%', '50%', '100%'], zorder=10)
    ax.set_rlabel_position(70)
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.2, 1), frameon=False)
    ax.set_title('{0} at {1} m from {2}:00 to {3}:00 LST'.format(site,
                                                                 height, 
                                                                 hours[0], 
                                                                 hours[1]))


def cum_wind_rose(data, site, height, ax=None):

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
        
        data_type = ['Normal', 'Heat wave']

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

    fig, ax = plt.subplots(dpi=300)
    vmin, vmax = np.nan, np.nan
    for site in sites:
        # Hold quantities to be plotted
        quantities, stds = [], []
        for hour in hours:
            temp = data[param].sel(site=site, hour=hour).values
            temp_std = data_std[param].sel(site=site, hour=hour).values
            if np.isnan(vmin) or np.isnan(vmax):
                vmin, vmax = np.nanmin(temp), np.nanmax(temp)
            elif np.nanmin(temp) < vmin:
                vmin = np.nanmin(temp) - np.nanmax(temp_std)
            elif np.nanmax(temp) > vmax:
                vmax = np.nanmax(temp) + np.nanmax(temp_std)
            quantities.append(temp)
            stds.append(temp_std)

        for i in range(len(quantities)):
            im = ax.plot(quantities[i], data.height.values,
                         label='{0}, {1}'.format(site, hours[i]))
            if plot_std:
                std = ax.fill_betweenx(data.height.values,
                                       quantities[i] - stds[i],
                                       quantities[i] + stds[i],
                                       alpha=0.2)

    ax.set_xlim([vmin, vmax])
    ax.set_ylim([min(data.height.values), max(data.height.values)])
    ax.set_title(param, loc='left', pad=20)
    ax.set_ylabel('Height [m]')
    ax.legend()


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

    nrows, ncols, szf = len(params), len(sites), 2
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows*1.3), nrows=nrows, ncols=ncols, dpi=300, sharex=True, sharey=True)

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')
        
        print(sites[j], n_avg[params[k]].sel(site=sites[j], height=height).max(),
              hw_avg[params[k]].sel(
            site=sites[j], height=height).max())

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title(site_names[sites[j]])
        unit = norm[params[k]].attrs['units']
        unit_str = r'$\mathregular{{%s}}$' %unit
        ax.set_xlabel('Hour [LST]')
        ax.set_ylabel('{0} [{1}]'.format(params[k].replace('_', ' ').title(), unit_str), labelpad=10)
        ax.label_outer()
        # Define formatting cycle for axis
        ax.set_prop_cycle(plt_cycler)

        # Main data
        im_n = ax.plot(n_avg.hour, n_avg[params[k]].sel(
            site=sites[j], height=height), label='Normal', lw=2, marker='o', markersize=3)
        im_hw = ax.plot(hw_avg.hour, hw_avg[params[k]].sel(
            site=sites[j], height=height), label='Heat Wave', lw=2, marker='s', markersize=3)

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
        if params[k] == 'specific_humidity':
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))

    # Figure formatting
    # fig.suptitle('Properties at {0} m'.format(height))
    fig.tight_layout()
    # Legend formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.575, 1.1))
    try:
        fig.align_ylabels(axs[:, 0])
    except:
        pass


def contour_timeseries(data, site='QUEE', param='ri'):
    
    # Redefine data with given parameters
    data = data[param].sel(site=site)
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
    data = data.where((data > (mean - sigma*std)) & (data < (mean + sigma*std)))
    
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
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    im = ax.contourf(times, data.height.values, values.T, 
                     norm=norm, cmap=cmap, levels=levels, extend='both')
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

    nrows, ncols, szf = len(params), len(sites), 2.5
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows),
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
        
        print(anom.max())
        
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
        
        ax.set_xlabel('Hour [LST]')
        ax.set_ylabel('Height [m]', labelpad=10)
        ax.label_outer()

        # Add subplot text identifier
        ann = ax.annotate(xy=(0, 0), xytext=(0.06, 0.88), text=subplot_letters[j], 
                    xycoords='axes fraction', fontsize=12)
        ann.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))

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
        # Define formatting cycle for axis
        ax.set_prop_cycle(plt_cycler)

        # Main data
        im_n = ax.plot(n_avg[params[k]].sel(
            site=sites[j], hour=hour), n_avg.height, label='Normal')
        im_hw = ax.plot(hw_avg[params[k]].sel(
            site=sites[j], hour=hour), hw_avg.height, label='Heat Wave')

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

def vertical_profiles_daily(data, sites=['BRON', 'QUEE', 'STAT'], hours=[6, 12, 18], params=['potential_temperature', 'mixing_ratio'], std=True):
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

    nrows, ncols, szf = len(sites), len(hours), 3.5
    fig, axs = plt.subplots(figsize=(0.5*szf*ncols, szf*nrows),
                            nrows=nrows, ncols=ncols, dpi=300, sharex=True, sharey=True)

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')

        # Subplot formatting
        # Use normal data for minimum and heat wave data for maximum
        if not single:
            ax.set_xlim([np.nanmin(n_avg[params[0]]), np.nanmax(hw_avg[params[0]])])
        ax.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])
        # Define formatting cycle for axis
        # ax.set_prop_cycle(plt_cycler)

        if len(params) > 1:
            ax_ = ax.twiny()
            if not single:
                ax_.set_xlim([np.nanmin(n_avg[params[1]]), np.nanmax(hw_avg[params[1]])])
            ax_.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])
        
        # Main data
        for c, site in enumerate(sites):
            
            mc = 0
            im_n1 = ax.plot(
                n_avg[params[0]].sel(site=site, hour=hours[j]), 
                n_avg.height, 
                label=params[0], 
                marker='o', markevery=4, markersize=4, 
                color='b', zorder=2)
            
            if std:
                im_n1_std = ax.fill_betweenx(
                    n_avg.height,
                    n_avg[params[0]].sel(site=site, hour=hours[j]) - n_std[params[0]].sel(site=site, hour=hours[j]),
                    n_avg[params[0]].sel(site=site, hour=hours[j]) + n_std[params[0]].sel(site=site, hour=hours[j]),
                    color='b',
                    alpha=0.1, zorder=2)
            
            if not single:
                im_hw1 = ax.plot(
                    hw_avg[params[0]].sel(site=site, hour=hours[j]), 
                    hw_avg.height, 
                    marker='s', markevery=4, markersize=4, 
                    label=params[0], 
                    color='r', 
                    zorder=2)
                
                if std:
                    im_hw1_std = ax.fill_betweenx(
                        hw_avg.height,
                        hw_avg[params[0]].sel(site=site, hour=hours[j]) - hw_std[params[0]].sel(site=site, hour=hours[j]),
                        hw_avg[params[0]].sel(site=site, hour=hours[j]) + hw_std[params[0]].sel(site=site, hour=hours[j]),
                        color='r',
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
        if i == 0:
            ax.set_ylabel('Height [m]')
            
        # Modify plot formatting based on parameter
        if params[k] == 'U' or params[k] == 'w':
            ax.set_ylim([100, 2500])
        if params[k] == 'w':
            ax.set_xlim([-1, 1])
        
        if params[k] == 'w':
            ax.set_title('{0}:00 LST'.format(hours[j]))
        else:
            t = ax.text(0.95, 0.93, '{0}:00 LST'.format(hours[j]), 
                    transform=ax.transAxes, ha='right')
            t.set_bbox(dict(facecolor='white', edgecolor='none', zorder=99))

    # Figure formatting
    # Primary axis label
    fig.text(0.5, 0, 
             ('{0} [$\mathregular{{{1}}}$]'.format(
                 params[0].replace('_', ' ').title(), 
                 norm[params[0]].attrs['units'])), 
             ha='center')
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
             mpl.lines.Line2D([0], [0], color='r', label='Heat wave'),
             mpl.lines.Line2D([0], [0], color='k', marker='o', lw=0, label=params[0].replace('_', ' ').title()),
             mpl.lines.Line2D([0], [0], color='k', marker='o', lw=0, fillstyle='none', label=params[1].replace('_', ' ').title())]
    
    fig.legend(handles=lines, frameon=False, bbox_to_anchor=(0.5, 1.1), loc='center', ncol=4)
    fig.subplots_adjust(top=0.9)

def histogram(datasets, site='QUEE', primary_group='time.hour', secondary_group='wind_direcction', height=0):
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
    dfs = []
    # Iterate through given datasets
    for dataset in datasets:
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
        
        dfs.append(df)
    
    ''' Plotting. '''
    fig, ax = plt.subplots(ncols=len(datasets), dpi=300, figsize=(6, 3), sharey=True)
    if secondary_group != 'wind_direction':
        cmap = 'RdBu'
    else:
        cmap = 'tab20c'
        
    for i, ax_ in enumerate(ax):
        im = dfs[i].plot(kind='bar', stacked=True, 
                     cmap=cmap, ax=ax_, width=0.7, label=None, legend=None)
        ax_.set_ylim([0, 100])
        
        ax_.set_xlabel('Hour [LST]', labelpad=10)
        ax_.set_xticklabels(ax_.get_xticklabels(), rotation = 0, ha='center')
        [l.set_visible(False) for (i,l) in enumerate(ax_.xaxis.get_ticklabels()) if i % 3 != 0]
        ax_.set_ylabel('Occurence frequency [%]', labelpad=10)
    
    fig.legend(cols, ncol = len(cols), bbox_to_anchor=(0.5, 1.02), loc='center', frameon=False)
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

if __name__ == '__main__':
    print('Run test...')
