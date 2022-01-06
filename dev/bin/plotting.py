"""
Urban Boundary Layer Observation Data Processing
Script name:    Plotting Functions
Path:           ~/bin/plotting.py
Description:    This script contains functions to plot data.
"""

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from adjustText import adjust_text
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from metpy.plots import Hodograph as hodo
from windrose import WindroseAxes

# Change font
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

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
              'pressure': 'cividis',
              'U': 'Blues',
              'u': 'Blues',
              'v': 'Blues',
              'w': 'Blues'}
        return cs[param]
    elif mode == 'divergent':
        cs = {'temperature': 'RdBu_r',
              'potential_temperature': 'RdBu_r',
              'virtual_potential_temperature': 'RdBu_r',
              'mixing_ratio': 'BrBG',
              'relative_humidity': 'BrBG',
              'pressure': 'RdBu_r',
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


def lst_sst_plot(data):

    # Set up imports for custom function
    import bin.goes_data_processing
    import cartopy
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import sys
    from bin.goes_data_processing import main as goes_data_processing_main
    import scipy.ndimage
    from scipy.interpolate import griddata 
    # Add a GOES data processing package to the local path
    sys.path.insert(
        0, '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/bin/goes_data_processing')
    # Get date range and format it to pass extremes as arguments to the GOES data processing script
    date_range = [int(pd.to_datetime(date).strftime('%Y%m%d%H'))
                  for date in data.time.values]

    max_temp = 400

    # Get land surface temperature data
    lsts = goes_data_processing_main.main(-74.0060, 40.7128, 0.5,
                                          goes_product='LSTC', date_range=[date_range[0], date_range[-1]])
    # Get sea surface temperature data
    ssts = goes_data_processing_main.main(-74.0060, 40.7128, 0.5,
                                          goes_product='SSTF', date_range=[date_range[0], date_range[-1]])

    # Get list of temperature extremes from all datasets for future colorbar limits
    lst_temps = [[np.nanmin(np.where(data[3] > max_temp, np.nan, data[3])),
                  np.nanmax(np.where(data[3] > max_temp, np.nan, data[3]))] for data in lsts]
    sst_temps = [[np.nanmin(np.where(data[3] > max_temp, np.nan, data[3])),
                  np.nanmax(np.where(data[3] > max_temp, np.nan, data[3]))] for data in ssts]
    # Calculate minimum and maximum values. Escape if data is not available.
    try:
        vmin, vmax = [min([subitem for item in (lst_temps + sst_temps) for subitem in item]),
                      max([subitem for item in (lst_temps + sst_temps) for subitem in item])]
    except:
        return None

    lat, lon, combined, time, extent = None, None, None, None, None
    
    # Iterate through data and plot for each timestamp
    for i in range(len(lsts)):
        print(i, np.shape(lsts))
        # Get land surface temperature data
        lat, lon, lst, time, extent = lsts[i][1], lsts[i][2], lsts[i][3], lsts[i][7], lsts[i][8]
        print(extent)
        # Trim data to match SST data size
        lst = lst[:, :-1]
        # Get sea surface temperature data
        _, _, sst = ssts[i][1], ssts[i][2], ssts[i][3]
        # Get combined LST and SST data by placing SST data where LST data is empty
        combined = np.where(np.isnan(lst), sst, lst)
        # Additional safeguard to remove potentially erroneous data
        combined = np.where(combined > max_temp, np.nan, combined)
        
        # Attempt interpolation using scipy.interpolate.griddata using nearest neighbor
        combined_ = np.ma.masked_invalid(combined)
        lon_ = lon[:, :-1][~combined_.mask]
        lat_ = lat[:, :-1][~combined_.mask]
        combined_ = combined_[~combined_.mask]
        coord_zip = [list(i) for i in zip(lon_.flatten(), lat_.flatten())]
        grid_interp = griddata(coord_zip, combined_.ravel(), (lon[:, :-1], lat[:, :-1]), method='cubic')
    
        # Store point data for SST and LST
        lst_storage, sst_storage = [], []

        # Begin plotting
        proj = ccrs.Orthographic(central_longitude=-74, central_latitude=40.5)
        try:
            fig, ax = plt.subplots(dpi=300, subplot_kw={'projection': proj})
            cmap = mpl.cm.plasma
            ax.set_extent(extent)
            # Set up colorbar levels
            levels = np.linspace(np.around(vmin/5, decimals=0)*5,
                                 np.around(vmax/5, decimals=0)*5,
                                 11)
            # Plot contour
            # im = ax.pcolormesh(lon[:, :-1], lat[:, :-1], combined, zorder=3, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, shading='gouraud')
            im = ax.contourf(lon[:, :-1], lat[:, :-1], grid_interp, zorder=3, transform=ccrs.PlateCarree(), 
                             cmap=cmap, levels=levels, extend='both', interpolate='nearest', alpha=1)

            # Create darkened contour values based on colormap (0 --> black, 1 --> original colors)
            darken_factor = 0.8
            cmap_dark = cmap(np.linspace(0, 1, len(levels))) * \
                [darken_factor, darken_factor, darken_factor, 1]

            im_c = ax.contour(lon[:, :-1], lat[:, :-1], grid_interp, levels=im.levels,
                              zorder=5, linewidth=0.5, colors=cmap_dark, transform=ccrs.PlateCarree())
            ax.clabel(im_c, levels=im.levels[::2],
                      fmt='%2.0f', fontsize=8, colors='k')
            # Plot coastlines
            ax.coastlines('10m', zorder=6)
            # Add colorbar
            cax = fig.add_axes(
                [ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
            cb = plt.colorbar(im, cax=cax)
            
            cb.set_label('Temperature [K]', rotation=270, labelpad=15)
            
            # Grab LST data from Queens (28, 18)
            lst_storage.append(grid_interp[28, 18])
            # Grab SST data from the ocean (28, 30)
            sst_storage.append(grid_interp[28, 30])
            
            ax.set_title(time, loc='left')
        except:
            continue

    fig, ax = plt.subplots(dpi=300)
    print(lst_storage, sst_storage)
    # ax.plot(data.time, lst_storage-sst_storage, 'ko')
    
    return lat, lon, combined, time, extent

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

    if mode == 'hourly':
        data = data.sel(site=site).groupby('time.hour').mean()
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
        im = ax.pcolormesh(data.hour.values, data.height.values,
                           data[param].values, alpha=0.75, cmap=cmap, vmin=vmin, vmax=vmax)
        vectors = ax.quiver(data.hour.values, data.height.values,
                            data['u'].values, data['v'].values, scale=200, cmap=cmap, pivot='mid', width=0.004, zorder=1)

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
        im = ax.pcolormesh(data.hour.values, data.height.values,
                           data[param].values, alpha=0.75, cmap=cmap, vmin=vmin, vmax=vmax)
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

    colorbar = fig.colorbar(im)
    colorbar.set_label('{0} [$\mathregular{{{1}}}$]'.format(param.replace('_', ' ').title(), data[param].units), rotation=270, labelpad=15)
    ax.set_ylabel('Height [m]')

    qk = ax.quiverkey(vectors, X=0.85, Y=1.04, U=10,
                      label='{0} m/s'.format(10), coordinates='axes', zorder=2, labelpos='E')

    if event:
        ax.set_title('{0}, {1}, {2}, {3}'.format(
            site, mode, param, event), loc='left')
    else:
        ax.set_title('{0}, {1}, {2}'.format(site, mode, param), loc='left')
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


def cum_wind_rose(data, site, height, ax=None):

    data = data.sel(site=site, height=height)

    df = pd.DataFrame()
    param = 'U'
    params = [param, 'wind_direction']
    
    bins = {'U': [0, 5, 10, 20, 50, 100],
            'wind_direction': np.linspace(0, 360, num=25, endpoint=True)}
    
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
    width = (2*np.pi) / len(theta)

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
            i/len(df_.columns)))
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

def cum_wind_rose_grouped(data, site, height):

    data = data.sel(site=site, height=height)

    df = pd.DataFrame()
    params = ['zeta', 'wind_direction']
    labels = ['Highly unstable', 'Unstable', 'Slightly unstable',
              'Slightly stable', 'Stable', 'Highly stable']
    bins = {'zeta': [-100, -1, -0.5, 0, 0.5, 1, 100],
            'wind_direction': np.linspace(0, 360, num=25, endpoint=True)}
    data_grouped = data.groupby_bins(
        params[1], bins=bins[params[1]], right=False, include_lowest=True)
    for key in data_grouped.groups.keys():
        s = {}
        temp = data_grouped[key].groupby_bins(
            params[0], bins=bins[params[0]], labels=labels, right=False, include_lowest=True)
        for subkey in temp.groups.keys():
            s[subkey] = len(temp[subkey].unstack().time.values)
        s['wind_direction'] = key.left
        df = df.append(s, ignore_index=True)
    df = df.set_index('wind_direction', drop=True)
    df['sum'] = df.sum(axis=1)
    df = df.iloc[:, :-1].div(df['sum'], axis=0)
    df = df.sort_index()

    # Fill nans with 0, since nan values indicating no occurences, which is equivalent to 0%
    df = df.fillna(0)

    df_ = df.copy()
    cols = ['Highly unstable', 'Unstable', 'Slightly unstable',
            'Slightly stable', 'Stable', 'Highly stable']
    for col in cols:
        if col not in df_.columns:
            df_[col] = 0
    df_ = df_[cols]
    df_.index = df_.index.array * np.pi/180
    bottom = 1.5

    theta = df.index.array * np.pi/180
    width = (2*np.pi) / len(theta)

    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    summation = []
    cmap = mpl.cm.RdBu
    for i in range(0, len(df_.columns)):
        if i == 0:
            summation = [bottom]*len(df_.iloc[:, i])
        else:
            summation = summation + df_.iloc[:, i-1]
        ax.bar(theta, df_.iloc[:, i], width=width, bottom=summation, label=df_.columns[i], color=cmap(
            i/len(df_.columns)), edgecolor=(0, 0, 0, 0.8))
    ax.set_yticklabels([])
    ax.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(1, 0))
    ax.set_title('{0} at {1} m'.format(site, height))

def wind_rose(data, sites=['QUEE'], heights=[100], hours=[0, 24]):

    # Filter times based on user input (end non-inclusive)
    data = data.sel(time=((data.time.dt.hour >= hours[0]) & (data.time.dt.hour < hours[1])))

    nrows, ncols, szf = len(heights), len(sites), 2.5
    fig = plt.figure(figsize=(szf*ncols, szf*nrows), dpi=300)
    plt.axis('off')

    for i in range(0, nrows*ncols):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')

        ax = fig.add_subplot(nrows, ncols, i+1, projection='windrose')

        ax.set_theta_zero_location('E')

        # Initialize data for wind direction
        wd = data['wind_direction'].sel(site=sites[j],
                                        height=heights[k]).values.ravel()
        # Initialize data for wind speed
        ws = data['U'].sel(site=sites[j],
                           height=heights[k]).values.ravel()

        # Bar plot data
        bins = np.arange(0, 20, 5)
        im = ax.bar(wd, ws, bins=bins, normed=True,
                    opening=1, edgecolor='#222222', cmap=plt.cm.Blues)
        
        norm_bins = np.linspace(0, 30, num=4)
        ax.set_yticks(norm_bins)
        ax.set_yticklabels(['{:2.0f}%'.format(i) for i in norm_bins])
        [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i == 0]
        ax.set_title('{0} at {1} m'.format(sites[j], heights[k]))
        
        if (j+1 == ncols) and k == 0:
            ax.legend(frameon=False, bbox_to_anchor=(1.2, 0.5))

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title('{0} \n {1} m'.format(sites[j], heights[k]))
        else:
            ax.set_title('{0} m'.format(heights[k]))

    fig.tight_layout()

def wind_rose_comparison(data, site='QUEE', heights=[100], hours=[0, 24]):
    
    ''' Provide side-by-side comparison between datasets. '''

    # Filter times based on user input (end non-inclusive)
    for dataset in data:
        dataset = dataset.sel(time=((dataset.time.dt.hour >= hours[0]) & (dataset.time.dt.hour < hours[1])))

    nrows, ncols, szf = len(heights), len(data), 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(szf*ncols, szf*nrows), dpi=300, subplot_kw={'projection': 'polar'})

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')
        
        ax = cum_wind_rose(data[j], site, heights[k], ax=ax)
        
        data_type = ['Normal', 'Heat wave']

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title('{0}'.format(data_type[j]))
        if j == 0:
            ax.set_ylabel('{0} m'.format(heights[k]))
            ax.yaxis.set_label_coords(-0.2,0.5)
            
    # Legend formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    names = ['{0} m s⁻¹'.format(s) for s in by_label.keys()]
    fig.legend(by_label.values(), names, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=3)
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
    ax.set_title(param, loc='left')
    ax.set_ylabel('Height [m]')
    ax.legend()


def timeseries(data, sites=['BRON', 'QUEE', 'STAT'], height=0, params=['temperature']):
    '''
    Plot timeseries for averaged data at a given height.
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
    fig, axs = plt.subplots(figsize=(szf*ncols, szf*nrows), nrows=nrows, ncols=ncols, dpi=300)

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title(sites[j])
        ax.set_ylabel('{0} [$\mathregular{{{1}}}$]'.format(params[k].replace('_', ' ').title(), norm[params[k]].attrs['units']))
        ax.label_outer()
        # Define formatting cycle for axis
        ax.set_prop_cycle(plt_cycler)

        # Main data
        im_n = ax.plot(n_avg.hour, n_avg[params[k]].sel(
            site=sites[j], height=height), label='Normal', lw=2)
        im_hw = ax.plot(hw_avg.hour, hw_avg[params[k]].sel(
            site=sites[j], height=height), label='Heat Wave', lw=2)

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

    # Figure formatting
    # fig.suptitle('Properties at {0} m'.format(height))
    fig.tight_layout()
    # Legend formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), frameon=False, ncol=2, loc='upper center')
    fig.subplots_adjust(bottom=0, top=0.93)
    fig.align_ylabels(axs[:, 0])


def contour_timeseries(data, sites=['BRON', 'QUEE', 'STAT'], params=['temperature']):
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
        
        print(hw_avg.height)
        print(n_avg.height)
        print(n_std.height)
        
        anom = xr.apply_ufunc(
            lambda x, m, s: (x-m)/s,
            hw_avg[params[k]].sel(site=sites[j]).T,
            n_avg[params[k]].sel(site=sites[j]).T,
            n_std[params[k]].sel(site=sites[j]).T)

        # If anomaly has both positive and negative data, use a divergent colormap. Else, use sequential.
        if np.nanmin(anom) < 0 and np.nanmax(anom) > 0:
            cmap = colormaps(params[k], 'divergent')
            norm = colors.TwoSlopeNorm(
                vmin=np.nanmin(np.ma.masked_invalid(anom)), 
                vcenter=0., 
                vmax=np.nanmax(np.ma.masked_invalid(anom)))
        else:
            cmap = colormaps(params[k], 'sequential')
            norm = None

        # Get colorbar limits at 2-sigma and levels
        levels = 10

        im = ax.contourf(x, y, anom, cmap=cmap, levels=levels, norm=norm)

        # Subplot formatting
        if k == 0:
            # Ensure only top row of place labels is plotted
            ax.set_title('{0} \n \n {1} '.format(sites[j], params[k].replace('_', ' ').title()), fontsize=10)
        else:
            ax.set_title('{0}'.format(params[k].replace('_', ' ').title(), hw[params[k]].attrs['units']), fontsize=10)
        ax.set_xlabel('Hour [LST]')
        ax.set_ylabel('Height [m]')
        ax.label_outer()

        # Place colorbar at the end of each row
        if j == ncols-1:
            colorbar = fig.colorbar(im, ax=axs[k, j], cmap=cmap)
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

def vertical_profiles_daily(data, sites=['BRON', 'QUEE', 'STAT'], hours=[6, 12, 18], params=['potential_temperature', 'mixing_ratio']):
    '''
    Plot vertical profile for averaged data at a given time.
    '''

    param = params[0]

    if len(data) != 2:
        return None

    norm, hw = data
    n_avg, n_std = norm.groupby(
        'time.hour').mean(), norm.groupby('time.hour').std()
    hw_avg, hw_std = hw.groupby(
        'time.hour').mean(), hw.groupby('time.hour').std()

    # Define cyclic properties
    # plt_cycler = cycler(color=['b', 'r', 'g', 'm'])
    markers = ['o', 's']

    nrows, ncols, szf = 1, len(hours), 3.5
    fig, axs = plt.subplots(figsize=(0.5*szf*ncols, szf*nrows),
                            nrows=nrows, ncols=ncols, dpi=300, sharey=True)

    for i, ax in enumerate(fig.axes):

        # Create indices for rows and columns
        j, k = i % ncols, np.floor(i // ncols).astype('int')

        # Subplot formatting
        # Use normal data for minimum and heat wave data for maximum
        ax.set_xlim([np.nanmin(n_avg[params[0]]), np.nanmax(hw_avg[params[0]])])
        ax.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])
        # Define formatting cycle for axis
        # ax.set_prop_cycle(plt_cycler)

        ax_ = ax.twiny()
        ax_.set_xlim([np.nanmin(n_avg[params[1]]), np.nanmax(hw_avg[params[1]])])
        ax_.set_ylim([np.nanmin(n_avg.height), np.nanmax(n_avg.height)])
        
        # Main data
        for c, site in enumerate(sites):
            im_n1 = ax.plot(
                n_avg[params[0]].sel(site=site, hour=hours[j]), 
                n_avg.height, 
                label=params[0], 
                marker=markers[c % 2], markevery=4, markersize=4, 
                color='b', zorder=3)
            '''
            im_n1_std = ax.fill_betweenx(
                n_avg.height,
                n_avg[params[0]].sel(site=site, hour=hours[j]) - n_std[params[0]].sel(site=site, hour=hours[j]),
                n_avg[params[0]].sel(site=site, hour=hours[j]) + n_std[params[0]].sel(site=site, hour=hours[j]),
                color='b',
                alpha=0.1)
            '''
            im_hw1 = ax.plot(
                hw_avg[params[0]].sel(site=site, hour=hours[j]), 
                hw_avg.height, 
                marker=markers[c % 2], markevery=4, markersize=4, 
                label=params[0], 
                color='r', 
                zorder=3)
            '''
            im_hw1_std = ax.fill_betweenx(
                hw_avg.height,
                hw_avg[params[0]].sel(site=site, hour=hours[j]) - hw_std[params[0]].sel(site=site, hour=hours[j]),
                hw_avg[params[0]].sel(site=site, hour=hours[j]) + hw_std[params[0]].sel(site=site, hour=hours[j]),
                color='r',
                alpha=0.1)
            '''
            
            im_n2 = ax_.plot(n_avg[params[1]].sel(site=site, hour=hours[j]), n_avg.height, label=params[1], marker=markers[c % 2], markevery=4, markersize=4, linestyle='dotted', color='steelblue', zorder=2, fillstyle='none')
            '''
            im_n2_std = ax_.fill_betweenx(
                n_avg.height,
                n_avg[params[1]].sel(site=site, hour=hours[j]) - n_std[params[1]].sel(site=site, hour=hours[j]),
                n_avg[params[1]].sel(site=site, hour=hours[j]) + n_std[params[1]].sel(site=site, hour=hours[j]),
                color='steelblue',
                alpha=0.2)
            '''
            im_hw2 = ax_.plot(hw_avg[params[1]].sel(site=site, hour=hours[j]), hw_avg.height, marker=markers[c % 2], markevery=4, markersize=4, label=params[1], linestyle='dotted', color='firebrick', zorder=2, fillstyle='none')
            '''
            im_hw2_std = ax_.fill_betweenx(
                hw_avg.height,
                hw_avg[params[1]].sel(site=site, hour=hours[j]) - hw_std[params[1]].sel(site=site, hour=hours[j]),
                hw_avg[params[1]].sel(site=site, hour=hours[j]) + hw_std[params[1]].sel(site=site, hour=hours[j]),
                color='firebrick',
                alpha=0.2)
            '''

        text = ax.text(0.89, 0.04, '{0:02d}:00 LST'.format(hours[j]), fontsize=9, ha='left', transform=ax.transAxes, bbox=dict(facecolor=(1, 1, 1, 1), edgecolor='none'), rotation=90)
        
        # Only plot one y-axis label. 
        # I'm sure there's a better way to do this using 'sharey'
        if i == 0:
            ax.set_ylabel('Height [m]')

    # Figure formatting
    # Primary axis label
    fig.text(0.5, 0, 
             ('{0} [$\mathregular{{{1}}}$]'.format(
                 params[0].replace('_', ' ').title(), 
                 norm[params[0]].attrs['units'])), 
             ha='center')
    # Secondary axis label
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

if __name__ == '__main__':
    print('Run test...')
