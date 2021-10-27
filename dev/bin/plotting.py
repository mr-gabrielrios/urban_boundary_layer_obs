"""
Urban Boundary Layer Observation Data Processing
Script name:    Plotting Functions
Path:           ~/bin/plotting.py
Description:    This script contains functions to plot data.
"""

import cartopy.crs as ccrs, numpy as np, matplotlib as mpl, matplotlib.colors as colors, matplotlib.dates as mdates, matplotlib.pyplot as plt, pandas as pd, xarray as xr

from adjustText import adjust_text
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from metpy.plots import Hodograph as hodo

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
    site_coords = {'BRON': (40.8729, -73.8945), 'MANH': (40.8200, -73.9493), 'QUEE': (40.7366, -73.8201), 'STAT': (40.6021, -74.1504)}
    
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
        h = hodo(ax=ax, component_range=max(np.nanmax(u.values), np.nanmax(v.values)))
        h.add_grid(increment=2)
        im = h.plot_colormapped(u.values, v.values, u.height)
        ax.set_title('Site = {0}, Time = {1}'.format(single_site, time), loc='left')
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
        fig, ax = plt.subplots(dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
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
                    offset_u, offset_v = [u.values[~np.isnan(u)][0], v.values[~np.isnan(v)][0]]
                    # Prepare data for plotting
                    x, y = [site_coords(site)[1] + scale*(u - offset_u), 
                            site_coords(site)[0] + scale*(v - offset_v)]
                    U = np.sqrt(u**2 + v**2)
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(0, U.max())
                    lc = LineCollection(segments, cmap='Blues', norm=norm)
                    # Set the values used for colormapping
                    lc.set_array(U)
                    lc.set_linewidth(2)
                    line = ax.add_collection(lc)
                    
                except:
                    continue
        
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
        im = ax.pcolormesh(data.hour.values, data.height.values, data[param].values, alpha=0.75, cmap=cmap, vmin=vmin, vmax=vmax)
        vectors = ax.quiver(data.hour.values, data.height.values, data['u'].values, data['v'].values, scale=200, pivot='mid', width=0.004, zorder=1)
    
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
        im = ax.pcolormesh(data.hour.values, data.height.values, data[param].values, alpha=0.75, cmap=cmap, vmin=vmin, vmax=vmax)
        vectors = ax.quiver(data.hour.values, data.height.values, data['u'].values, data['v'].values, scale=200, pivot='mid', width=0.004, zorder=1)
    
    else:
        data = data.sel(site=site)
        data = data.resample(time='1H').mean()
        data = data.transpose("height", "time")
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
        im = ax.pcolormesh(data.time.values, data.height.values, data[param].values, cmap='RdBu_r', alpha=0.5)
        vectors = ax.quiver(data.time.values, data.height.values, data['u'].values, data['v'].values, scale=100, scale_units='inches', pivot='mid', width=0.004, zorder=1) 
        
        myFmt = mdates.DateFormatter('%m-%d %H')
        ax.xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate()
        
    colorbar = fig.colorbar(im)
    colorbar.set_label('{0}'.format(param), rotation=270, labelpad=15)
    
    qk = ax.quiverkey(vectors, X=0.85, Y=1.04, U=10, label='{0} m/s'.format(10), coordinates='axes', zorder=2, labelpos='E')
    
    if event:
        ax.set_title('{0}, {1}, {2}, {3}'.format(site, mode, param, event), loc='left')
    else:
        ax.set_title('{0}, {1}, {2}'.format(site, mode, param), loc='left')
    fig.tight_layout()
    plt.show()
    
    return vmin, vmax

def multi_plot(data, param):
    fig, axs = plt.subplots(dpi=300, nrows=2, ncols=2, sharex=True, sharey=True)
    sites = data.site.values
    for i, ax in enumerate(fig.axes):
        bounds = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
        cmap = mpl.cm.RdBu
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='both')
        
        im = ax.pcolormesh(data.time.values, data.height.values, data[param].sel(site=sites[i]).values, norm=norm, cmap=cmap)
        over, under = str(colors.to_hex(cmap(1))), str(colors.to_hex(cmap(0)))
        cmap.set_over('#440154')
        
        ax.set_title('{0}'.format(sites[i]))
    
    fig.autofmt_xdate(rotation=30, ha='right')
    fig.suptitle('Bulk Richardson Number')
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([1.025, 0.15, 0.02, 0.7])
    colorbar = fig.colorbar(im, cax=cax)
    fig.tight_layout()

if __name__ == '__main__':
    param = 'lapse_rate'
    for site in heat_wave_data.site.values:
        vmin, vmax = quiver(heat_wave_data, event='heat wave', site=site, param=param)
        quiver(normal_data, event='normal', site=site, param=param, limits=[vmin, vmax])
        quiver(anomaly_data, site=site, param=param, mode='anomaly')
    
    
    