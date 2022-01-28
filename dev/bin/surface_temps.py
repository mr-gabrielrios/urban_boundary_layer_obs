"""
Urban Boundary Layer Observation Data Processing
Script name:    Surface Temperature Processor
Path:           ~/bin/surface_temps.py
Description:    Process and analyze GOES-R data for a combination of land and sea surface temperatures (LST, SST).
"""

# Set up imports for custom function
import bin.goes_data_processing
import cartopy
import cartopy.crs as ccrs
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.ndimage
import sys
import xarray as xr

from bin.goes_data_processing import main as goes_data_processing_main
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from scipy.interpolate import griddata 

# Add a GOES data processing package to the local path
sys.path.insert(0, '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/bin/goes_data_processing')

def processor(data):
    
    # Get date range and format it to pass extremes as arguments to the GOES data processing script
    date_range = [int(pd.to_datetime(date).strftime('%Y%m%d%H')) for date in data.time.values]
    # Define the maximum allowable surface temperature to prevent anomalous data
    max_temp = 350
    
    ''' Grab surface temperature data using the goes_data_processing package. '''
    # Get land surface temperature data
    lsts = goes_data_processing_main.main(-74.0060, 40.7128, 0.5,
                                          goes_product='LSTC', 
                                          date_range=[date_range[0], date_range[-1]])
    # Get sea surface temperature data
    ssts = goes_data_processing_main.main(-74.0060, 40.7128, 0.5,
                                          goes_product='SSTF', 
                                          date_range=[date_range[0], date_range[-1]])
    
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
    
    # For each new day, reset the values
    lat, lon, combined, time, extent = None, None, None, None, None
    
    # Initialize list of xArray Datasets to be concatenated.
    temperatures = []
    # Iterate through data and plot for each timestamp
    for i in range(len(lsts)):
        # Get land surface temperature data
        lat, lon, lst, time, extent = lsts[i][1], lsts[i][2], lsts[i][3], lsts[i][7], lsts[i][8]
        print(time)
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
        # Expand dimension for the time dimension in the xArray Dataset
        grid_interp = np.expand_dims(grid_interp, axis=2)
        # Define the time to be added to the Dataset
        time = pd.to_datetime(time) - datetime.timedelta(hours=4)
        time = [pd.to_datetime(time.strftime('%Y-%m-%d %H:00:00.0')).to_numpy()]
        
        # Populate temporary xArray Dataset to concatenate later
        temp = xr.Dataset(data_vars={'temperature': (['x', 'y', 'time'], grid_interp)},
                          coords={'lon': (['x', 'y'], lon[:, :-1]),
                                  'lat': (['x', 'y'], lat[:, :-1]),
                                  'time': time})
        temperatures.append(temp)
    # Concatenate Datasets
    temperatures = xr.concat(temperatures, dim='time')
    
    return temperatures

def plotter(data):

    # Get Dataset extrema
    vmin, vmax = np.nanmin(data['temperature'].values), np.nanmax(data['temperature'].values)
    
    # Define list of times to iterate over
    if data.time.shape == ():
        data['time'] = np.atleast_1d(data['time'])
        
    # Iterate over each time and plot
    for i, time in enumerate(data.time):
        
        # Load data from Dataset
        lon, lat, temperatures = [data.sel(time=time)['lon'], 
                                  data.sel(time=time)['lat'], 
                                  data.sel(time=time)['temperature']]
        try:
            # Begin plotting
            proj = ccrs.Orthographic(central_longitude=-74, central_latitude=40.5)
            fig, ax = plt.subplots(dpi=300, subplot_kw={'projection': proj})
            cmap = mpl.cm.plasma
            ax.set_extent([-74.5, -73.6, 40.25, 41.15])
            # Set up colorbar levels
            levels = np.linspace(np.around(vmin/5, decimals=0)*5,
                                 np.around(vmax/5, decimals=0)*5,
                                 11)
            # Plot filled temperature contours
            im = ax.contourf(lon, lat, temperatures, 
                             zorder=3, transform=ccrs.PlateCarree(), 
                             cmap=cmap, levels=levels, extend='both', 
                             interpolate='nearest', alpha=1)
    
            # Create darkened contour values based on colormap (0 --> black, 1 --> original colors)
            darken_factor = 0.8
            cmap_dark = cmap(np.linspace(0, 1, len(levels))) * \
                [darken_factor, darken_factor, darken_factor, 1]
            # Plot level contours
            im_c = ax.contour(lon, lat, temperatures, levels=im.levels,
                              zorder=5, linewidth=0.5, colors=cmap_dark, 
                              transform=ccrs.PlateCarree())
            # Plot level contour labels
            ax.clabel(im_c, levels=im.levels[::2], 
                      fmt='%2.0f', inline_spacing=30, fontsize=8, colors='k')
            # Plot coastlines
            ax.coastlines('10m', zorder=6)
            # Add colorbar
            cax = fig.add_axes([ax.get_position().x1+0.02, 
                                ax.get_position().y0, 
                                0.02, 
                                ax.get_position().height])
            cb = plt.colorbar(im, cax=cax)
            # Set a colorbar label
            cb.set_label('Temperature [K]', rotation=270, labelpad=15)
            # Define the time string to print
            time_str = pd.to_datetime(data.time.values[i]).strftime('%Y-%m-%d %H:%M')
            
            ax.set_title('{0} LST'.format(time_str), loc='left')
        except:
            pass

if __name__ == '__main__':
    print('Running...')