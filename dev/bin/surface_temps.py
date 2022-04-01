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
# Change font
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

def processor(date_range):
    
    # Get date range and format it to pass extremes as arguments to the GOES data processing script
    date_range = [int(pd.to_datetime(date).strftime('%Y%m%d%H')) 
                  for date in date_range]
    print(date_range)
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
        try:
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
        except:
            pass
        try:
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
        except:
            pass
    # Concatenate Datasets
    temperatures = xr.concat(temperatures, dim='time')
    
    return temperatures

def contours(data):

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
            levels = np.arange(np.around(vmin/5, decimals=0)*5,
                                 np.around(vmax/5, decimals=0)*5,
                                 2)
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

def temperature_profiles(lst_sst_temperatures, land_coords=[18, 28], sea_coords=[28, 28]):
    
    # Define coordinates for each point
    land_x, land_y = land_coords
    sea_x, sea_y = sea_coords
    
    # Define land and sea surface temperatures
    lsts = lst_sst_temperatures['temperature'].isel(x=land_x, y=land_y)
    ssts = lst_sst_temperatures['temperature'].isel(x=sea_x, y=sea_y)
    
     # Group by hour and acquire mean and standard deviations
    lsts_mean, lsts_std = [lsts.groupby('time.hour').mean(),
                           lsts.groupby('time.hour').std()]
    ssts_mean, ssts_std = [ssts.groupby('time.hour').mean(),
                         ssts.groupby('time.hour').std()]
    # Remove outliers
    sigma = 3
    # Initialize lists to hold hourly DataArrays for future concatenation
    lsts_, ssts_ = [], []
    # Iterate over all hours to remove outliers on an hourly basis
    for hour in range(0, 24):
        # Mask out all hours not belonging to iterand hour 
        lsts_mask = pd.to_datetime(lsts.time.values).hour == hour
        # Define working data
        lsts_working = lsts[lsts_mask]
        # Filter out outliers
        lsts_working = lsts_working.where((lsts_working > (lsts_mean.isel(hour=hour).values - sigma*lsts_std.isel(hour=hour).values)) & (lsts_working < (lsts_mean.isel(hour=hour).values + sigma*lsts_std.isel(hour=hour).values)), drop=True)
        # Append to lists for future concatenation
        lsts_.append(lsts_working)
        
        # Same routine, but for SSTS
        ssts_mask = pd.to_datetime(ssts.time.values).hour == hour
        ssts_working = ssts[ssts_mask]
        ssts_working = ssts_working.where((ssts_working > (ssts_mean.isel(hour=hour).values - sigma*ssts_std.isel(hour=hour).values)) & (ssts_working < (ssts_mean.isel(hour=hour).values + sigma*ssts_std.isel(hour=hour).values)), drop=True)
        ssts_.append(ssts_working)
        
    # Concatenate LSTs
    lsts_ = xr.concat(lsts_, dim='time')
    # Remove duplicate times
    _, index = np.unique(lsts_['time'], return_index=True)
    lsts_ = lsts_.isel(time=index)
    # Get mean and standard deviations
    lsts_mean, lsts_std = lsts_.groupby('time.hour').mean(), lsts_.groupby('time.hour').std()
    
    # Concatenate SSTs
    ssts_ = xr.concat(ssts_, dim='time')
    # Remove duplicate times
    _, index = np.unique(ssts_['time'], return_index=True)
    ssts_ = ssts_.isel(time=index)
    # Get mean and standard deviations
    ssts_mean, ssts_std = ssts_.groupby('time.hour').mean(), ssts_.groupby('time.hour').std()
    
    # Get difference between Queens (18, 28) and New York Bight (28, 28)
    diff = lsts_ - ssts_
    mean_diff, std_diff = diff.groupby('time.hour').mean(), diff.groupby('time.hour').std()
    
    ''' Plotting. '''
    fig, ax = plt.subplots(2, 1, dpi=300, figsize=(4, 4), sharex=True, 
                           gridspec_kw={'height_ratios': [2, 3]})
    # Plot LST data
    lst_color = 'forestgreen'
    lsts_im = ax[0].plot(lsts_mean.hour, lsts_mean, label='LST', color=lst_color)
    lsts_im_ = ax[0].fill_between(lsts_mean.hour, 
                                  lsts_mean - lsts_std, lsts_mean + lsts_std, alpha=0.2, color=lst_color)
    # Plot SST data
    sst_color = 'royalblue'
    ssts_im = ax[0].plot(ssts_mean.hour, ssts_mean, label='SST', color=sst_color)
    ssts_im_ = ax[0].fill_between(ssts_mean.hour, 
                                  ssts_mean - ssts_std, ssts_mean + ssts_std, alpha=0.2, color=sst_color)
    
    ax[0].annotate('a', (0, 0), (0.93, 0.83), xycoords='axes fraction', fontsize=16)
    
    # Plot temperature differences
    diff_color = 'tab:purple'
    ax[1].plot([0, 24], [0, 0], color=(0, 0, 0, 0.5), linewidth=1, linestyle=':')
    im_mean = ax[1].plot(mean_diff.hour, mean_diff, color=diff_color)
    im_std = ax[1].fill_between(mean_diff.hour, 
                                mean_diff - std_diff, mean_diff + std_diff, alpha=0.2, color=diff_color)
    
    ax[1].annotate('b', (0, 0), (0.93, 0.85), xycoords='axes fraction', fontsize=16)
    
    ax[0].legend(frameon=False, loc='upper left')
    ax[1].set_xlim([0, 23])
    
    # Define x-tick labels
    for ax_ in fig.axes:
        ax_.set_xticks(np.arange(0, 24, 3))
        n = 2  # Keeps every nth label
        [l.set_visible(False) for (i,l) in 
         enumerate(ax_.xaxis.get_ticklabels()) if i % n != 0]
    
    # Plot metadata
    ax[0].set_ylabel('Temperature [K]', labelpad=10)
    ax[1].set_ylabel('$T_{LST} - T_{SST}$ [K]', labelpad=10)
    ax[1].set_xlabel('Hour [LST]', labelpad=10)
    fig.tight_layout()

if __name__ == '__main__':
    # Boolean for sample runs of the script
    trial = False
    # Pull list of pre-generated days with sea breeze events from .csv
    sea_breeze_days = [datetime.datetime.strptime(date, '%m/%d/%y')
                       for date in pd.read_csv('bin/assets/sea_breeze_days.csv')['date'].tolist()]
    # Shorten list of dates for trial runs
    if trial:
        sea_breeze_days = sea_breeze_days[::10]
    # Generate hourly date ranges for each day
    sea_breeze_range = [pd.date_range(day, day+datetime.timedelta(days=1), freq='H')
                        for day in sea_breeze_days if day.year >= 2020]
    # Try loading lst_sst_temperatures if loaded into memory. Else, run it.
    try:
        lst_sst_temperatures
    except:
        # Initialize list of temperatures to hold xArray data
        lst_sst_temperatures = []
        # Iterate through all dates and append xArray to list
        lst_sst_temperatures = xr.concat([processor(date_range) for date_range in sea_breeze_range], dim='time')
        
    temperature_profiles(lst_sst_temperatures)
        
   
    