"""
Urban Boundary Layer Observation Data Processing
Script name:    Microwave Radiometer Analysis
Path:           ~/bin/mwr.py
Description:    Process and analyze data from the microwave radiometer on top of Steinman Hall.
"""

import netCDF4 as nc, numpy as np, os, pandas as pd, xarray as xr

from bs4 import BeautifulSoup
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import os
import pandas as pd
import requests
import ssl
import urllib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.captureWarnings(True)

def accessor(date_range=[2021080100], data_access='online'):
    '''
    Accesses raw data from flux or ts_data files from the ORSL Loggernet or from the data folder.

    Parameters
    ----------
    date_range : list
        2-item list of dates in YYYYMMDDHH format.
    data_access : str, optional
        Lets user select what the data source is. The defauly is 'online'.
        The URL corresponding to the online option is: https://datadb.noaacrest.org/public/ORSL/Archived_Microwave_Radiometer/
        The directory corresponding to the local option is: '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/mwr'

    Returns
    -------
    data : Pandas DataFrame
        Contains tabular data for the given parameters.

    '''
    
    # Define range of dates in datetime format
    date_range = pd.date_range(start=date_range[0], end=date_range[-1], freq='S').to_pydatetime()
    # Access data from NOAA-CREST database
    if data_access == 'online':
        # Initialize data list
        data_list = []
        # Define URL for flux tower files
        dirpath = 'https://datadb.noaacrest.org/public/ORSL/Archived_Microwave_Radiometer/'
        # Define range of years through which URLs will be accessed. +1 added to stop date since np.arange is stop-exclusive.
        years = np.arange(start=date_range[0].year, stop=date_range[-1].year+1)
        # Iterate through years to access all relevant files
        for year in years:
            # Certificate access workaround
            ssl._create_default_https_context = ssl._create_unverified_context
            # Generate year-specific URL to properly access data
            # 2017:         YYYY/data-lv2/*_lv2.csv
            # 2018:         YYYY/lv2/*_lv2.csv
            # 2019/2020:    YYYY/YYYYMMDD_MWR/*_lv2.csv
            # 2021:         YYYY/YYYY/YYYYMMDD_MWR/*_lv2.csv
            url = ''
            if year in [2017]:
                # Build path based on directory name containing string 'data-lv2'
                url = os.path.join(dirpath, str(year), 'data-lv2')
                # Generate BeautifulSoup for this HTML page
                soup = BeautifulSoup(requests.get(url, verify=False).content, features='html.parser')
                print(url)
                # Get all file paths. Use the icon image as the relevant child tag.
                for img in soup.find_all('img', alt='[TXT]'):
                    # Build path to relevant files
                    href = os.path.join(url, img.find_parent('a')['href'])
                    # Get full date string of each file
                    date = datetime.datetime.strptime(href.split('MFT')[0].split('/')[-1].split('_')[0], '%Y-%m-%d')
                    # Filter file by date range. If file not in date range, skip it
                    if date_range[0] <= date < date_range[-1]:
                        print('\t', href)
                        # Process data from the selected .csv file and return an xArray Dataset
                        temp = processor(href)
                        # Append this Dataset to a list for future concatenation
                        data_list.append(temp)
                    
            elif year in [2018]:
                # Build path based on directory name containing string 'lv2'
                url = os.path.join(dirpath, str(year), 'lv2')
                # Generate BeautifulSoup for this HTML page
                soup = BeautifulSoup(requests.get(url, verify=False).content, features='html.parser')
                # Get all file paths. Use the icon image as the relevant child tag.
                for img in soup.find_all('img', alt='[TXT]'):
                    # Build path to relevant files
                    href = os.path.join(url, img.find_parent('a')['href'])
                    # Get full date string of each file
                    # Get full date string of each file
                    date = datetime.datetime.strptime(href.split('MFT')[0].split('/')[-1].split('_')[0], '%Y-%m-%d')
                    # Filter file by date range. If file not in date range, skip it
                    if date_range[0] <= date < date_range[-1]:
                        print('\t', href)
                        # Process data from the selected .csv file and return an xArray Dataset
                        temp = processor(href)
                        # Append this Dataset to a list for future concatenation
                        data_list.append(temp)
                
            elif year in [2019, 2020, 2021]:
                # Build path to directory containing all data for the given year
                if year in [2019, 2020]:
                    url = os.path.join(dirpath, str(year))
                else:
                    
                    url = os.path.join(dirpath, str(year), str(year))
                # Generate BeautifulSoup for this HTML page
                soup = BeautifulSoup(requests.get(url, verify=False).content, features='html.parser')
                # Get all file paths. Use the icon image as the relevant child tag.
                for img in soup.find_all('img', alt='[DIR]'):
                    # Build path to relevant files
                    href = os.path.join(url, img.find_parent('a')['href'])
                    # Get full date string of each directory if it is a daily directory. Else, continue
                    if 'MWR' in href:
                        date = datetime.datetime.strptime(href.split('/')[-2].split('_')[0],
                                                     '%Y%m%d')
                    else:
                        continue
                    # Get all file paths to daily data for directories within the specified date range. Use the icon image as the relevant child tag.
                    if date_range[0] <= date < date_range[-1]: 
                        # Get BeautifulSoup for each daily directory page
                        day_soup = BeautifulSoup(requests.get(href, verify=False).content, features='html.parser')
                        for img in day_soup.find_all('img', alt='[TXT]'):
                            # Build path to CSV within the daily directory
                            day_href = os.path.join(href, img.find_parent('a')['href'])
                            # Only select the level 2 data
                            if 'lv2' in day_href:
                                print('\t', day_href)
                                # Process data from the selected .csv file and return an xArray Dataset
                                temp = processor(day_href)
                                # Append this Dataset to a list for future concatenation
                                data_list.append(temp)
                    else:
                        continue
                    
        data = xr.concat(data_list, dim='time')
        return data
    
    elif data_access == 'local':
        # Define temporal frequency for data interpolation
        freq = '5T'
         # Access all data in the directory containing radiometer data.
        data_fpath = '/Volumes/UBL Data/data/mwr'
        # Store all radiometer files in list. Required for temporal file sorting and mass read-in (xArray open_mfdataset()).
        file_list = []
        # Iterate through all folders in the directory. Aggregate all netCDF file data for future processing.
        # Assume file structure is 'mwr' > 'PROF_[SITE]' > '[YYYY]' > '[MM]', where text between brackets is variable.
        for subdir, dirs, files in os.walk(data_fpath):
            for file in files:
                # Scrape file date from file name
                date = datetime.datetime.strptime(file.split('_')[0], '%Y%m%d')
                # Filter out dates outside of date range
                if date_range[0] <= date < date_range[-1]:
                    # Define universal file path
                    fpath = subdir + os.sep + file
                    # Process netCDF files, extract xArray Dataset, write to .nc file, and compile new .nc files into list. 
                    # This allows the use of xr.open_mfdataset with the list. 
                    # This is convoluted, but is much faster than using xr.concat.
                    if fpath.endswith('.nc') and 'processed' not in fpath:
                        print(file)
                        # Remove file if it exists
                        if os.path.isfile(fpath.split('.')[0]+'_processed.nc'):
                            os.remove(fpath.split('.')[0]+'_processed.nc')
                        # Get site location
                        site = fpath.split('_')[-1].split('.')[0]
                        # Store temporary xArray dataset
                        temp_xr = xr.open_dataset(fpath)
                        if 'MANH' not in file.split('_')[-1]:
                            # Define processed file path
                            proc_fpath = os.path.splitext(fpath)[0] + '_processed.nc'
                            # Rename dimensions for consistency
                            temp_xr = temp_xr.rename({'range': 'height'})
                            # Average across radiometer angles
                            temp_xr = temp_xr.mean(dim='lv2_processor')
                            # Time interpolations for each time axis
                            temp_xr = temp_xr.resample(time_integrated=freq).interpolate('linear')
                            temp_xr = temp_xr.resample(time_surface=freq).interpolate('linear')
                            temp_xr = temp_xr.resample(time_vertical=freq).interpolate('linear')
                            # Get set of data that is in all 3 time axes
                            setlist = [list(set(list(temp_xr.time_surface.values)) - set(list(temp_xr.time_vertical.values))),
                                    list(set(list(temp_xr.time_surface.values)) - set(list(temp_xr.time_integrated.values))),
                                    list(set(list(temp_xr.time_vertical.values)) - set(list(temp_xr.time_integrated.values)))]
                            # Flatten list and get unique values
                            setlist = list(set([item for sublist in setlist for item in sublist]))
                            # Identify the time axis that is out of order for future dropping
                            if not np.logical_and((len(temp_xr.time_surface.values) == len(temp_xr.time_integrated.values)), (len(temp_xr.time_integrated.values) == len(temp_xr.time_vertical.values))):
                                # Construct dictionary to hold times where there are discrepancies between axes
                                times = {}
                                # Create accessible dictionary (dims.items() produces a FrozenDict)
                                for key, value in temp_xr.dims.items():
                                   times[key] = value
                                # Identify the time axis with the mismatching value
                                drop_dim = [key for key, value in times.items() if value == max(times.values())][0]
                                # For each mismatching value, drop it and its corresponding data from the Dataset
                                for item in setlist:
                                    temp_xr = temp_xr.drop_sel({drop_dim : item})
                            # Create singular time axis spanning 1 day, since each file is daily
                            time = list(temp_xr.time_integrated.values)
                            # Merge time axes into singular time axis
                            temp_xr = temp_xr.reset_index(['time_vertical', 'time_integrated', 'time_surface'], drop=True).assign_coords(time=time).rename({'time_vertical': 'time', 'time_integrated': 'time', 'time_surface': 'time'}).reindex({'time': time})
                        else:
                            proc_fpath = os.path.splitext(fpath)[0] + '_processed.nc'
                            temp_xr = temp_xr.isel(height=range(1, len(temp_xr.height)))
                            temp_xr['height'] = temp_xr['height']*1000
                        # Add site label
                        temp_xr.coords['site'] = site
                        # Add site dimension
                        temp_xr = temp_xr.expand_dims(dim='site')
                        # Add height dimension to all data variables without it (these are surface quantities)
                        for varname in temp_xr.data_vars:
                            if 'height' not in temp_xr[varname].coords:
                                temp_xr[varname] = temp_xr[varname].expand_dims(height=temp_xr.height.values, axis=2)
                                # Set all values for above-surface heights to nan
                                temp_xr[varname] = temp_xr[varname].where(temp_xr.height == 0, np.nan)
                        # Set height coordinate to floats (converts to object datatype as default)
                        temp_xr['height'] = temp_xr['height'].astype('float')
                        # print(list(temp_xr.height.values))
                        # Write netCDF file
                        ds = temp_xr.to_netcdf(path=proc_fpath, mode='w', format='netcdf4')
                        # Append dataset to list for future concatenation
                        file_list.append(proc_fpath)
    
        # Concatenate all data into singular xArray Dataset
        data = xr.open_mfdataset(file_list, concat_dim='site')
        
        print(data)
        
        return data
        
    
               
def processor(url):
    '''
    Processes raw microwave radiometer data .csv file and returns an xArray Dataset.

    Parameters
    ----------
    url : str
        URL to .csv that will be converted into a DataFrame.

    Returns
    -------
    data : xArray Dataset
        xArray Dataset containing microwave radiometer data.

    '''
    
    # Read in CSV and skip irrelevant rows
    raw_data = pd.read_csv(url, skiprows=2)
    # Delete the first 3 rows
    raw_data = raw_data.iloc[3:]
    # Re-define column names to temporary values
    raw_data.columns = raw_data.iloc[0]
    # Remove columns only containing nans
    raw_data = raw_data.iloc[1:].dropna(axis=1, how='all')
    # Rename columns for consistency with NYS Mesonet variable naming conventions
    raw_data = raw_data.rename(columns={'Date/Time': 'time'})
    # Convert all Pandas time values to datetime values
    raw_data['time'] = pd.to_datetime(raw_data['time'])
    # Construct dictionary to dynamically assign data variable names for final xArray
    param_dict = {'temperature': 401, 'vapor_density': 402, 'liquid': 403, 'relative_humidity': 404}
    # Define list of times for use as a dimension and coordinate in final xArray
    times = [i.to_pydatetime() for i in raw_data.loc[raw_data[400] == 401]['time']]
    # Initialize empty list to hold temporary xArrays before merging to final xArray
    params = []
    df = ''
    # Iterate through keys and values to dynamically construct xArrays
    for key, value in param_dict.items():
        # Format DataFrame corresponding to the current iterand by dropping certain columns and re-indexing
        df = raw_data.loc[raw_data[400].isin([value])].drop(columns=['Record', 400, 'LV2 Processor']).set_index('time')
        # Handle blank DataFrame entries by filling them with nans
        df.iloc[np.where(df.applymap(lambda x: x == '  '))[0], np.where(df.applymap(lambda x: x == '  '))[1]] = np.nan
        # Convert all non-index columns to numeric types for future processing
        df = df.apply(pd.to_numeric)
        # Convert all Series with 'object' data types to floats
        for i, item in enumerate(df.dtypes):
            col = df.columns[i]
            if item == 'object':
                df[col] = df[col].str.replace(' ', '').astype(float)
        # Create height range based on DataFrame column names and convert km to m
        heights = [float(i)*1000 for i in df.columns]
        # Only build the Dataset if the value array shapes match the time axis length. Else, find missing rows and append nans
        if df.values.shape[0] != len(times):
            # Get difference in length between DataFrame times and reference time vector
            diff = len(times) - len(df.index.to_pydatetime())
            # Calculate list of timedeltas between the two time vectors referenced above
            delta = [df.index.to_pydatetime()[i] - times[i] for i in np.arange(0, len(df.index.values))]
            # Find any missing data entries by determining if timedelta between temperature datetime and parameter datetime is greater than 40 seconds (typical radiometer return time)
            idxs = np.where(delta >= np.timedelta64(40,'s'))
            # Get array of indexes where timedelta is greater than 40 seconds, suggesting a missing data entry
            # If size of array is 0, just append nans. Else, insert nans in corresponding indices
            if idxs[0].size == 0:
                for i in range(0, diff):
                    temp_df = pd.DataFrame(np.nan, index=[times[len(times) + i - 1]], columns=df.columns)
                    df = df.append(temp_df)
        # Construct temporary xArray using times and heights as dimensions and coordinates
        temp_xr = xr.Dataset(data_vars={key: (['time', 'height'], df.values)}, coords={'time': times, 'height': heights})
        # Append temporary xArray to list of xArrays for future merging
        params.append(temp_xr)
    # Merge all xArrays into singular xArray
    data = xr.merge(params)
    
    return data

def nc_write(root, data):
    '''
    Saves data from xArray Dataset to a netCDF file to remove need for accessing NOAA-CREST database.

    Parameters
    ----------
    root : str
        String defining root directory for storage of the netCDF file.
    data : xArray Dataset
        xArray Dataset containing data within date range specified by user.

    Returns
    -------
    None.

    '''
    
    # Get bounding dates for use in filename string
    dates = [pd.to_datetime(data.time[0].values), pd.to_datetime(data.time[-1].values)]
    # Convert datetimes into strings
    dates = [date.strftime('%Y%m%d%H%M%S') for date in dates]
    # Define filename string
    filename = 'MWR_MANH_s{0}_e{1}.nc'.format(dates[0], dates[-1])
    # Write netCDF file
    data.to_netcdf(os.path.join(root, 'mwr', filename))
                
if __name__ == '__main__':
    root_dir = '/Volumes/UBL Data/data'
    date_range = pd.date_range(start='2020-08-10', end='2020-08-20', freq='D')
    # data = accessor(date_range, data_access='online')
    data = accessor(date_range, data_access='local')
    # nc_write(root_dir, data)
    
 