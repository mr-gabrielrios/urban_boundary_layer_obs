#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urban Boundary Layer Observation Data Processing
Script name:    Main
Path:           ~
"""

''' Imports '''
import datetime, netCDF4 as nc, numpy as np, os, pandas as pd, time, xarray as xr

''' Data access - LiDAR '''
def data_access_lidar(temp_freq='1H'):
    '''
    Method accesses raw data and returns xArray Datasets for processing.

    Parameters
    ----------
    temp_freq : str
        Temporal frequency for upsampling of time axis, allowing for uniform timestamps across all data.
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects for reference on how to format this string. Default value of '1H', or 1 hour, provided.

    Returns
    -------
    lidar_data : xArray Dataset
        Contains xArray Dataset of compiled data from the netCDF4 files identified with the os.walk script.

    '''
    
    # Access all data in the directory containing lidar data.
    data_fpath = os.path.join(os.getcwd(), 'data/lidar')
    # Store all lidar files in list. Required for temporal file sorting and mass read-in (xArray open_mfdataset()).
    file_list = []
    # Iterate through all folders in the directory. Aggregate all netCDF file data for future processing.
    # Assume file structure is 'lidar' > 'PROF_[SITE]' > '[YYYY]' > '[MM]', where text between brackets is variable.
    for subdir, dirs, files in os.walk(data_fpath):
        for file in files:
            # Define universal file path
            fpath = subdir + os.sep + file
            # Process netCDF files, extract xArray Dataset, write to .nc file, and compile new .nc files into list. 
            # This allows the use of xr.open_mfdataset with the list. 
            # This is convoluted, but is ~15x faster than using xr.concat.
            if fpath.endswith('.nc') and 'processed' not in fpath:
                # Remove file if it exists
                if os.path.isfile(fpath.split('.')[0]+'_processed.nc'):
                    os.remove(fpath.split('.')[0]+'_processed.nc')
                # Get site location
                site = subdir.split('_')[-1].split('/')[0]
                # Define processed file path
                proc_fpath = os.path.splitext(fpath)[0] + '_processed.nc'
                # Store temporary netCDF4 dataset
                temp_nc = nc.Dataset(fpath)
                # Get netCDF4 Dataset subgroup (varies file to file)
                group_id = list(temp_nc['radial'].groups.keys())[0]
                # Store temporary xArray dataset
                temp_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(temp_nc['radial'][group_id]))
                # Rename dimensions for consistency
                temp_xr = temp_xr.rename({'range': 'height'})
                # Resample time to ensure common time axis
                temp_xr = temp_xr.resample(time=temp_freq).interpolate('linear')
                # Add site label
                temp_xr.coords['site'] = site
                # Add site dimension
                temp_xr = temp_xr.expand_dims(dim='site')
                # Ensure time dimension is only 24 entries long
                if len(temp_xr.time) != 24:
                    continue
                # Write netCDF file
                ds = temp_xr.to_netcdf(path=proc_fpath, mode='w', format='netcdf4')
                # Append dataset to list for future concatenation
                file_list.append(proc_fpath)
    
    # Concatenate all data into singular xArray Dataset
    data = xr.open_mfdataset(file_list, concat_dim='site')
    
    return data
    
''' Data access - Microwave radiometer '''
def data_access_mwr(temp_freq):
    
    # Access all data in the directory containing radiometer data.
    data_fpath = os.path.join(os.getcwd(), 'data/mwr')
    # Store all radiometer files in list. Required for temporal file sorting and mass read-in (xArray open_mfdataset()).
    file_list = []
    # Iterate through all folders in the directory. Aggregate all netCDF file data for future processing.
    # Assume file structure is 'mwr' > 'PROF_[SITE]' > '[YYYY]' > '[MM]', where text between brackets is variable.
    for subdir, dirs, files in os.walk(data_fpath):
        for file in files:
            # Define universal file path
            fpath = subdir + os.sep + file
            # Process netCDF files, extract xArray Dataset, write to .nc file, and compile new .nc files into list. 
            # This allows the use of xr.open_mfdataset with the list. 
            # This is convoluted, but is much faster than using xr.concat.
            if fpath.endswith('.nc') and 'processed' not in fpath:
                # Remove file if it exists
                if os.path.isfile(fpath.split('.')[0]+'_processed.nc'):
                    os.remove(fpath.split('.')[0]+'_processed.nc')
                # Get site location
                site = fpath.split('_')[-1].split('.')[0]
                # Define processed file path
                proc_fpath = os.path.splitext(fpath)[0] + '_processed.nc'
                # Store temporary xArray dataset
                temp_xr = xr.open_dataset(fpath)
                # Rename dimensions for consistency
                temp_xr = temp_xr.rename({'range': 'height'})
                # Average across radiometer angles
                temp_xr = temp_xr.mean(dim='lv2_processor')
                # Time interpolations for each time axis
                temp_xr = temp_xr.resample(time_integrated=temp_freq).interpolate('linear')
                temp_xr = temp_xr.resample(time_surface=temp_freq).interpolate('linear')
                temp_xr = temp_xr.resample(time_vertical=temp_freq).interpolate('linear')
                # Create singular time axis spanning 1 day, since each file is daily
                time = list(temp_xr.time_surface.values)
                # Merge time axes into singular time axis
                temp_xr = temp_xr.reset_index(['time_vertical', 'time_integrated', 'time_surface'], drop=True).assign_coords(time=time).rename({'time_vertical': 'time', 'time_integrated': 'time', 'time_surface': 'time'}).reindex({'time': time})
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
                # Write netCDF file
                ds = temp_xr.to_netcdf(path=proc_fpath, mode='w', format='netcdf4')
                # Append dataset to list for future concatenation
                file_list.append(proc_fpath)
    
    # Concatenate all data into singular xArray Dataset
    data = xr.open_mfdataset(file_list, concat_dim='site')\
    
    return data

def interp(data, dim='height'):
    '''
    Interpolate data within an xArray Dataset.

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset containing data with nans along the given dimension.
    dim : str
        String containing dimension name along which to perform interpolation.

    Returns
    -------
    data : xArray Dataset
        xArray Dataset containing data without nans along the given dimension.

    '''
    
    for varname in data.data_vars:
        data[varname] = data[varname].interpolate_na(dim=dim, method='linear')
        
    return data
        
if __name__ == '__main__':
    timer = time.time()
    lidar_data = data_access_lidar(temp_freq='1H')
    mwr_data = data_access_mwr(temp_freq='1H')
    # Merge lidar and radiometer data
    data = xr.merge([mwr_data, lidar_data])
    # Interpolate data linearly on missing height intervals
    data = interp(data)
    print('Elapsed time: {0} s'.format(time.time() - timer))