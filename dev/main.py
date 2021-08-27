#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urban Boundary Layer Observation Data Processing
Script name:    Main
Path:           ~
"""

''' Imports '''
import netCDF4 as nc, numpy as np, os, pandas as pd, time, xarray as xr

''' Data access '''
def data_access():
    '''
    Method accesses raw data and returns xArray Datasets for processing.
    '''
    
    # Temporal frequency for downsampling
    temp_freq = '5S'
    # Access all data in the directory containing lidar data.
    lidar_data_fpath = os.path.join(os.getcwd(), 'data/lidar')
    # Store all lidar files in list. Required for temporal file sorting and mass read-in (xArray open_mfdataset()).
    lidar_file_list = []
    # Iterate through all folders in the directory. Aggregate all netCDF file data for future processing.
    # Assume file structure is 'lidar' > 'PROF_[SITE]' > '[YYYY]' > '[MM]', where text between brackets is variable.
    for subdir, dirs, files in os.walk(lidar_data_fpath):
        for file in files:
            # Define universal file path
            fpath = subdir + os.sep + file
            # Process netCDF files, extract xArray Dataset, write to .nc file, and compile new .nc files into list. 
            # This allows the use of xr.open_mfdataset with the list. 
            # This is convoluted, but is ~15x faster than using xr.concat.
            if fpath.endswith('.nc') and 'processed' not in fpath:
                timer = time.time()
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
                # Resample time to ensure common time axis
                temp_xr = temp_xr.resample(time=temp_freq).interpolate('linear')
                # Add site label
                temp_xr.coords['site'] = site
                # Add site dimension
                temp_xr = temp_xr.expand_dims(dim='site')
                # Write netCDF file
                ds = temp_xr.to_netcdf(path=proc_fpath, mode='w', format='netcdf4')
                # Append dataset to list for future concatenation
                lidar_file_list.append(proc_fpath)
                print(time.time()-timer)
    
    # Concatenate all data into singular xArray Dataset
    lidar_data = xr.open_mfdataset(lidar_file_list, concat_dim='site')
    # Delete all generated files from directory
    for file in lidar_file_list:
        if os.path.isfile(file):
            os.remove(file)
    
    return lidar_data
    
if __name__ == '__main__':
    timer = time.time()
    data = data_access()
    print(time.time() - timer)