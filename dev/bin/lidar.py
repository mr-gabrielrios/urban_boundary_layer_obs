"""
Urban Boundary Layer Observation Data Processing
Script name:    Lidar Analysis
Path:           ~/bin/lidar.py
Description:    Process and analyze data from the Leosphere WindCube lidar on top of Steinman Hall.
"""

import datetime, netCDF4 as nc, numpy as np, os, pandas as pd, xarray as xr

# Notes:
# Data files 
# Reconstructed lidar data comes in at 2 or 4 s intervals (see Leosphere WindCube Scan Software Suite User Manual, Version 20.a, Section 6.6)

def processor(date_range, sites=['BRON', 'MANH', 'QUEE', 'STAT']):
    '''
    Processes raw lidar data and returns an aggregate xArray over the dates and locations provided.

    Parameters
    ----------
    date_range : list
        Dates over which lidar data is desired.
    sites : list, optional
        List of sites at which data is desired. The default is ['BRON', 'MANH', 'QUEE', 'STAT'].

    Returns
    -------
    data : xArray Dataset
         xArray with data over the dates and locations provided.

    '''
    
    # Define file path to external hard drive
    fpath = '/Volumes/UBL Data/data/lidar'
    # Initialize empty dictionary for future use
    file_dict = {}
    # Build list of netCDF files for future concatenation into an xArray
    for root, dirs, files in os.walk(fpath):
        for directory in dirs:
            # Catch the loop and skip to next location if the site was not requested
            if directory not in sites:
                print('Skipping ', directory)
            else:
                print('Processing ', directory)
                # Build temporary list of files for each location
                temp = []
                # Iterate through files for each location
                for file in os.listdir(os.path.join(root, directory)):
                    # Only select netCDF files
                    if file.split('.')[-1] == 'nc':
                        filename = file.split('.')[0]
                        # Handle date formatting for different lidar networks
                        # Manhattan: City College of New York
                        # Bronx, Queens, Staten Island: New York State Mesonet
                        if directory == 'MANH':
                            date = datetime.datetime.strptime(filename, '%Y%m%d%H%M%S')
                        else:
                            date = datetime.datetime.strptime(filename, '%Y%m%d')
                        # If file isn't within the specified date range, ignore it
                        if date_range[0] < date < date_range[-1]:
                            temp.append(os.path.join(root, directory, filename + '.nc'))
                # Populate site-specific dictionary item with aggregated list
                file_dict[directory] = temp
    
    # Aggregate lidar data into single xArray by iterating through the dictionary of files
    ds_list = []
    for key in file_dict.keys():
        print(key)
        if key == 'MANH':
            # Initialize dictionary to hold sweep names that correspond to each file
            sweeps = {}
            # Initialize list to hold all xArray Datasets for future concatenation
            data_list = []
            for file in file_dict[key]:
                sweeps[file] = list(nc.Dataset(file).groups.keys())[1]
            for file in file_dict['MANH']:
                print('\t ', file)
                temp = xr.open_dataset(file, group=sweeps[file], decode_times=False)
                # Quality mask using confidence interval > 99%
                mask = temp['radial_wind_speed_ci'].data > 99
                # Obtain masked array of vertical velocity
                w = np.ma.array(temp['radial_wind_speed'].data, mask=~mask)
                # Create custom xArray
                temp = xr.Dataset(data_vars={'w': (['time', 'height'], w), 
                                             'ci': (['time', 'height'], temp['radial_wind_speed_ci'].data)}, 
                                  coords={'time': temp.time.values, 
                                          'height': temp.range.values})
                # Masked data
                data_list.append(temp)
            # Concatenate data and sort by time
            data = xr.concat(data_list, dim='time').sortby('time')
            # Assign the current file location to the xArray Dataset
            data = data.assign_coords(site=key).expand_dims('site')
        else:
            for file in file_dict[key]:
                print('\t ', file)
                # Open netCDF file
                temp = nc.Dataset(file)
                # Get group name where data is stored
                group = list(temp['radial'].groups.keys())[0]
                # Quality mask using confidence interval > 99%
                mask = temp['radial'][group]['confidence'][:, :].data > 99
                # Obtain masked array of vertical velocity
                w = np.ma.array(temp['radial'][group]['w'][:, :].data, mask=~mask)
                # Get date corresponding to current file
                date = datetime.datetime.strptime(file.split('/')[-1].split('.')[0], '%Y%m%d')
                # Create time vector from milliseconds array in the netCDF 'time' variable
                times = [datetime.timedelta(milliseconds=float(t)) + date for t in temp['radial'][group]['time'][:].data]
                # Get height vector from array in the netCDF 'height' variable
                heights = temp['radial'][group]['range'][:].data
                # Create custom xArray
                data = xr.Dataset(data_vars={'w': (['time', 'height'], w), 
                                             'ci': (['time', 'height'], temp['radial'][group]['confidence'][:, :].data)}, 
                                  coords={'time': times, 
                                          'height': heights})
                
                data = data.assign_coords(site=key).expand_dims('site')
                ds_list.append(data)
        data = xr.concat(ds_list, dim='time')    
    
    return data

if __name__ == '__main__':
    date_range = pd.date_range(start='2021-08-01', end='2021-08-02', freq='D')
    data = processor(date_range, sites=['BRON', 'MANH', 'QUEE', 'STAT'])