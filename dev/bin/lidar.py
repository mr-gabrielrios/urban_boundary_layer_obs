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

def processor(date_range, spectral_analysis=False, sites=['BRON', 'MANH', 'QUEE', 'STAT']):
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
                continue
            else:
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
                        # Note: "if date in date_range" being trialed to work with specific dates
                        if date_range[0] <= date < date_range[-1]:
                        # if date.date() in date_range:
                            temp.append(os.path.join(root, directory, filename + '.nc'))
                # Populate site-specific dictionary item with aggregated list
                file_dict[directory] = temp
    
    # Aggregate lidar data into single xArray by iterating through the dictionary of files
    ds_list = []
    for key in file_dict.keys():
        if key == 'MANH':
            # Initialize dictionary to hold sweep names that correspond to each file
            sweeps = {}
            # Initialize list to hold all xArray Datasets for future concatenation
            data_list = []
            for file in file_dict[key]:
                sweeps[file] = list(nc.Dataset(file).groups.keys())[1]
            for file in file_dict['MANH']:
                temp = xr.open_dataset(file, group=sweeps[file], decode_times=False)
                # Quality mask using confidence interval > 99%
                mask = temp['radial_wind_speed_ci'].data > 99
                # Obtain masked array of vertical velocity
                w = np.ma.array(temp['radial_wind_speed'].data, mask=~mask)
                # Create time vector from 'seconds since Unix time' array
                times = [datetime.datetime.utcfromtimestamp(float(t)) for t in temp.time]
                print(times)
                # Create custom xArray
                temp = xr.Dataset(data_vars={'w': (['time', 'height'], w), 
                                             'ci': (['time', 'height'], temp['radial_wind_speed_ci'].data)}, 
                                  coords={'time': times, 
                                          'height': temp.range.values})
                # Assign the current file location to the xArray Dataset
                temp = temp.assign_coords(site=key).expand_dims('site')
                # Resample data to lower frequencies if not being used for spectral analysis
                if not spectral_analysis:
                    temp = temp.resample(time='30T').mean()
                    temp_std = temp.resample(time='30T').std()
                    # Assign standard deviation for resampling operation (average 1-s sampling rate)
                    for var in temp.data_vars:
                        temp = temp.assign({'{0}_std'.format(var): temp_std[var]})
                # Masked data
                data_list.append(temp)
            # Concatenate data and sort by time, if data found
            if data_list:
                temp = xr.concat(data_list, dim='time').sortby('time')
                _, index = np.unique(temp.time, return_index=True)
                temp = temp.isel(time=index)
                ds_list.append(temp)
                
        else:
            for file in file_dict[key]:
                # Open netCDF file
                temp = nc.Dataset(file)
                # Get group name where data is stored. Skip file if there's nothing in there.
                if 'radial' not in temp.groups:
                    continue
                groups = list(temp['radial'].groups.keys())
                # If the first radial group doesn't contain the relevant values, try other groups. Continue if none found.
                group = ''
                for elem in groups:
                    data_flag = all(i in list(temp['radial'][elem].variables) for i in ['u', 'v', 'w', 'direction'])
                    if data_flag:
                        group = elem
                        break
                if not group:
                    continue
                # Quality mask using confidence interval > 99%
                mask = temp['radial'][group]['confidence'][:, :].data <= 95
                # Obtain masked array of velocity components
                u = np.ma.array(temp['radial'][group]['u'][:, :].data, mask=mask, fill_value=np.nan)
                v = temp['radial'][group]['v'][:, :].data
                w = temp['radial'][group]['w'][:, :].data
                # Obtain wind direction array
                direction = temp['radial'][group]['direction'][:, :].data
                # Get date corresponding to current file
                date = datetime.datetime.strptime(file.split('/')[-1].split('.')[0], '%Y%m%d')
                # Create time vector from milliseconds array in the netCDF 'time' variable
                times = [(datetime.timedelta(milliseconds=float(t)) + date) for t in temp['radial'][group]['time'][:].data]
                print(times)
                
                # Get height vector from array in the netCDF 'height' variable
                heights = [float(i) for i in temp['radial'][group]['range'][:].data]
                # Create custom xArray and ensure time sorting
                temp = xr.Dataset(data_vars={'u': (['time', 'height'], u.data),
                                             'v': (['time', 'height'], v.data),
                                             'w': (['time', 'height'], w.data), 
                                             'wind_direction': (['time', 'height'], direction.data), 
                                             'ci': (['time', 'height'], temp['radial'][group]['confidence'][:, :].data)}, 
                                  coords={'time': times, 
                                          'height': heights}).sortby('time')
                # Assign the current file location to the xArray Dataset
                temp = temp.assign_coords(site=key).expand_dims('site')
                # Resample data to lower frequencies if not being used for spectral analysis
                if not spectral_analysis:
                    temp = temp.resample(time='5T').mean()
                # Get standard deviation using the population standard deviation
                temp_std = temp.resample(time='30T').std(ddof=0)
                # Assign standard deviation for resampling operation (average 1-s sampling rate)
                for var in temp.data_vars:
                    temp = temp.assign({'{0}_std'.format(var): temp_std[var]})
                _, index = np.unique(temp.time, return_index=True)
                temp = temp.isel(time=index)
                ds_list.append(temp)
                
    # Concatenate data and sort by time. This for loop cuts off the last lidar data entry of the day to prevent time axis conflicts when merging.
    for i, ds in enumerate(ds_list):
        ds_list[i] = ds_list[i].drop_isel(time=-1)
    data = xr.merge(ds_list, fill_value=np.nan)
    if len(data.dims.values()) > 0:
        data = data.transpose("time", "site", "height")
        if 'time' in data.dims:
            data = data.sortby('time')
        data = quality_filter(data)
    
    return data

def quality_filter(data):
    '''
    Method to filter through lidar data and preserve high confidence data within boundary layer.

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset containing confidence interval ('ci') data variable.

    Returns
    -------
    data : xArray Dataset
        xArray Dataset containing confidence interval ('ci') data variable.

    '''
    
    # Note: this method is imperfect and still lets through some high-quality data above the PBLH.
    
    # Initialize empty data array to hold mask values
    arrs = np.full(shape=data['ci'].shape, fill_value=np.nan)
    # Iterate to filter at each location
    for site_index, site in enumerate(list(data.site.values)):
        site_data = data.sel(site=site)
        # Get data confidence intervals
        arr = site_data['ci'].values
        # Round to prevent floating point issues
        arr = np.round(arr, 2)
        div = 100
        arr = np.around(arr/div, decimals=0)*div
        # Define boundary layer height search sequence
        sequence = np.array([100, 0]) 
        # 
        for i, row in enumerate(arr):
            # Check if row is all nans or zeroes
            null_row = np.isnan(row).all()
            zero_row = ((row == 0).all())
            # Initialize null index value
            index = ''
            # Set all-zero row to nan
            if not null_row and zero_row:
                arr[i, :] = np.nan
            # Identify non-zero and non-nan rows
            if not null_row and not zero_row:
                # Use NumPy sliding window algorithm to find match for sequence array in each time entry
                index = np.all(np.lib.stride_tricks.sliding_window_view(row, len(sequence)) == sequence, axis=1).nonzero()
            # If there's a hit, set all values above identified first index to nan.
            if index:
                if len(index[0]) > 0:
                    value = index[0][0]
                    arr[i, value:] = np.nan
        # Add values to empty array
        arrs[:, site_index, :] = arr
    # Plug filtered values into Dataset
    data = data.assign(ci=(["time", "site", "height"], arrs))
    data = data.where(~np.isnan(data['ci']))
    data = data.where(data['ci'] != 0)
    
    return data

if __name__ == '__main__':
    # Boolean control to handle if spectral analysis is generated
    spectral = False
    if spectral:
        dates = pd.date_range(start='2021-07-31', end='2021-09-01', freq='M', closed='left')
        directory = '/Volumes/UBL Data/data/storage/lidar'
        for i in range(0, len(dates)-1):
            date_range = [dates[i], dates[i+1]]      
            print('Processing: ', date_range)
            data = processor(date_range, spectral_analysis=True, sites=['MANH'])
            if len(data) != 0:
                data = quality_filter(data)
                filename = 'lidar_data_{0}-{1:02d}_spectral.nc'.format(date_range[-1].date().year, date_range[-1].date().month)
                data.to_netcdf(os.path.join(directory, filename))
    else:
        dates = pd.date_range(start='2021-06-01', end='2021-08-01', freq='M', closed='left')
        directory = '/Volumes/UBL Data/data/storage/lidar'
        for i in range(0, len(dates)-1):
            date_range = [dates[i], dates[i+1]]      
            print('Processing: ', date_range)
            data = processor(date_range, spectral_analysis=False, sites=['BRON', 'QUEE', 'STAT'])
            if len(data) != 0:
                data = quality_filter(data)
                filename = 'lidar_data_{0}-{1:02d}.nc'.format(date_range[-1].date().year, date_range[-1].date().month)
                data.to_netcdf(os.path.join(directory, filename))