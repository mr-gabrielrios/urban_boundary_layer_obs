"""
Urban Boundary Layer Observation Data Processing
Script name:    Lidar Analysis
Path:           ~/bin/lidar.py
Description:    Process and analyze data from the Leosphere WindCube lidar on top of Steinman Hall.
"""

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import datetime, netCDF4 as nc, numpy as np, os, pandas as pd, time, xarray as xr

# Notes:
# Data files 
# Reconstructed lidar data comes in at 2 or 4 s intervals (see Leosphere WindCube Scan Software Suite User Manual, Version 20.a, Section 6.6)

def processor(date_range, spectral_analysis=False, sites=['BRON', 'MANH', 'QUEE', 'STAT'], scan_type='FXD'):
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
    if scan_type == 'DBS' and 'MANH' in sites:
        file_dict['MANH'] = [os.path.join('/Volumes/UBL Data/data/lidar/MANH/PROF_MANH', file) for file in os.listdir('/Volumes/UBL Data/data/lidar/MANH/PROF_MANH') if 'dbs' in file]
        
        file_dict['MANH'] = [file for file in file_dict['MANH'] 
                             if (date_range[0] <= 
                                 datetime.datetime.strptime(file.split('/')[-1].split('_')[1], '%Y-%m-%d') < 
                                 date_range[-1])]
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
                # Ensure fixed scan data (FXD) is being handled
                if scan_type == 'FXD':
                    print('Processing FXD: ', file)
                    # Quality mask using confidence interval > 99%
                    mask = temp['radial_wind_speed_ci'].data > 99
                    # Obtain masked array of vertical velocity
                    w = np.ma.array(temp['radial_wind_speed'].data, mask=~mask)
                    # Create time vector from 'seconds since Unix time' array
                    times = [datetime.datetime.utcfromtimestamp(float(t)) for t in temp.time]
                    # Create custom xArray
                    temp = xr.Dataset(data_vars={'w': (['time', 'height'], w), 
                                                 'ci': (['time', 'height'], temp['radial_wind_speed_ci'].data)}, 
                                      coords={'time': times, 
                                              'height': temp.range.values})
                # Handle Doppler beam swinging (DBS) data
                elif scan_type == 'DBS':
                    print('Processing DBS: ', file)
                    # Ensure that DBS data is being handled.
                    if 'dbs' not in file:
                        continue
                    # Open the file and convert to xArray Dataset
                    temp = xr.open_dataset(file, group=sweeps[file], decode_times=False)
                    # Obtain quality filtering for wind data
                    mask = temp['wind_speed_ci'].data > 99
                    # Get zonal wind component from horizontal wind speed
                    u = np.ma.array(temp['horizontal_wind_speed'].values * np.cos(temp['wind_direction'] * np.pi/180), mask=~mask)
                    # Get meridional wind component from horizontal wind speed
                    v = np.ma.array(temp['horizontal_wind_speed'].values * np.sin(temp['wind_direction'] * np.pi/180), mask=~mask)
                    # Get vertical wind component
                    w = np.ma.array(temp['vertical_wind_speed'].values)
                    # Get datetime values for scan times
                    times = [datetime.datetime.utcfromtimestamp(float(t)) for t in temp.time]
                    # Get height array (all values equivalent over axis = 1)
                    heights = temp['measurement_height'].values[0]
                    # Construct the new Dataset
                    temp = xr.Dataset(data_vars={'u': (['time', 'height'], u), 
                                                 'v': (['time', 'height'], v), 
                                                 'w': (['time', 'height'], w), 
                                                 'wind_direction': (['time', 'height'], temp['wind_direction'].data),
                                                 'ci': (['time', 'height'], temp['wind_speed_ci'].data)},
                                      coords={'time': times, 
                                              'height': heights})
                # Assign the current file location to the xArray Dataset
                temp = temp.assign_coords(site=key).expand_dims('site')
                # Resample data to lower frequencies if not being used for spectral analysis
                if not spectral_analysis:
                    temp = temp.resample(time='30T').mean()
                    temp_std = temp.resample(time='30T').std()
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
                print(file)
                benchmark = time.time()
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
                print('Checkpoint 1: {0:.4f}s'.format(time.time() - benchmark))
                
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
                print('Checkpoint 2: {0:.4f}s'.format(time.time() - benchmark))
                if not spectral_analysis:
                    temp = temp.resample(time='5T').mean()
                _, index = np.unique(temp.time, return_index=True)
                temp = temp.isel(time=index)
                ds_list.append(temp)
                
                print('Checkpoint 3: {0:.4f}s'.format(time.time() - benchmark))
                
    # Concatenate data and sort by time. This for loop cuts off the last lidar data entry of the day to prevent time axis conflicts when merging.
    for i, ds in enumerate(ds_list):
        ds_list[i] = ds_list[i].drop_isel(time=-1)
    data = xr.merge(ds_list, fill_value=np.nan)
    if len(data.dims.values()) > 0:
        data = data.transpose("time", "site", "height")
        if 'time' in data.dims:
            data = data.sortby('time')
        data = quality_filter(data)
    
    print('Checkpoint 4: {0:.4f}s'.format(time.time() - benchmark))
    
    return data

def detrend(data, site, averaging_period='30T', detrend_period='5T'):
    import time
    '''
    De-trend raw lidar data to obtain velocity fluctuations and average for a given period.

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset with a < 5 sec temporal resolution.
    period : str, optional
        De-trending period in minutes. The default is '5T'.

    Returns
    -------
    data : xArray Dataset
        xArray Dataset with averaged data along with primary turbulent derived characteristics.
    '''
    
    # Try parallelizing
    from joblib import Parallel, delayed
    import multiprocessing 
    
    def turb_vars_primary(period_data, period_name):
        # Calculate mean velocity quantity per period
        period_data['w_mean'] = period_data['w'].mean()
        
        # Calculate fluctuating velocity quantity
        period_data['w_prime'] = period_data['w'] - period_data['w_mean']
        # If u is in the DataFrame, get the mean and fluctuating quantities
        if 'u' in period_data.columns:
            period_data['u_mean'] = period_data['u'].mean()
            # Calculate fluctuating velocity quantity
            period_data['u_prime'] = period_data['u'] - period_data['u_mean']
        # If u is in the DataFrame, get the mean and fluctuating quantities
        if 'v' in period_data.columns:
            period_data['v_mean'] = period_data['v'].mean()
            # Calculate fluctuating velocity quantity
            period_data['v_prime'] = period_data['v'] - period_data['v_mean']
        return period_data
    
    def turb_vars_secondary(subdata, subgroup):
         # Turbulent kinetic energy
        e = (subdata['u_prime']**2 + subdata['v_prime']**2 + subdata['w_prime']**2)/2
        # Variances
        var_u = subdata['u_prime']**2
        var_v = subdata['v_prime']**2
        var_w = subdata['w_prime']**2
        # Covariances
        cov_u_w = (subdata['u_prime']**2)*(subdata['w_prime']**2)
        cov_u_e = subdata['u_prime']*e
        cov_w_e = subdata['w_prime']*e
        # Append to the DataFrame
        subdata['e'] = e
        subdata['var_u'], subdata['var_v'], subdata['var_w'] = var_u, var_v, var_w
        subdata['cov_u_w'], subdata['cov_u_e'], subdata['cov_w_e'] = cov_u_w, cov_u_e, cov_w_e
        
        mean = subdata.mean().to_frame().T
        mean['time'] = subgroup
        
        return mean
    
    
    def applyParallel(dfGrouped, func):
        ''' Parallel function to be applied to grouped Pandas operations. '''
        retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group, name) for name, group in dfGrouped)
        return pd.concat(retLst)
    
    benchmark = time.time()
    # Initialize temporary container list
    dfs = []
    
    for height in data.height.values:
        working_data = data.sel(site=site, height=height).to_dataframe().reset_index()
        working_data_grouped = working_data.groupby(pd.Grouper(key='time', freq=detrend_period))
        
        '''
        # Iterate over every de-trending period
        for group, period_data in working_data.groupby(pd.Grouper(key='time', freq=detrend_period)):
            # Calculate mean velocity quantity per period
            period_data['w_mean'] = period_data['w'].mean()
            
            # Calculate fluctuating velocity quantity
            period_data['w_prime'] = period_data['w'] - period_data['w_mean']
            # If u is in the DataFrame, get the mean and fluctuating quantities
            if 'u' in period_data.columns:
                period_data['u_mean'] = period_data['u'].mean()
                # Calculate fluctuating velocity quantity
                period_data['u_prime'] = period_data['u'] - period_data['u_mean']
            # If u is in the DataFrame, get the mean and fluctuating quantities
            if 'v' in period_data.columns:
                period_data['v_mean'] = period_data['v'].mean()
                # Calculate fluctuating velocity quantity
                period_data['v_prime'] = period_data['v'] - period_data['v_mean']
            dfs.append(period_data)
        '''
        # Apply parallel processing (about 30x speedup)
        dfs_ = applyParallel(working_data_grouped, turb_vars_primary)
        dfs.append(dfs_)
        
    df = pd.concat(dfs).sort_values('time')
    dfs_ = []
    print('Checkpoint 5a. {0:.4f} s'.format(time.time() - benchmark))
    
    for group, height in df.groupby('height'):
        '''
        for subgroup, subdata in height.groupby(pd.Grouper(key='time', freq='30T')):
            # Turbulent kinetic energy
            e = (subdata['u_prime']**2 + subdata['v_prime']**2 + subdata['w_prime']**2)/2
            # Variances
            var_u = subdata['u_prime']**2
            var_v = subdata['v_prime']**2
            var_w = subdata['w_prime']**2
            # Covariances
            cov_u_w = (subdata['u_prime']**2)*(subdata['w_prime']**2)
            cov_u_e = subdata['u_prime']*e
            cov_w_e = subdata['w_prime']*e
            
            # Append to the DataFrame
            subdata['e'] = e
            subdata['var_u'], subdata['var_v'], subdata['var_w'] = var_u, var_v, var_w
            subdata['cov_u_w'], subdata['cov_u_e'], subdata['cov_w_e'] = cov_u_w, cov_u_e, cov_w_e
            
            mean = subdata.mean().to_frame().T
            mean['time'], mean['height'] = subgroup, group
        '''
        
        # Apply parallel processing (about 7x speedup)
        height_grouped = height.groupby(pd.Grouper(key='time', freq='30T'))
        mean = applyParallel(height_grouped, turb_vars_secondary)
        mean['height'] = group
            
        dfs_.append(mean)
    print('Checkpoint 5b. {0:.4f} s'.format(time.time() - benchmark))
    
    df = pd.concat(dfs_)
    df = df.assign(site=site)
    df = df.set_index(['time', 'height', 'site'])
    
    return df.to_xarray() 

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
    # Filter data below 3000m to reduce outgoing file size
    data = data.where(data['height'] <= 3000, drop=True)
    
    return data

def parallel_check(site, date):
    ''' Method to inspect differences between datasets processed using serial and parallel methods. Date must be in YYYY-mm-dd format. '''
    directory = '/Volumes/UBL Data/data/storage/lidar'
    # Define file paths. Assume only one match per file type.
    path_serial = [os.path.join(directory, file) for file in os.listdir(directory) if (site in file) and (date in file) and ('parallel' not in file) and ('DBS' not in file)][0]
    path_parallel = [os.path.join(directory, file) for file in os.listdir(directory) if (site in file) and (date in file) and ('parallel' in file) and ('DBS' not in file)][0]
    # Load serial dataset
    serial = xr.open_dataset(path_serial)
    # Load parallel dataset
    parallel = xr.open_dataset(path_parallel)
    # Check differences. Note that np.inf or -np.inf may be possible for smaller values.
    for var_ in list(serial.data_vars):
        max_err_mag = np.nanmax(np.abs((serial[var_].sel(site=site) - parallel[var_].sel(site=site))))
        max_err_pct = 100*np.nanmax((serial[var_].sel(site=site) - parallel[var_].sel(site=site))/serial[var_].sel(site=site))
        print("Variable '{0}' had a maximum error of {1:.3f} and {2:.2f}%".format(var_, max_err_mag, max_err_pct))

if __name__ == '__main__':
    # Boolean control to handle if spectral analysis is generated
    spectral = True
    if spectral:
        dates = pd.date_range(start='2021-08-03', end='2021-08-31', freq='1D', closed='left')
        directory = '/Volumes/UBL Data/data/storage/lidar'
        for i in range(0, len(dates)-1):
            date_range = [dates[i], dates[i+1]]      
            print('Processing: ', date_range)
            scan_type = 'DBS'
            sites = ['BRON']
            for site in sites:
                start = time.time()
                data = processor(date_range, spectral_analysis=True, sites=site, scan_type=scan_type)
                start = time.time()
                if len(data) != 0:
                    data = quality_filter(data)
                    data = detrend(data, site)
                    filename = 'lidar_data_{0:02d}-{1:02d}-{2:02d}_spectral_{3}_turb.nc'.format(date_range[-1].date().year, date_range[-1].date().month, date_range[-1].date().day, site)
                    data.to_netcdf(os.path.join(directory, filename))
                    print(time.time()-start)
    else:
        dates = pd.date_range(start='2021-05-31', end='2021-07-01', freq='M', closed='left')
        directory = '/Volumes/UBL Data/data/storage/lidar'
        for i in range(0, len(dates)-1):
            date_range = [dates[i], dates[i+1]]      
            print('Processing: ', date_range)
            data = processor(date_range, spectral_analysis=False, sites=['BRON'])
            if len(data) != 0:
                data = quality_filter(data)
                filename = 'lidar_data_{0}-{1:02d}.nc'.format(date_range[-1].date().year, date_range[-1].date().month)
                data.to_netcdf(os.path.join(directory, filename))
