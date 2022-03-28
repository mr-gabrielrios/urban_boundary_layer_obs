"""
Urban Boundary Layer Observation Data Processing
Script name:    Main
Path:           ~/main.py
Description:    Process and analyze observational atmospheric boundary layer data from multiple locations in New York City.
"""

# Library imports
import datetime, netCDF4 as nc, numpy as np, os, pandas as pd, pytz, random, sys, time, xarray as xr
# Import custom scripts
from bin import asos, flux_tower, functions, grouping, lidar, mesonet, mwr, pblh

def unify(dates, storage_dir=None):
    '''
    Get unified xArray Dataset with data synthesized from multiple sources.
    '''
    
    # Lidar data access. 
    # If data is in storage location, access that month's data (short runtime). Else, run it (long runtime).
    # Output variables: u, v, w, wind_direction, ci (all heights)
    if storage_dir and os.path.isdir(os.path.join(storage_dir, 'lidar')):
        lidar_data, files = [], []
        for date in sorted(dates):
            print(date)
            filename = 'lidar_data_{0}-{1:02d}.nc'.format(date.year, date.month)
            full_path = os.path.join(storage_dir, 'lidar', filename)
            if full_path not in files:
                if os.path.isfile(full_path):
                    temp = xr.open_dataset(full_path)
                    # Only select the date of interest from month-long data
                    date_list = [date, date + datetime.timedelta(days=1)]
                    date_list = [date.strftime('%Y-%m-%dT%H:%M:%S') for date in date_list]
                    temp = temp.sel(time=slice(date_list[0], date_list[1]))
                    # Drop unnecessary variables
                    temp = temp.drop(labels=['ci'])
                    lidar_data.append(temp)
        lidar_data = xr.merge(lidar_data)
    else:
        lidar_data = lidar.processor(dates, sites=['BRON', 'MANH', 'QUEE', 'STAT'])
        lidar_data = lidar.quality_filter(lidar_data)
    print('Lidar data accessed.')
    
    # Microwave radiometer data access. 
    # If data is in storage location, access that month's data (short runtime). Else, run it (long runtime).
    # Output variables: the rest
    if storage_dir and os.path.isdir(os.path.join(storage_dir, 'mwr')):
        mwr_data, files = [], []
        for date in dates:
            print(date)
            filename = 'mwr_data_{0}-{1:02d}.nc'.format(date.year, date.month)
            full_path = os.path.join(storage_dir, 'mwr', filename)
            if full_path not in files: 
                if os.path.isfile(full_path):
                    temp = xr.open_dataset(full_path)
                    # Only select the date of interest from month-long data
                    date_list = [date, date + datetime.timedelta(days=1)]
                    date_list = [date.strftime('%Y-%m-%dT%H:%M:%S') for date in date_list]
                    temp = temp.sel(time=slice(date_list[0], date_list[1]))
                    # Remove unnecessary data
                    temp = temp.drop(labels=['liquid',
                                             'cloud_base',
                                             'rain_flag',
                                             'relative_humidity_std',
                                             'vapor_density_std',
                                             'surface_pressure_std',
                                             'integrated_liquid_std',
                                             'liquid_std',
                                             'temperature_qc_std',
                                             'surface_temperature_std',
                                             'surface_relative_humidity_std',
                                             'cloud_base_std',
                                             'surface_qc_std',
                                             'relative_humidity_qc_std',
                                             'integrated_qc_std',
                                             'rain_flag_std',
                                             'liquid_qc_std',
                                             'integrated_vapor_std',
                                             'temperature_std',
                                             'ir_temperature_std',
                                             'vapor_density_qc_std'])
                    
                    mwr_data.append(temp)
        mwr_data = xr.merge(mwr_data)
    else:
        mwr_data = mwr.accessor(dates, data_access='local')
    print('Microwave radiometer data accessed.')
    
    # Ignore ts_data when not performing spectral analysis or Manhattan-based analysis
    '''
    ts_data = flux_tower.processor(date_range, data_type='ts_data', data_access='local', height=mwr_data.height.values, spectral_analysis=False)
    '''
    ts_data = None
    
    # Get common times to prevent duplicate time values and ensure time series consistency
    if ts_data:
        times = list(sorted(set(lidar_data.time.values) & 
                        set(mwr_data.time.values) & 
                        set(ts_data.time.values)))
        data = xr.merge([lidar_data, mwr_data, ts_data], fill_value=np.nan)
    else:
        print(lidar_data, mwr_data)
        times = list(sorted(set(lidar_data.time.values) & 
                        set(mwr_data.time.values)))  
        data = xr.merge([lidar_data, mwr_data], fill_value=np.nan)
    
    ''' Merge data '''
    dates = [pd.date_range(start=dates[i], end=dates[i+1], freq='5T') for i in range(0, len(dates)-1)]
    dates = [datetime for date in dates for datetime in date]
    data = data.sel(time=times)
    # Copy Bronx surface pressure to Manhattan
    data['pressure'].loc[{'site': 'MANH'}] = data['pressure'].sel(site='BRON').copy()
    
    print('Data merged!')
    
    # Only select lower 3km
    data = data.sel(height=slice(0, 3000))
    
    # Add linear interpolation to fill in missing heights
    for varname in data.data_vars: 
        data[varname] = data[varname].interpolate_na(dim='height', method='linear')
    print('Data interpolated!')
    
    # Filter out all dates not selected, as lidar and radiometer data comes continuously due to each file being month-long
    mask = data.time.dt.date.isin(dates).values
    times = np.where(mask, data.time.values, np.datetime64('NaT'))
    data = data.assign(time=times)
    data = data.where(~np.isnan(data.time), drop=True)
    
    return data

def time_adjust(data, timezone='America/New_York'):
    '''
    Adjust data time to local time.

    Parameters
    ----------
    data : xArray Dataset
    timezone : str
        String with pytz-compatible time zone. Lima chosen to avoid DST issues.

    Returns
    -------
    data : xArray Dataset

    '''
    
    times = [datetime.datetime.utcfromtimestamp(date.tolist()/1e9) for date in data.time.values]
    times = [date + datetime.timedelta(hours=-5) for date in times]
    data = data.assign(time=times)
    
    return data

def derived_parameters(data):
    
    data = functions.lapse_rate(data)
    data = functions.pressure(data)
    data = functions.mixing_ratio(data)
    data = functions.specific_humidity(data)
    data = functions.potential_temperature(data)
    data = functions.virtual_potential_temperature(data)
    data = functions.mean_horizontal_wind(data)
    data = functions.bulk_richardson_number(data)
    data = pblh.parcel(data)

    return data

def ts_data(data):
    # List files containing relevant data
    files = ['/Volumes/UBL Data/data/flux/MANH/TOA5_10560_s20210720_e20210802_flux.dat', 
             '/Volumes/UBL Data/data/flux/MANH/TOA5_10560_s20210802_e20210818_flux.dat']
    
    # Initialize list to hold DataFrames that will later be concatenated
    ts_data = []
    # Iterate through files to read the .dat files
    for file in files: 
        # Read in the .dat file. Keep 2nd row for column headings, skip the following rows (unnecessary metadata), replace all bad values with np.nan
        temp = pd.read_table(file, sep=',', header=[1], skiprows=[2, 3], na_values='NAN')
        ts_data.append(temp)
    # Concatenate the .dat file DataFrames
    ts_data = pd.concat(ts_data)
    # Adjust time index for daylight savings time (UTC-4)
    ts_data['TIMESTAMP'] = pd.to_datetime(ts_data['TIMESTAMP']) - datetime.timedelta(hours=4)
    # Define Marshak building height.
    # See https://data.cityofnewyork.us/Housing-Development/Building-Footprints/nqwf-w8eh for building height data
    ts_data['z'] = 50.6
    # Calculate Obukhov length. See Stull, 1988, Eq. 5.7c.
    ts_data['L'] = -ts_data['Ts_Avg'].to_numpy().flatten()*ts_data['u_star'].to_numpy().flatten()/(0.4*9.81*ts_data['Ts_Uz_cov'].to_numpy().flatten())
    # Calculate atmospheric stability parameter.
    ts_data['zeta'] = ts_data['z']/ts_data['L']
    # Isolate Manhattan data
    times = data['zeta'].sel(site='MANH')
    # Match lidar and flux tower data by timestamp
    times_, zetas_ = [], []
    # Iterate over each timestamp from the xArray Dataset
    for time in times.sel(height=0).to_dataframe().index.get_level_values('time'):
        # Append time to the list that will be sent to a DataFrame
        times_.append(time)
        # If the xArray timestamp is in the ts_data DataFrame, append.
        # Else, append np.nan
        if time in ts_data['TIMESTAMP'].unique():
            zetas_.append(ts_data.loc[ts_data['TIMESTAMP'] == time]['zeta'].values[0])
        else:
            zetas_.append(np.nan)
    # Create a container DataFrame
    df_ = pd.DataFrame(columns=['time', 'zeta'])
    df_['time'] = times_
    df_['zeta'] = zetas_
    # Repeat values along the height axis
    values = np.tile(df_['zeta'].to_numpy(), (len(data.height.values), 1))
    # Create DataArray
    data['zeta'].loc[{'site': 'MANH'}] = xr.DataArray(data = values,
                                                      dims = times.dims)
    
    return data

def sea_breeze(data):
    # Find data where mean horizontal wind at low-levels is below 5 m/s
    # See Miller et al. (2003), Section 3.5.2.2.
    
    # Generate list of days
    days = pd.to_datetime(data.time.values)
    days = days.normalize().unique()
    
    # Initialize list of data
    sea_breeze_data = []
    
    # Iterate over all sites and days. If the conditions of < 5 m/s is met, append.
    for site in data.site:
        for day in days:
            # Drop all nonconforming times
            temp = data.where(data.sel(site=site, 
                                       time=day.strftime('%Y-%m-%d'), 
                                       height=slice(0, 200))['U'].mean(axis=1) < 5, 
                              drop=True)
            # Use Staten Island surface wind direction filtering to catch potential sea breeze days
            # Rationale: Staten Island observation site is ~4 mi from Lower New York Bay, closer than all others.
            temp = temp.where((temp['wind_direction'].sel(site='STAT', height=slice(0, 200)).mean() >= 90) & (temp['wind_direction'].sel(site='STAT', height=slice(0, 200)).mean() <= 180), drop=True)
            # Append all non-empty Datasets
            if len(temp.time.values) != 0:
                sea_breeze_data.append(temp)
    # Concatenate all Datasets
    sea_breeze_data = xr.concat(sea_breeze_data, dim='time').sortby('time')
    # Remove duplicate values
    _, index = np.unique(sea_breeze_data.time, return_index=True)
    sea_breeze_data = sea_breeze_data.isel(time=index)
    
    return sea_breeze_data

def consolidate(dates):
    storage_dir = '/Volumes/UBL Data/data/storage'
    
    data = unify(dates, storage_dir=storage_dir)
    
    data = derived_parameters(data)
    
    data = mesonet.xr_merge(data, 'zeta')
    data = mesonet.xr_merge(data, 'H')
    data = mesonet.xr_merge(data, 'wind_direction')
    
    data = time_adjust(data)
    
    return data

def attrs(data):
    ''' Adds attributes to xArray Dataset. '''
    
    # Check if parameter is in Dataset
    params = ['temperature', 'potential_temperature', 'virtual_potential_temperature', 'mixing_ratio', 'U', 'u', 'v', 'w', 'relative_humidity', 'specific_humidity', 'wind_direction', 'pressure', 'pblh', 'H', 'vapor_density']
    
    units = {'temperature': 'K',
             'potential_temperature': 'K',
             'virtual_potential_temperature': 'K',
             'mixing_ratio': 'kg \ kg^{-1}',
             'specific_humidity': 'kg \ kg^{-1}',
             'U': 'm \ s^{-1}',
             'u': 'm \ s^{-1}',
             'v': 'm \ s^{-1}',
             'w': 'm \ s^{-1}',
             'relative_humidity': r'%',
             'wind_direction': 'degrees',
             'vapor_density': 'kg \ kg^{-1}',
             'pressure': 'hPa',
             'pblh': 'm',
             'H': ' W \ m^{-2}'}
    
    for param in data.data_vars:
        if param not in params:
            continue
        else:
            data[param].attrs =  {'units': units[param]}
            
    return data
    

def dst_filter(dates):
    '''
    Filter out dates that may fall within the range of a Daylight Savings Time day to prevent datetime errors.

    Parameters
    ----------
    dates : list of dates or DatetimeIndex values

    Returns
    -------
    dates : list of dates or DatetimeIndex values

    '''
    
    dates = pd.to_datetime(dates)
    
    # Filter out the second week in March
    dates = dates.where(~((dates.month == 3) & ((dates.day > 7) & (dates.day < 15))))
    # Filter out the first week in November
    dates = dates.where(~((dates.month == 11) & ((dates.day >= 1) & (dates.day < 8))))
    
    return dates

def ri_stability_export(data, data_type='normal'):
    '''
    This method exports bulk Richardson method stability data to a .dat file for turbulence grouping.
    '''
    
    # Get DataFrame with bulk Richardson stability data
    out = data[['pblh', 'ri_stability']].to_dataframe().reset_index()
    # Get time bounds
    min_time = datetime.datetime.strftime(pd.to_datetime(min(data['ri_stability'].time.values)), '%Y%m%d')
    max_time = datetime.datetime.strftime(pd.to_datetime(max(data['ri_stability'].time.values)), '%Y%m%d')
    # Generate file name
    fname = '{0}_s{1}_e{2}.dat'.format(data_type, min_time, max_time)
    # Generate file path
    fpath = os.path.join('/Users/gabriel/Documents/urban_boundary_layer_obs/dev/bin/assets', fname)
    # Save to local directory
    out.to_csv(fpath)

def main(dates, mode='auxiliary', date_selection='all', data=None):
    '''
    Manage the execution of scripts for integrated data management.

    Parameters
    ----------
    dates : list
        List of 2 strings with dates of format yyyy-mm-dd.
    mode : str, optional
        Mode of operation of the script. The default is 'auxiliary'.
        Modes are 'run' (full execution of script, long runtime) and 'auxiliary' (execution of supplementary scripts)
    date_selection: str, optional
        Mode of date selection of the script. The default is 'all'.
        Modes are 'all' (all days within the range are computed for) and 'select' only some days are chosen for analysis.
    '''
    
    ''' Run auxiliary functions to pre-existing data. '''
    if mode == 'auxiliary':
        normal_data, heat_wave_data = data
        # Add PBLH data variable to Datasets
        normal_data = functions.virtual_potential_temperature(normal_data)
        heat_wave_data = functions.virtual_potential_temperature(heat_wave_data)
        # Add PBLH data variable to Datasets
        normal_data = pblh.gradient(normal_data)
        heat_wave_data = pblh.gradient(heat_wave_data)
        # Add PBLH data variable to Datasets
        normal_data = functions.bulk_richardson_number(normal_data, mode='surface')
        heat_wave_data = functions.bulk_richardson_number(heat_wave_data, mode='surface')
        # Assign units to data
        normal_data = attrs(normal_data)
        heat_wave_data = attrs(heat_wave_data)
        # Assign stability classification
        normal_data = functions.bulk_richardson_number_stability(normal_data)
        heat_wave_data = functions.bulk_richardson_number_stability(heat_wave_data)
        
        ''' Filter out stability data by surface wind direction due to flow obstructions. '''
        # Only filter out Queens data
        normal_data['zeta'] = normal_data['zeta'].where((normal_data['wind_direction'].sel(site='QUEE', height=0) >= 180) & (normal_data['wind_direction'].sel(site='QUEE', height=0) < 360))
        heat_wave_data['zeta'] = heat_wave_data['zeta'].where((heat_wave_data['wind_direction'].sel(site='QUEE', height=0) >= 180) & (heat_wave_data['wind_direction'].sel(site='QUEE', height=0) < 360))
        
        # Remove standard deviations from selected fields. Only enable for unfiltered data
        sigma = 4
        for var in ['U']:
            normal_data[var] = normal_data[var].where(
                (normal_data[var] <= (normal_data[var].mean() + sigma*normal_data[var].std())) & 
                (normal_data[var] >= (normal_data[var].mean() - sigma*normal_data[var].std()))) 
            heat_wave_data[var] = heat_wave_data[var].where(
                (heat_wave_data[var] <= (heat_wave_data[var].mean() + sigma*heat_wave_data[var].std())) & 
                (heat_wave_data[var] >= (heat_wave_data[var].mean() - sigma*heat_wave_data[var].std()))) 
        
        ''' Generate sea breeze data. '''
        sea_breeze_data_normal = sea_breeze(normal_data)
        sea_breeze_data_hw = sea_breeze(heat_wave_data)
        sea_breeze_data = xr.merge([sea_breeze_data_normal, 
                                    sea_breeze_data_hw], 
                                   compat='override')
        
        # Filter out dates with sea breezes for heat wave dates
        times = list(set(list(heat_wave_data.time.values)) - 
                     set(list(sea_breeze_data_hw.time.values)))
        bools = [True if time in times else False for time in heat_wave_data.time.values]
        heat_wave_data_no_sb = heat_wave_data.where(heat_wave_data.time[bools])
        
        # Keep list of days that are ideal sea breeze case study dates.
        sea_breeze_days = ['2018-07-03',
                           '2018-09-04',
                           '2018-09-05',
                           '2020-07-20',
                           '2020-07-21',
                           '2020-08-01',
                           '2021-05-12',
                           '2021-06-02',
                           '2021-07-15',
                           '2021-08-24',
                           '2021-08-25']
        
        # Organize sea breeze data
        sea_breeze_datasets = [sea_breeze_data_normal, sea_breeze_data_hw, heat_wave_data_no_sb]
        
        return normal_data, heat_wave_data, sea_breeze_datasets

    # Run scripts to generate data
    elif mode == 'run':
        date_range = pd.date_range(start=dates[0], end=dates[-1], freq='D')
        # Obtain days pertaining to heat wave events
        heat_wave_dates = [date.to_pydatetime() for date in list(sorted(set(asos.main(date_range))))]
        
        ''' Define clear sky/precipitation day distribution. '''
        # Number of days
        number_of_days = 36
        # Share of clear sky days
        share = 1
        
        # Get clear sky and precipitation days.
        # Skip if these are loaded into memory already.
        if 'clear_sky_days' not in globals():
            clear_sky_days_raw, precip_days_raw = asos.sky_props_finder(date_range, locs=[(0, 0)])
            # Filter out DST dates
            clear_sky_days_raw = dst_filter(clear_sky_days_raw)
            # Pick random dates
            clear_sky_days = [clear_sky_days_raw[i] for i in 
                              random.sample(range(0, len(clear_sky_days_raw)),
                                            int(len(clear_sky_days_raw) * share))]
            # Filter out DST dates
            precip_days_raw = dst_filter(precip_days_raw)
            # Pick random dates
            precip_days = [precip_days_raw[i] for i in 
                              random.sample(range(0, len(precip_days_raw)),
                                            int(len(precip_days_raw) * (1-share)))]
        
        # Initialize final list of normal dates
        normal_dates = []
        # Populate preliminary list with set of these days
        if date_selection == 'select':
            normal_dates_ = pd.to_datetime(list(sorted(set(clear_sky_days).union(set(precip_days)))))
        else:
            normal_dates_ = date_range
        
        # Ensure days from selected months are presented
        years, months, samples = [[min(date_range).year, max(date_range).year], 
                                  [min(date_range).month, max(date_range).month+1],
                                  number_of_days // 12]
        months = range(5, 10) if (min(months) < 5 or max(months) > 10) else months
        
        # List of dates with low data quality
        low_quality_dates = [datetime.date(2018, 2, 15),
                              datetime.date(2018, 7, 22),
                              datetime.date(2018, 7, 7),
                              datetime.date(2018, 8, 1),
                              datetime.date(2018, 8, 2),
                              datetime.date(2018, 8, 3),
                              datetime.date(2019, 2, 13),
                              datetime.date(2019, 3, 3),
                              datetime.date(2019, 6, 4),
                              datetime.date(2020, 6, 9),
                              datetime.date(2021, 2, 4),
                              datetime.date(2021, 3, 2),
                              datetime.date(2021, 5, 29),
                              datetime.date(2021, 5, 30),
                              datetime.date(2021, 6, 15),
                              datetime.date(2021, 7, 7)]
            
        # Iterate over each year and each month and grab a date from each
        for year in years:
            for month in months:
                # Get dates specific to the current iterand month and year
                filter_ = normal_dates_.where(~((normal_dates_.year == year) & 
                                                (normal_dates_.month == month)))
                # Create boolean for dates that don't fit the criteri
                filter_bool = pd.isnull(filter_)
                dates = normal_dates_[~filter_bool]
                # If the list of dates isn't empty, pick random dates
                if list(dates):
                    if date_selection == 'all':
                        dates_ = list(dates)
                    else:
                        dates_ = random.sample(list(dates), samples)
                    # Remove low quality dates
                    [normal_dates.append(date) for date in dates_ if date not in low_quality_dates]
                else:
                    continue 
        
        normal_data = consolidate(normal_dates)
        heat_wave_data = consolidate(heat_wave_dates)
        # Assign zeta values for Manhattan data
        normal_data = ts_data(normal_data)
        heat_wave_data = ts_data(heat_wave_data)
        
    return normal_data, heat_wave_data, None
        

if __name__ == '__main__':
    
    dates = ['2021-08-01', '2021-08-31']
    
    if 'normal_data' in globals() and 'heat_wave_data' in globals():
        normal_data = normal_data
        heat_wave_data = heat_wave_data
        # Grab data
        normal_data, heat_wave_data, sea_breeze_datasets = main(dates, mode='auxiliary', data=[normal_data, heat_wave_data])
        # Export stability data
        ri_stability_export(normal_data, data_type='normal')
        ri_stability_export(heat_wave_data, data_type='hw')
    else:
        normal_data, heat_wave_data, sea_breeze_datasets = main(dates, mode='run', data=None)
        normal_data, heat_wave_data, sea_breeze_datasets = main(dates, mode='auxiliary', data=[normal_data, heat_wave_data])