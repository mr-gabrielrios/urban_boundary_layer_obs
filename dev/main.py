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
    values = np.tile(df_['zeta'].to_numpy(), (len(normal_data.height.values), 1))
    # Create DataArray
    data['zeta'].loc[{'site': 'MANH'}] = xr.DataArray(data = values,
                                                      dims = times.dims)
    
    return data

def sea_breeze(data):
    # Find data where mean horizontal wind at low-levels is below 3 m/s
    sea_breeze_data = data.where(data['U'].sel(height=slice(0, 1000)).mean(axis=2) < 3)
    return sea_breeze_data

def main(dates):
    storage_dir = '/Volumes/UBL Data/data/storage'
    
    data = unify(dates, storage_dir=storage_dir)
    
    data = derived_parameters(data)
    
    data = mesonet.xr_merge(data, 'zeta')
    data = mesonet.xr_merge(data, 'H')
    
    data = time_adjust(data)
    
    return data

def attrs(data):
    ''' Adds attributes to xArray Dataset. '''
    
    # Check if parameter is in Dataset
    params = ['temperature', 'potential_temperature', 'virtual_potential_temperature', 'mixing_ratio', 'U', 'u', 'v', 'w', 'relative_humidity', 'wind_direction', 'pressure', 'pblh']
    
    units = {'temperature': 'K',
             'potential_temperature': 'K',
             'virtual_potential_temperature': 'K',
             'mixing_ratio': 'kg \ kg^{-1}',
             'U': 'm \ s^{-1}',
             'u': 'm \ s^{-1}',
             'v': 'm \ s^{-1}',
             'w': 'm \ s^{-1}',
             'relative_humidity': '%',
             'wind_direction': 'degrees',
             'pressure': 'hPa',
             'pblh': 'm'}
    
    for param in data.data_vars:
        if param not in params:
            continue
        else:
            data[param].attrs =  {'units': units[param]}
            
    return data

def merge():
    '''
    Merge all the data from each year. 
    Note that the variables are pre-existing and must be preloaded.
    '''
    
    clear_sky_days = clear_sky_days_2018 + clear_sky_days_2019 + clear_sky_days_2020 + clear_sky_days_2021
    precip_days = precip_days_2018 + precip_days_2019 + precip_days_2020 + precip_days_2021
    
    normal_dates = normal_dates_2018 + normal_dates_2019 + normal_dates_2020 + normal_dates_2021
    normal_data = xr.merge([normal_data_2018, normal_data_2019, normal_data_2020, normal_data_2021])
    
    return clear_sky_days, precip_days, normal_dates, normal_data
    

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

if __name__ == '__main__':
    
    # Define boolean for loading data if needed. Should be False assuming pickle data is loaded.
    # If spectra_qty is true, data will be generated only for the dates given
    load_data = False
    
    dates = ['2018-06-01', '2021-08-31']
    # Obtain days pertaining to heat wave events
    date_range = pd.date_range(start=dates[0], end=dates[-1], freq='D')
    
    if load_data:
        # Obtain days pertaining to heat wave events
        date_range = pd.date_range(start=dates[0], end=dates[-1], freq='D')
        heat_wave_dates = [date.to_pydatetime() for date in list(sorted(set(asos.main(date_range))))]
        
        ''' Pick 75% clear sky days, 25% days with precipitation. '''
        
        number_of_days = 36
        # Share of clear sky days
        share = 0.75
        
        # Get clear sky and precipitation days
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
        normal_dates_ = pd.to_datetime(list(sorted(set(clear_sky_days).union(set(precip_days)))))
        # Ensure days from each month are presented
        years, months, samples = range(2018, 2022), range(1, 13), 2
        # Iterate over each year and each month and grab a date from each
        for year in years:
            for month in months:
                # Get dates specific to the current iterand month and year
                dates = normal_dates_.drop(normal_dates_.where(
                        ~((normal_dates_.year == year) & 
                          (normal_dates_.month == month))))
                # If the list of dates isn't empty, pick 2 random dates
                if list(dates):
                    dates_ = random.sample(list(dates), 2)
                    [normal_dates.append(date) for date in dates_]
                else:
                    continue
                
        # List of dates with low data quality
        low_quality_dates = [datetime.date(2018, 2, 15),
                             datetime.date(2018, 7, 22),
                             datetime.date(2018, 8, 1),
                             datetime.date(2018, 8, 2),
                             datetime.date(2019, 6, 4),
                             datetime.date(2021, 5, 29),
                             datetime.date(2021, 5, 30)]
        # Remove low quality dates from the list
        for date in low_quality_dates:
            if date in normal_dates:
                normal_dates.remove(date)
                
        # heat_wave_data = main(heat_wave_dates)
        normal_data = main(normal_dates)
        # Assign zeta values for Manhattan data
        normal_data = ts_data(normal_data)
        
    
    else:
        # Add PBLH data variable to Datasets
        normal_data = pblh.parcel(normal_data)
        heat_wave_data = pblh.parcel(heat_wave_data)
        # Assign units to data
        normal_data = attrs(normal_data)
        heat_wave_data = attrs(heat_wave_data)
        
        ''' Ceilometer data - inactive for now. '''
        # Load ceilometer data for microwave radiometer testing
        # Note: ceilometer data comes from MANH, valid from 2020-06-15 to 2021-12-31
        # ceilometer = pd.read_csv('/Volumes/UBL Data/data/storage/ceilometer_data_s20200615_e20211231.csv')
        # ceilometer['datetime'] = pd.to_datetime(ceilometer['datetime'])
        # ceilometer['datetime'] = ceilometer['datetime'] + datetime.timedelta(hours=-5)
        # ceilometer = ceilometer.drop(columns=['datetime.1'])
        # ceilometer.set_index('datetime')
        
        ''' Generate sea breeze data. '''
        sea_breeze_data_normal = sea_breeze(normal_data)
        sea_breeze_data_hw = sea_breeze(heat_wave_data)
        sea_breeze_data = xr.merge([sea_breeze_data_normal, sea_breeze_data_hw], compat='override')
        
        del dates, date_range, sea_breeze_data_normal, sea_breeze_data_hw