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
                    temp = temp.drop(labels=['ci', 'u_std', 'v_std','w_std',
                                             'wind_direction_std',
                                             'ci_std',
                                             'liquid',
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
    
    # Merge data
    dates = [pd.date_range(start=dates[i], end=dates[i+1], freq='5T') for i in range(0, len(dates)-1)]
    dates = [datetime for date in dates for datetime in date]
    data = data.sel(time=times)
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
    data = pblh.gradient(normal_data)

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
    
    # data = time_adjust(data)
    
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

if __name__ == '__main__':
    
    # Define boolean for loading data if needed. Should be False assuming pickle data is loaded.
    # If spectra_qty is true, data will be generated only for the dates given
    load_data = False
    if load_data:
        runtime = time.time() 
        
        dates = ['2021-06-01', '2021-08-31']
        # Obtain days pertaining to heat wave events
        date_range = pd.date_range(start=dates[0], end=dates[-1], freq='D')
        heat_wave_dates = [date.to_pydatetime() for date in list(sorted(set(asos.main(date_range))))]
        
        normal_dates = random.sample([date.to_pydatetime() for date in date_range if date not in heat_wave_dates and date.date().month not in [11, 12, 1, 2, 3]], len(heat_wave_dates))
        
        heat_wave_data = main(heat_wave_dates)
        normal_data = main(normal_dates)

        print(time.time() - runtime)
        
        
        # Days where all ASOS sites (KEWR, KJFK, KLGA) report >50% clear days. Obtained using the ASOS script (bin/asos.py)
        clear_sky_days = ['2018-01-09',
                          '2018-01-18',
                            '2018-01-19',
                            '2018-01-20',
                            '2018-01-25',
                            '2018-01-26',
                            '2018-01-27',
                            '2018-01-31',
                            '2018-02-03',
                            '2018-02-27',
                            '2018-03-06',
                            '2018-03-17',
                            '2018-03-18',
                            '2018-03-19',
                            '2018-03-26',
                            '2018-03-31',
                            '2018-04-13',
                            '2018-04-14',
                            '2018-04-20',
                            '2018-04-21',
                            '2018-04-22',
                            '2018-04-23',
                            '2018-05-01',
                            '2018-05-02',
                            '2018-05-08',
                            '2018-05-09',
                            '2018-05-21',
                            '2018-05-24',
                            '2018-05-25',
                            '2018-06-12',
                            '2018-06-16',
                            '2018-07-01',
                            '2018-07-02',
                            '2018-07-07',
                            '2018-07-08',
                            '2018-07-09',
                            '2018-07-10',
                            '2018-07-13',
                            '2018-07-16',
                            '2018-07-19',
                            '2018-07-20',
                            '2018-07-30',
                            '2018-08-05',
                            '2018-08-06',
                            '2018-08-24',
                            '2018-08-25',
                            '2018-08-27',
                            '2018-08-28',
                            '2018-08-29',
                            '2018-09-30',
                            '2018-10-16',
                            '2018-10-18',
                            '2018-10-19',
                            '2018-10-25',
                            '2018-10-30',
                            '2018-11-04',
                            '2018-11-07',
                            '2018-11-08',
                            '2018-11-11',
                            '2018-11-12',
                            '2018-12-05',
                            '2018-12-09',
                            '2018-12-10',
                            '2018-12-11',
                            '2018-12-18',
                            '2018-12-19',
                            '2019-01-04',
                            '2019-01-13',
                            '2019-01-14',
                            '2019-01-15',
                            '2019-02-03',
                            '2019-02-04',
                            '2019-02-16',
                            '2019-02-19',
                            '2019-02-22',
                            '2019-02-28',
                            '2019-03-05',
                            '2019-03-13',
                            '2019-03-17',
                            '2019-03-19',
                            '2019-03-20',
                            '2019-03-23',
                            '2019-03-24',
                            '2019-03-26',
                            '2019-03-27',
                            '2019-03-28',
                            '2019-04-01',
                            '2019-04-03',
                            '2019-04-04',
                            '2019-04-24',
                            '2019-05-18',
                            '2019-05-27',
                            '2019-06-01',
                            '2019-06-04',
                            '2019-06-08',
                            '2019-06-09',
                            '2019-06-12',
                            '2019-06-23',
                            '2019-06-24',
                            '2019-07-01',
                            '2019-07-09',
                            '2019-07-13',
                            '2019-07-14',
                            '2019-07-15',
                            '2019-07-16',
                            '2019-07-26',
                            '2019-07-30',
                            '2019-08-05',
                            '2019-08-12',
                            '2019-08-29',
                            '2019-08-30',
                            '2019-08-31',
                            '2019-09-03',
                            '2019-09-17',
                            '2019-09-19',
                            '2019-09-20',
                            '2019-09-21',
                            '2019-09-25',
                            '2019-09-27',
                            '2019-09-29',
                            '2019-10-14',
                            '2019-10-19',
                            '2019-10-23',
                            '2019-10-24',
                            '2019-10-25',
                            '2019-11-01',
                            '2019-11-02',
                            '2019-11-03',
                            '2019-11-04',
                            '2019-11-06',
                            '2019-11-11',
                            '2019-11-13',
                            '2019-11-14',
                            '2019-11-15',
                            '2019-11-16',
                            '2019-11-25',
                            '2019-11-26',
                            '2019-11-30',
                            '2019-12-07',
                            '2019-12-12',
                            '2019-12-20',
                            '2019-12-22',
                            '2019-12-23',
                            '2019-12-24',
                            '2020-01-02',
                            '2020-01-09',
                            '2020-01-20',
                            '2020-01-21',
                            '2020-01-22',
                            '2020-01-23',
                            '2020-01-24',
                            '2020-01-30',
                            '2020-02-14',
                            '2020-02-21',
                            '2020-02-22',
                            '2020-02-23',
                            '2020-03-07',
                            '2020-03-18',
                            '2020-03-26',
                            '2020-04-11',
                            '2020-04-19',
                            '2020-04-25',
                            '2020-04-28',
                            '2020-05-02',
                            '2020-05-05',
                            '2020-05-10',
                            '2020-05-13',
                            '2020-05-19',
                            '2020-05-20',
                            '2020-05-31',
                            '2020-06-08',
                            '2020-06-09',
                            '2020-06-10',
                            '2020-06-14',
                            '2020-06-15',
                            '2020-06-16',
                            '2020-06-22',
                            '2020-06-23',
                            '2020-07-09',
                            '2020-07-12',
                            '2020-07-20',
                            '2020-07-21',
                            '2020-07-27',
                            '2020-08-03',
                            '2020-08-10',
                            '2020-08-20',
                            '2020-08-21',
                            '2020-08-22',
                            '2020-08-23',
                            '2020-08-24',
                            '2020-09-04',
                            '2020-09-05',
                            '2020-09-06',
                            '2020-09-15',
                            '2020-09-19',
                            '2020-09-20',
                            '2020-09-21',
                            '2020-09-22',
                            '2020-09-23',
                            '2020-09-24',
                            '2020-09-25',
                            '2020-10-03',
                            '2020-10-04',
                            '2020-10-08',
                            '2020-10-09',
                            '2020-10-14',
                            '2020-10-17',
                            '2020-11-04',
                            '2020-11-05',
                            '2020-11-09',
                            '2020-11-10',
                            '2020-11-14',
                            '2020-11-16',
                            '2020-11-19',
                            '2020-11-20',
                            '2020-11-29',
                            '2020-12-03',
                            '2020-12-06',
                            '2020-12-07',
                            '2020-12-11',
                            '2020-12-15',
                            '2020-12-23',
                            '2020-12-29',
                            '2020-12-30',
                            '2021-01-07',
                            '2021-01-08',
                            '2021-01-10',
                            '2021-01-11',
                            '2021-01-12',
                            '2021-01-21',
                            '2021-01-25',
                            '2021-02-04',
                            '2021-02-17',
                            '2021-02-21',
                            '2021-02-24',
                            '2021-02-25',
                            '2021-02-26',
                            '2021-03-02',
                            '2021-03-03',
                            '2021-03-05',
                            '2021-03-07',
                            '2021-03-08',
                            '2021-03-09',
                            '2021-03-10',
                            '2021-03-13',
                            '2021-03-15',
                            '2021-03-19',
                            '2021-03-20',
                            '2021-03-21',
                            '2021-03-22',
                            '2021-03-27',
                            '2021-03-30',
                            '2021-04-03',
                            '2021-04-05',
                            '2021-04-06',
                            '2021-04-08',
                            '2021-04-20',
                            '2021-04-27',
                            '2021-04-28',
                            '2021-05-06',
                            '2021-05-07',
                            '2021-05-14',
                            '2021-05-19',
                            '2021-06-02',
                            '2021-06-06',
                            '2021-06-16',
                            '2021-06-17',
                            '2021-06-18',
                            '2021-06-28',
                            '2021-06-29',
                            '2021-07-07',
                            '2021-07-24',
                            '2021-07-27',
                            '2021-07-31',
                            '2021-08-06',
                            '2021-08-12',
                            '2021-08-13',
                            '2021-08-24',
                            '2021-08-25',
                            '2021-08-26',
                            '2021-09-07',
                            '2021-09-10',
                            '2021-09-11',
                            '2021-09-12',
                            '2021-09-19',
                            '2021-09-25',
                            '2021-09-26',
                            '2021-09-27']
    
    else:
        # Remove anomalous data from Staten Island on 2021-05-29 to 2021-05-30
        normal_data = normal_data.where(~((normal_data.time.dt.month == 6) & (normal_data.time.dt.day == 4) & (normal_data.time.dt.year == 2019)))
        # Add PBLH data variable to Datasets
        normal_data = pblh.parcel(normal_data)
        heat_wave_data = pblh.parcel(heat_wave_data)
        # Assign units to data
        normal_data = attrs(normal_data)
        heat_wave_data = attrs(heat_wave_data)
        
        # Load ceilometer data for microwave radiometer testing
        # Note: ceilometer data comes from MANH, valid from 2020-06-15 to 2021-12-31
        ceilometer = pd.read_csv('/Volumes/UBL Data/data/storage/ceilometer_data_s20200615_e20211231.csv')
        ceilometer['datetime'] = pd.to_datetime(ceilometer['datetime'])
        ceilometer['datetime'] = ceilometer['datetime'] + datetime.timedelta(hours=-5)
        ceilometer = ceilometer.drop(columns=['datetime.1'])
        ceilometer.set_index('datetime')
        
        # Generate sea breeze data
        sea_breeze_data_normal = sea_breeze(normal_data)
        sea_breeze_data_hw = sea_breeze(heat_wave_data)
        sea_breeze_data = xr.merge([sea_breeze_data_normal, sea_breeze_data_hw], compat='override')
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
                    temp = temp.drop(labels=['ci', 'u_std', 'v_std',
 'w_std',
 'wind_direction_std',
 'ci_std',
 'liquid',
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
    
    # Merge data
    dates = [pd.date_range(start=dates[i], end=dates[i+1], freq='5T') for i in range(0, len(dates)-1)]
    dates = [datetime for date in dates for datetime in date]
    data = data.sel(time=times)
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
    data = pblh.gradient(normal_data)

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
    
    # data = time_adjust(data)
    
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

if __name__ == '__main__':
    
    # Define boolean for loading data if needed. Should be False assuming pickle data is loaded.
    # If spectra_qty is true, data will be generated only for the dates given
    load_data = False
    if load_data:
        runtime = time.time() 
        
        dates = ['2021-06-01', '2021-08-31']
        # Obtain days pertaining to heat wave events
        date_range = pd.date_range(start=dates[0], end=dates[-1], freq='D')
        heat_wave_dates = [date.to_pydatetime() for date in list(sorted(set(asos.main(date_range))))]
        
        normal_dates = random.sample([date.to_pydatetime() for date in date_range if date not in heat_wave_dates and date.date().month not in [11, 12, 1, 2, 3]], len(heat_wave_dates))
        
        heat_wave_data = main(heat_wave_dates)
        normal_data = main(normal_dates)

        print(time.time() - runtime)
        
        
        # Days where all ASOS sites (KEWR, KJFK, KLGA) report >50% clear days. Obtained using the ASOS script (bin/asos.py)
        clear_sky_days = ['2018-01-09',
                          '2018-01-18',
                            '2018-01-19',
                            '2018-01-20',
                            '2018-01-25',
                            '2018-01-26',
                            '2018-01-27',
                            '2018-01-31',
                            '2018-02-03',
                            '2018-02-27',
                            '2018-03-06',
                            '2018-03-17',
                            '2018-03-18',
                            '2018-03-19',
                            '2018-03-26',
                            '2018-03-31',
                            '2018-04-13',
                            '2018-04-14',
                            '2018-04-20',
                            '2018-04-21',
                            '2018-04-22',
                            '2018-04-23',
                            '2018-05-01',
                            '2018-05-02',
                            '2018-05-08',
                            '2018-05-09',
                            '2018-05-21',
                            '2018-05-24',
                            '2018-05-25',
                            '2018-06-12',
                            '2018-06-16',
                            '2018-07-01',
                            '2018-07-02',
                            '2018-07-07',
                            '2018-07-08',
                            '2018-07-09',
                            '2018-07-10',
                            '2018-07-13',
                            '2018-07-16',
                            '2018-07-19',
                            '2018-07-20',
                            '2018-07-30',
                            '2018-08-05',
                            '2018-08-06',
                            '2018-08-24',
                            '2018-08-25',
                            '2018-08-27',
                            '2018-08-28',
                            '2018-08-29',
                            '2018-09-30',
                            '2018-10-16',
                            '2018-10-18',
                            '2018-10-19',
                            '2018-10-25',
                            '2018-10-30',
                            '2018-11-04',
                            '2018-11-07',
                            '2018-11-08',
                            '2018-11-11',
                            '2018-11-12',
                            '2018-12-05',
                            '2018-12-09',
                            '2018-12-10',
                            '2018-12-11',
                            '2018-12-18',
                            '2018-12-19',
                            '2019-01-04',
                            '2019-01-13',
                            '2019-01-14',
                            '2019-01-15',
                            '2019-02-03',
                            '2019-02-04',
                            '2019-02-16',
                            '2019-02-19',
                            '2019-02-22',
                            '2019-02-28',
                            '2019-03-05',
                            '2019-03-13',
                            '2019-03-17',
                            '2019-03-19',
                            '2019-03-20',
                            '2019-03-23',
                            '2019-03-24',
                            '2019-03-26',
                            '2019-03-27',
                            '2019-03-28',
                            '2019-04-01',
                            '2019-04-03',
                            '2019-04-04',
                            '2019-04-24',
                            '2019-05-18',
                            '2019-05-27',
                            '2019-06-01',
                            '2019-06-04',
                            '2019-06-08',
                            '2019-06-09',
                            '2019-06-12',
                            '2019-06-23',
                            '2019-06-24',
                            '2019-07-01',
                            '2019-07-09',
                            '2019-07-13',
                            '2019-07-14',
                            '2019-07-15',
                            '2019-07-16',
                            '2019-07-26',
                            '2019-07-30',
                            '2019-08-05',
                            '2019-08-12',
                            '2019-08-29',
                            '2019-08-30',
                            '2019-08-31',
                            '2019-09-03',
                            '2019-09-17',
                            '2019-09-19',
                            '2019-09-20',
                            '2019-09-21',
                            '2019-09-25',
                            '2019-09-27',
                            '2019-09-29',
                            '2019-10-14',
                            '2019-10-19',
                            '2019-10-23',
                            '2019-10-24',
                            '2019-10-25',
                            '2019-11-01',
                            '2019-11-02',
                            '2019-11-03',
                            '2019-11-04',
                            '2019-11-06',
                            '2019-11-11',
                            '2019-11-13',
                            '2019-11-14',
                            '2019-11-15',
                            '2019-11-16',
                            '2019-11-25',
                            '2019-11-26',
                            '2019-11-30',
                            '2019-12-07',
                            '2019-12-12',
                            '2019-12-20',
                            '2019-12-22',
                            '2019-12-23',
                            '2019-12-24',
                            '2020-01-02',
                            '2020-01-09',
                            '2020-01-20',
                            '2020-01-21',
                            '2020-01-22',
                            '2020-01-23',
                            '2020-01-24',
                            '2020-01-30',
                            '2020-02-14',
                            '2020-02-21',
                            '2020-02-22',
                            '2020-02-23',
                            '2020-03-07',
                            '2020-03-18',
                            '2020-03-26',
                            '2020-04-11',
                            '2020-04-19',
                            '2020-04-25',
                            '2020-04-28',
                            '2020-05-02',
                            '2020-05-05',
                            '2020-05-10',
                            '2020-05-13',
                            '2020-05-19',
                            '2020-05-20',
                            '2020-05-31',
                            '2020-06-08',
                            '2020-06-09',
                            '2020-06-10',
                            '2020-06-14',
                            '2020-06-15',
                            '2020-06-16',
                            '2020-06-22',
                            '2020-06-23',
                            '2020-07-09',
                            '2020-07-12',
                            '2020-07-20',
                            '2020-07-21',
                            '2020-07-27',
                            '2020-08-03',
                            '2020-08-10',
                            '2020-08-20',
                            '2020-08-21',
                            '2020-08-22',
                            '2020-08-23',
                            '2020-08-24',
                            '2020-09-04',
                            '2020-09-05',
                            '2020-09-06',
                            '2020-09-15',
                            '2020-09-19',
                            '2020-09-20',
                            '2020-09-21',
                            '2020-09-22',
                            '2020-09-23',
                            '2020-09-24',
                            '2020-09-25',
                            '2020-10-03',
                            '2020-10-04',
                            '2020-10-08',
                            '2020-10-09',
                            '2020-10-14',
                            '2020-10-17',
                            '2020-11-04',
                            '2020-11-05',
                            '2020-11-09',
                            '2020-11-10',
                            '2020-11-14',
                            '2020-11-16',
                            '2020-11-19',
                            '2020-11-20',
                            '2020-11-29',
                            '2020-12-03',
                            '2020-12-06',
                            '2020-12-07',
                            '2020-12-11',
                            '2020-12-15',
                            '2020-12-23',
                            '2020-12-29',
                            '2020-12-30',
                            '2021-01-07',
                            '2021-01-08',
                            '2021-01-10',
                            '2021-01-11',
                            '2021-01-12',
                            '2021-01-21',
                            '2021-01-25',
                            '2021-02-04',
                            '2021-02-17',
                            '2021-02-21',
                            '2021-02-24',
                            '2021-02-25',
                            '2021-02-26',
                            '2021-03-02',
                            '2021-03-03',
                            '2021-03-05',
                            '2021-03-07',
                            '2021-03-08',
                            '2021-03-09',
                            '2021-03-10',
                            '2021-03-13',
                            '2021-03-15',
                            '2021-03-19',
                            '2021-03-20',
                            '2021-03-21',
                            '2021-03-22',
                            '2021-03-27',
                            '2021-03-30',
                            '2021-04-03',
                            '2021-04-05',
                            '2021-04-06',
                            '2021-04-08',
                            '2021-04-20',
                            '2021-04-27',
                            '2021-04-28',
                            '2021-05-06',
                            '2021-05-07',
                            '2021-05-14',
                            '2021-05-19',
                            '2021-06-02',
                            '2021-06-06',
                            '2021-06-16',
                            '2021-06-17',
                            '2021-06-18',
                            '2021-06-28',
                            '2021-06-29',
                            '2021-07-07',
                            '2021-07-24',
                            '2021-07-27',
                            '2021-07-31',
                            '2021-08-06',
                            '2021-08-12',
                            '2021-08-13',
                            '2021-08-24',
                            '2021-08-25',
                            '2021-08-26',
                            '2021-09-07',
                            '2021-09-10',
                            '2021-09-11',
                            '2021-09-12',
                            '2021-09-19',
                            '2021-09-25',
                            '2021-09-26',
                            '2021-09-27']
    
    else:
        # Remove anomalous data from Staten Island on 2021-05-29 to 2021-05-30
        normal_data = normal_data.where(~((normal_data.time.dt.month == 6) & (normal_data.time.dt.day == 4) & (normal_data.time.dt.year == 2019)))
        # Add PBLH data variable to Datasets
        normal_data = pblh.parcel(normal_data)
        heat_wave_data = pblh.parcel(heat_wave_data)
        # Assign units to data
        normal_data = attrs(normal_data)
        heat_wave_data = attrs(heat_wave_data)
        
        # Load ceilometer data for microwave radiometer testing
        # Note: ceilometer data comes from MANH, valid from 2020-06-15 to 2021-12-31
        ceilometer = pd.read_csv('/Volumes/UBL Data/data/storage/ceilometer_data_s20200615_e20211231.csv')
        ceilometer['datetime'] = pd.to_datetime(ceilometer['datetime'])
        ceilometer['datetime'] = ceilometer['datetime'] + datetime.timedelta(hours=-5)
        ceilometer = ceilometer.drop(columns=['datetime.1'])
        ceilometer.set_index('datetime')
        
        # Generate sea breeze data
        sea_breeze_data_normal = sea_breeze(normal_data)
        sea_breeze_data_hw = sea_breeze(heat_wave_data)
        sea_breeze_data = xr.merge([sea_breeze_data_normal, sea_breeze_data_hw], compat='override')