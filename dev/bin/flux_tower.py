"""
Urban Boundary Layer Observation Data Processing
Script name:    Marshak Flux Tower Reader
Path:           ~/bin/flux_tower.py
Description:    Process, analyze, and visualize data collected by the flux tower atop the Marshak Building at The City College of New York.
"""

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
import xarray as xr
import urllib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.captureWarnings(True)

def processor(date_range, data_type='flux', data_access='online', height=np.nan, spectral_analysis=False, directory=None):
    '''
    Reads raw data from flux or ts_data files from the ORSL Loggernet or from the data folder.

    Parameters
    ----------
    date_range : list   
        2-item list of dates in YYYYMMDDHH format.
    data_type : str, optional
        Lets user select which type of data file is being parsed. The default is 'flux'.
    data_access : str, optional
        Lets user select what the data source is. The defauly is 'online'.
        The URL corresponding to the online option is: https://datadb.noaacrest.org/public/ORSL/Archived_Loggernet/CCNY_Flux_Tower/
        The directory corresponding to the local option is: '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/flux/MANH'
    heights : NumPy array, optional
        A list of heights that will serve as a dimension for the output xArray. This will be flowed in from the microwave radiometer Dataset.

    Returns
    -------
    data : Pandas DataFrame
        Contains tabular data of flux tower for the given parameters.

    '''
    
    # Define range of dates in datetime format
    date_range = pd.date_range(start=date_range[0], end=date_range[-1], freq='S').to_pydatetime()
    # Initialize data list
    data_list = []
    # Access data from NOAA-CREST database
    if data_access == 'online':
        # Define URL for flux tower files
        dirpath = 'https://datadb.noaacrest.org/public/ORSL/Archived_Loggernet/CCNY_Flux_Tower/'
        # Define range of years through which URLs will be accessed. +1 added to stop date since np.arange is stop-exclusive.
        years = np.arange(start=date_range[0].year, stop=date_range[-1].year+1)
        # Iterate through years to access all relevant files
        for year in years:
            # Certificate access workaround
            ssl._create_default_https_context = ssl._create_unverified_context
            # Define year-specific URL
            url = os.path.join(dirpath, str(year))
            # Generate BeautifulSoup for this HTML page
            soup = BeautifulSoup(requests.get(url, verify=False).content, features='html.parser')
            # Get all file paths. Use the icon image as the relevant child tag.
            for img in soup.find_all('img', alt='[TXT]'):
                # Build path to relevant files
                href = os.path.join(url, img.find_parent('a')['href'])
                # Only grab files that match the desired data type
                if href.split('MFT')[-1].split('.')[0][1:] == data_type:
                    # Get full date string of each file
                    date = datetime.datetime.strptime(href.split('MFT')[0].split('/')[-1], '%Y%m%d')
                    # Ensure file reports data in selected date range:
                    if ((date >= date_range[0]) & (date < date_range[-1])) | (date.date() == date_range[0].date()):
                        
                        # Build header from pre-defined CSV
                        header = pd.read_csv('/Volumes/UBL Data/data/reference/{0}_header.csv'.format(data_type))
                        # Remove all quotation marks
                        header = [s.strip('"') for s in list(header.columns)] 
                        # Read in raw data
                        data = pd.read_csv(href)
                        # Re-assign headers
                        data.columns = header
                        # These 5 lines build a timestamp column from separate numeric columns
                        data['M'] = data['M'].astype(str).apply(lambda x: x.zfill(4))
                        data['H'] = data['M'].str[0:2]
                        data['M'] = data['M'].str[2:]
                        data = data.apply(pd.to_numeric)
                        data['timestamp'] =  data['Y']*1000000000+data['DOY']*1000000+data['H']*10000+data['M']*100+data['S']
                        if data_type == 'ts_data':
                            data['time'] = data['timestamp'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%j%H%M%S.%f'))
                        else:
                            data['time'] = data['timestamp'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%j%H%M%S'))
                        # Remove datetime columns
                        data = data.drop(columns=['Y', 'DOY', 'H', 'M', 'S', 'timestamp'])
                        # Re-index to allow for future data manipulation
                        data = data.set_index('time', drop=False)
                        # Append to list of DataFrames
                        data_list.append(data)
                        # If no data exists for this time, return nan. Else, process it and calculate H.
        if not data_list:
            print('No data was found during the dates specified.')
            data = np.nan
            return data
        else:
            return data
    # Only Manhattan data can be accessed
    else:
        if not directory:
            directory = '/Volumes/UBL Data/data/flux/MANH'
        # Ignore flux data and just use ts_data.
        # 30-min 'flux' data is somewhat useless, since relevant parameters can be calculated from 1 Hz meteorological data.
        # Build header from pre-defined CSV
        header = pd.read_csv('/Volumes/UBL Data/data/reference/{0}_header.csv'.format(data_type))
        # Remove all quotation marks
        header = [s.strip('"') for s in list(header.columns)] 
        # The second condition prevents working files from being included in this list
        data_list = [os.path.join(directory, file) for file in os.listdir(directory) if ('MFT_ts_data' in file) and ('._' not in file)]
        # Filter by date
        data_list = [pd.read_csv(file, names=header) for file in data_list 
                     if (date_range[0] <= 
                         datetime.datetime.strptime(file.split('/')[-1][0:8], '%Y%m%d') < date_range[-1])]
        
        # Boolean that is true when data from 2021-07-31 to 2021-08-10 is read
        dat_file = False
        if not data_list:
            # Supplement the missing dates with consolidated flux data
            try:
                header = ['TIMESTAMP', 'RECORD', 'Ux', 'Uy', 
                          'Uz', 'Ts', 'diag_sonic', 'CO2', 
                          'H2O', 'diag_irga', 'amb_tmpr', 'amb_press', 
                          'CO2_sig_strgth', 'H2O_sig_strgth']
                # Get all files in the directory
                data_list = [os.path.join(directory, file) 
                             for file in os.listdir(directory)
                             if file.split('.')[-1] == 'dat' and
                             'ts_data' in file.split('.')[-2]]
                # Filter by date
                data_list = [pd.read_csv(file, 
                                         names=header, 
                                         header=1, 
                                         skiprows=[2, 3], 
                                         na_values=['NAN', -9999]) 
                             for file in data_list]
                # Rename columns
                data = pd.concat(data_list)
                data = data.rename(columns={'RECORD': 'Record', 'TIMESTAMP': 'time'})
                # Reformat time string
                data['time'] = pd.to_datetime(data['time'])
                # Switch Boolean
                dat_file = True
            except:
                return None
        
        # This conditional block handles data outside of the period
        # 2021-07-31 to 2021-08-11
        if not dat_file:
            # Concatenate all DataFrames into single DataFrame
            data = pd.concat(data_list)
            # These 5 lines build a timestamp column from separate numeric columns
            data['M'] = data['M'].astype(str).apply(lambda x: x.zfill(4))
            data['H'] = data['M'].str[0:2]
            data['M'] = data['M'].str[2:]
            data = data.apply(pd.to_numeric)
            data['timestamp'] =  data['Y']*1000000000+data['DOY']*1000000+data['H']*10000+data['M']*100+data['S']
            if data_type == 'ts_data':
                data['time'] = data['timestamp'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%j%H%M%S.%f'))
            else:
                data['time'] = data['timestamp'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%j%H%M%S'))
            # Remove datetime columns
            data = data.drop(columns=['Y', 'DOY', 'H', 'M', 'S', 'timestamp'])
       
        # Re-index to allow for future data manipulation
        data = data.set_index('time', drop=False)
        # Replace all invalid data with nans
        data = data.replace(-7999, np.nan)
        # Drop irrelevant data
        data = data.drop(columns=['Record', 'diag_sonic', 'CO2', 'H2O', 'diag_irga', 'amb_tmpr', 'amb_press', 'CO2_sig_strgth', 'H2O_sig_strgth'])
        # Rename data columns to match NYS Mesonet convention
        data = data.rename(columns={'Ux': 'u', 'Uy': 'v', 'Uz': 'w', 'Ts': 'sonic_temperature'})
        # Clip data according to date range
        data = data[(data.index >= date_range[0]) & (data.index < date_range[-1])]
        # Convert Dataframe to xArray Dataset
        data = xr.Dataset.from_dataframe(data)
        # Add height dimension to the Dataset
        data = data.assign_coords(height=height).expand_dims('height')
        # Add site information to Dataset
        data = data.assign_coords(site='MANH').expand_dims('site')
        
        return data

def plotter(data, param='H', date_range=None):
    '''
    This function plots data for a user-defined parameter between given dates.

    Parameters
    ----------
    data : Pandas DataFrame
        Contains DataFrame of converted flux tower data.
    param : string, optional
        This lets you choose what parameter to plot against time. The default is 'QH'.
    date_range : 2-item list of datetimes
        Contains dates for plotting in YYYYMMDDHH format.

    Returns
    -------
    None.

    '''
    
    # Plot formatting
    fig, ax = plt.subplots(dpi=300)
    im = sns.lineplot(data=data, x='timestamp', y=data[param], ax=ax)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')


if __name__ == '__main__':
    # USER INPUT. Define directory where CCNY flux data is kept.
    directory = None
    # USER INPUT. Start date and end dates must be input in the YYYYMMDD format. Saves data to the 'ts_data' variable.
    ts_data = processor(['20210731', '20210811'], data_type='flux', data_access='local', height=0, directory=directory)
    
    # Boolean to trigger file saving line. 
    # Normally used for the period between 2021-07-31 to 2021-08-10
    save_file = False
    if save_file:
        dir_ = '/Volumes/UBL Data/data/flux/MANH'
        file = 'ts_data_s202107310000_e202108110000_r202201180830.nc'
        ts_data.to_netcdf(os.path.join(dir_, file))
    