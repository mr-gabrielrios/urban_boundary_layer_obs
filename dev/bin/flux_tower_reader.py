'''
City College of New York - Marshak Flux Tower Data Reader
Objective: Calculate sensible heat flux (Q_H)
'''

from bs4 import BeautifulSoup
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def processor(date_range, data_type='flux', data_access='online'):
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

    Returns
    -------
    data : Pandas DataFrame
        Contains tabular data of flux tower for the given parameters.

    '''
    
    # Access data from NOAA-CREST database
    if data_access == 'online':
        # Initialize file list
        files = []
        # Define URL for flux tower files
        dirpath = 'https://datadb.noaacrest.org/public/ORSL/Archived_Loggernet/CCNY_Flux_Tower/'
        # Define range of years through which URLs will be accessed. +1 added to stop date since np.arange is stop-exclusive.
        years = np.arange(start=int(dates[0][0:4]), stop=int(dates[1][0:4])+1)
        # Iterate through years to access all relevant files
        for year in years:
            # Define year-specific URL
            url = os.path.join(dirpath, str(year))
            # Generate BeautifulSoup for this HTML page
            soup = BeautifulSoup(requests.get(url, verify=False).content, features='html.parser')
            # Get all file paths
            for img in soup.find_all('img', alt='[TXT]'):
                href = os.path.join(url, img.find_parent('a')['href'])
                print(href)
            
        
    '''
    # Read data into Pandas DataFrame.
    data = pd.read_csv(fpath, header=1, skiprows=[2, 3])
    # Convert pressure from kPa to hPa
    data['amb_press_mean'] = data['amb_press_mean'].astype('float')*10
    # Cast covariance data as floats
    data['Ts_Uz_cov'] = data['Ts_Uz_cov'].astype('float')
    # Reference pressure (hPa)
    p0 = 1000
    
    # Calculate turbulent sensible heat flux
    # Method reference: Stull, R. Introduction to Boundary Layer Meteorology, 1988.
    data['QH'] = data['amb_press_mean'] * data['Ts_Uz_cov']
    
    
    # Format data timestamps to datetime format
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP']) - datetime.timedelta(hours=5)
    
    # Format dates into datetime format.
    date_format = '%Y%m%d%H'
    if date_range and len(date_range) == 2:
        date_range = [datetime.datetime.strptime(date_range[0], date_format),
                      datetime.datetime.strptime(date_range[1], date_format)]
        print(date_range)
        # Filter data for timestamps within given date range
        mask = (data['TIMESTAMP'] > date_range[0]) & (data['TIMESTAMP'] <= date_range[1])
        data = data.loc[mask]
        
    return data
    '''

def plotter(data, param='QH', date_range=None):
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
    fig, ax = plt.subplots(dpi=144)
    im = sns.lineplot(data=data, x='TIMESTAMP', y=data[param], ax=ax)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')

def scintillometer_data(fpath, date_range):
    data = pd.read_csv(fpath, delimiter='\t')
    data.columns = ['date', 'time', 'Cn2', 'CT2', 'H', 'pressure', 'temp', 'humidity', 'error']
    
    data['timestamp'] = data['date'] + ' ' + data['time']
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.drop(columns=['date', 'time'])
    
    # Format dates into datetime format.
    date_format = '%Y%m%d%H'
    if date_range and len(date_range) == 2:
        date_range = [datetime.datetime.strptime(date_range[0], date_format),
                      datetime.datetime.strptime(date_range[1], date_format)]
        print(date_range)
        # Filter data for timestamps within given date range
        mask = (data['timestamp'] > date_range[0]) & (data['timestamp'] <= date_range[1])
        data = data.loc[mask]
        
    data = data.set_index('timestamp', drop=False)
    
    for col in data.columns:
        if col != 'timestamp':
            data[col] = data[col].str.split(' ').str[0]
            data[col] = data[col].astype(float)
    
    data = data.resample('30T').mean()
    
    return data

if __name__ == '__main__':
    # Temporary file path  
    fpath = r'/Users/gabriel/Documents/ufo_ccny/flux_tower/data/TOA5_10560.flux.dat'
    dates = ['2021080200', '2021081800']
    data = processor(dates, data_type='flux', data_access='online')