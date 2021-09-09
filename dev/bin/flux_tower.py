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
import numpy as np
import os
import pandas as pd
import requests
import ssl
import urllib
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
    
    # Define range of dates in datetime format
    date_range = pd.date_range(start=date_range[0], end=date_range[1], freq='S').to_pydatetime()
    print(date_range)
    # Access data from NOAA-CREST database
    if data_access == 'online':
        # Initialize data list
        data_list = []
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
                    if (date >= date_range[0]) & (date < date_range[-1]):
                        print(href)
                        # Build header from pre-defined CSV
                        header = pd.read_csv('/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/flux/MANH/{0}_header.csv'.format(data_type))
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
                        data['timestamp'] = pd.to_datetime(data['Y']*1000000000+data['DOY']*1000000+data['H']*10000+data['M']*100+data['S'], format='%Y%j%H%M%S')
                        # Remove datetime columns
                        data = data.drop(columns=['Y', 'DOY', 'H', 'M', 'S'])
                        # Re-index to allow for future data manipulation
                        data = data.set_index('timestamp', drop=False)
                        # Append to list of DataFrames
                        data_list.append(data)
    # Concatenate all DataFrames into single DataFrame
    data = pd.concat(data_list)
    # Correct timestamp from UTC+5 to UTC
    data['timestamp'] = pd.to_datetime(data['timestamp']) - datetime.timedelta(hours=5)
    # Replace all invalid data with nans
    data = data.replace(-7999, np.nan)
    
    # Calculate sensible heat flux (H) from collected flux data
    if data_type == 'flux':
        # Convert pressure from kPa to hPa
        data['amb_press_mean'] = data['amb_press_mean'].astype('float')*10
        # Cast covariance data as floats
        data['Ts_Uz_cov'] = data['Ts_Uz_cov'].astype('float')
        # Reference pressure (hPa)
        p0 = 1000
        # Calculate turbulent sensible heat flux
        # Method reference: Stull, R. Introduction to Boundary Layer Meteorology, 1988.
        data['H'] = data['amb_press_mean'] * data['Ts_Uz_cov']
            
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
    # Temporary file path  
    fpath = r'/Users/gabriel/Documents/ufo_ccny/flux_tower/data/TOA5_10560.flux.dat'
    dates = ['2020-07-16 00:00:00', '2020-07-16 00:05:00']
    data_new = processor(dates, data_type='ts_data', data_access='online')