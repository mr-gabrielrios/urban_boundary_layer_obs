"""
Urban Boundary Layer Observation Data Processing
Script name:    ASOS Processing
Path:           ~/bin/asos.py
Description:    Use local ASOS data to identify heat wave events in New York City.
"""

import datetime, numpy as np, os, pandas as pd

def processor(dirpath):
    '''
    Read .csv data downloaded from the NCEI Climate database.

    Parameters
    ----------
    dirpath : str
        String with absolute path to folder containing relevant .csv files.

    Returns
    -------
    data : Pandas DataFrame
        DataFrame containing daily maximum and minimum temperatures from 1950 to 2020.

    '''
    
    # Initialize empty list of DataFrames
    data = []
    # Iterate through each file and append DataFrame to list
    for file in os.listdir(dirpath):
        fname = os.path.join(dirpath, file)
        df = pd.read_csv(fname)
        data.append(df)
    # Concatenate all DataFrames to each other
    data = pd.concat(data)
    # Convert to timestamps and set to index
    data['DATE'] = pd.to_datetime(data['DATE'])
    data = data.set_index('DATE', drop=False)
    # Calculate average temperature
    data['TAVG'] = (data['TMAX'] - data['TMIN'])/2
    
    return data

def heat_wave_finder(data):
    '''
    Finds heat wave events, as defined by the National Weather Service.
    Reference: https://www.weather.gov/okx/excessiveheat

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing daily maximum and minimum temperatures from 1950 to 2020.

    Returns
    -------
    heat_wave_days : list
        List of days classified as days under heat wave conditions.

    '''
    
    # Identify heat wave candidates based on temperature threshold
    candidates = data.where(data['TMAX'] >= 90).dropna()
    # Filter out data with non-consecutive days meeting threshold
    candidates = candidates.where((np.abs(candidates['DATE'].diff(-1)) == datetime.timedelta(days=1)) | (np.abs(candidates['DATE'].diff(1)) == datetime.timedelta(days=1))).dropna()
    # Get boolean mask to filter out data not meeting this threshold
    heat_wave_event_bool = ((candidates['DATE'].diff(-1) == -datetime.timedelta(days=1)) | ((candidates['DATE'].diff(1) == datetime.timedelta(days=1))))
    # Get list of data meeting this threshold
    dates = [pd.to_datetime(index) for index, row in candidates[heat_wave_event_bool].iterrows()]
    # Identify 3 consecutive days using boolean logic
    flag = [False]*len(dates)
    for i in range(1, len(dates)-1):
        if ((dates[i] - dates[i-1]) == datetime.timedelta(1)) and ((dates[i+1] - dates[i]) == datetime.timedelta(1)):
            flag[i-1] = True
            flag[i] = True
            flag[i+1] = True
    # Convert all days to datetimes
    heat_wave_days = pd.to_datetime(list(candidates[flag]['DATE']))
    
    return heat_wave_days

def main(date_range, dirpath='/Volumes/UBL Data/data/ncei'):
    # Read in data from directory
    data = processor(dirpath)
    # Filter out data beyond date range
    heat_wave_days = [i.date() for i in heat_wave_finder(data) if date_range[0] <= i.date() <= date_range[-1]] 
    
    return heat_wave_days
    
if __name__ == '__main__':
    date_range = pd.date_range(start='2017-01-01', end='2021-07-31', freq='D')
    heat_wave_days = main(date_range)