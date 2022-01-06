"""
Urban Boundary Layer Observation Data Processing
Script name:    Main
Path:           ~/bin/grouping.py
Description:    Contains function for various grouping methods.
"""

import datetime, numpy as np, os, pandas as pd, xarray as xr

def wind_direction(data):
    
    # Initialize empty dictionary to hold grouped data
    grouped_dict = {}
    # Subtract 360 from all data with wind direction > 315 degrees from north
    data['wind_direction'] = data.where(data['wind_direction'] <= 315, other=data['wind_direction']-360)['wind_direction']
    # Define group bins
    dirs = {'northerly': -45, 'easterly': 45, 'southerly': 135, 'westerly': 225, 'end': 315}
    # Group the Dataset by bins
    groups = data.groupby_bins('wind_direction', list(dirs.values()), labels=list(dirs.keys())[:-1])
    # Port the grouped data into a dictionary
    for key, value in groups:
        grouped_dict[key] = value
        # Add 360 to all data with wind direction < 0 degrees from north
        grouped_dict[key]['wind_direction'] = value.where(value['wind_direction'] < 0, value['wind_direction'] + 360)['wind_direction']
    
    grouped_dict = {key: value.unstack().sortby('time', 'height').transpose() for key, value in groups}
    
    # Calculate hourly grouped data
    hourly_dict = {key: value.groupby('time.hour').mean().transpose('site', 'height', 'hour') for key, value in grouped_dict.items()}
    
    return grouped_dict, hourly_dict