"""
Urban Boundary Layer Observation Data Processing
Script name:    Derived Functions
Path:           ~/bin/functions.py
Description:    This script contains functions to derive certain meteorological variables for analysis of the urban boundary layer.
"""

# Imports
import datetime, netCDF4 as nc, numpy as np, os, pandas as pd, time, xarray as xr

def gamma(data):
    '''
    Calculate lapse rate in K km^-1.

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset with all information from input data sources.

    Returns
    -------
    gamma : xArray DataArray
        xArray Dataset containing a new data variable for lapse rate.

    '''
    
    # Calculate lapse rate in (K km^-1) from 50m up. Convert height dimensions (m) to (km)
    gamma = -data['temperature'].diff(dim='time') / (data['height'].diff(dim='height') / 1000)
    # Prepand empty lapse rate entry for 0m full of nans
    temp = gamma.sel(height=50)
    temp['height'] = 0
    temp.values = np.full(temp.values.shape, np.nan)
    gamma = xr.concat([gamma, temp], dim='height').sortby('height')
    data = data.assign(lapse_rate=gamma)
    
    return data

def pressure(data):
    '''
    

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset containing all observed data AND lapse rate data.

    Returns
    -------
    data : xArray Dataset
         xArray Dataset containing all observed data AND derived pressure estimates.

    '''
    
    # Copy 0m pressure data to 50m pressure data
    data['pressure'].loc[{'height': 50}] = data['pressure'].loc[{'height': 0}]
    
    # Initialize empty list of DataArrays to concatenate after iteration
    pressures = []
    # Iterate over all height values to calculate pressure at all levels.
    for i in range(0, len(data.height.values)):
        # Get current iteration height
        curr_ht = data.height.values[i]
        print(curr_ht)
        # Preload values into concatenated xArray Dataset
        if i < 2:
            temp = data['pressure'].loc[{'height': curr_ht}]
        # Derive pressure for heights above 100m
        # Current model: https://home.chpc.utah.edu/~hallar/Thermo/Lectures/Lecture6.pdf
        # Note: equation needs work to address weird values
        else:
            prev_ht = data.height.values[i-1]
            # Re-use "temp" to reduce computation time
            temp = temp * (data['temperature'].loc[{'height': curr_ht}]/data['temperature'].loc[{'height': prev_ht}])**(9.81/(287 * data['lapse_rate'].loc[{'height': prev_ht}]))
        # Adjust height of "temp" to current iteration height value
        temp['height'] = curr_ht
        # Append to list of DataArrays to concatenate
        pressures.append(temp)
    # Concatenate data and assign as a data variable to main Dataset
    data['pressure'] = xr.concat(pressures, dim='height')
    
    return data    
    
        