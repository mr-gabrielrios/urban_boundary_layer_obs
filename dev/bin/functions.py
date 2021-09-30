"""
Urban Boundary Layer Observation Data Processing
Script name:    Derived Functions
Path:           ~/bin/functions.py
Description:    This script contains functions to derive certain meteorological variables for analysis of the urban boundary layer.
"""

# Imports
import datetime, netCDF4 as nc, numpy as np, os, pandas as pd, time, xarray as xr

def lapse_rate(data):
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
    gamma = data['temperature'].diff(dim='height') / (data['height'].diff(dim='height') / 1000)
    # Prepand empty lapse rate entry for 0m full of nans
    temp = gamma.sel(height=50)
    temp['height'] = 0
    temp.values = np.full(temp.values.shape, np.nan)
    gamma = xr.concat([gamma, temp], dim='height').sortby('height')
    # Assign to Dataset
    data = data.assign(lapse_rate=gamma)
    # Assign units
    data['lapse_rate'].attrs = {'units': 'K km^-1'}
    
    return data

def pressure(data):
    '''
    Calculate atmospheric pressure over all given times and heights.

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
        # Preload values into concatenated xArray Dataset
        if i < 2:
            temp = data['pressure'].loc[{'height': curr_ht}]
        # Derive pressure for heights above 100m
        # Current model: https://home.chpc.utah.edu/~hallar/Thermo/Lectures/Lecture6.pdf
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
    # Assign units
    data['pressure'].attrs = {'units': 'hPa'}
    
    return data    
    
def potential_temperature(data):
    '''
    Calculates potential temperature for given Dataset.

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset containing all observed data AND pressure data.

    Returns
    -------
    data : TYPE
        xArray Dataset containing potential temperature in addition to pre-existing variables.

    '''
    
    # Calculate potential temperature using surface pressure as p_0 term
    data['potential_temperature'] = data['temperature'] * (data['pressure'].sel(height=0)/data['pressure']) ** (287/1004)
    # Assign units
    data['potential_temperature'].attrs = {'units': 'K'}
    
    return data

def virtual_potential_temperature(data):
    '''
    Calculates virtual potential temperature for given Dataset.

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset containing all observed data AND pressure data.

    Returns
    -------
    data : TYPE
        xArray Dataset containing virtual potential temperature in addition to pre-existing variables.

    '''
    
    # Calculate partial pressure of water vapor in hPa
    e = (data['vapor_density']/1000 * 461.5 * data['temperature'])/100
    # Calculate virtual potential temperature, using a mixing ratio of 0.622
    data['virtual_potential_temperature'] = data['temperature']/(1 - (1-0.622)*e/data['pressure'])
    # Assign units
    data['virtual_potential_temperature'].attrs = {'units': 'K'}
    
    return data
    
    