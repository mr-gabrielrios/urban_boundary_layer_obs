"""
Urban Boundary Layer Observation Data Processing
Script name:    Derived Functions
Path:           ~/bin/functions.py
Description:    This script contains functions to derive certain meteorological variables for analysis of the urban boundary layer.
"""

# Imports
import datetime, netCDF4 as nc, numpy as np, os, pandas as pd, time, xarray as xr

import bin.turbulence

def mean_horizontal_wind(data):
    '''
    Calculate mean horizontal wind given a zonal and meridional wind vector.

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset with all information from input data sources.

    Returns
    -------
    U : xArray DataArray
        xArray Dataset containing a new data variable for horizontal wind.

    '''
    
    if 'u' in data.data_vars and 'v' in data.data_vars:
        data = data.assign(U=np.sqrt(data['u']**2 + data['v']**2))
    
    return data

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

def saturation_vapor_pressure(data):
    
    T = data['temperature'] - 273.15
    es = np.exp(34.494 - 4924.99/(T + 237.1))/(T + 105)**1.57
    data['saturation_vapor_pressure'] = es/100
    
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
    
    # Assuming as isothermal atmosphere
    H = 287 * data['temperature'].sel(height=0) / 9.81
    data['pressure'] = data['pressure'].sel(height=0) * np.exp(-data.height/H)
    
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
    data : xArray Dataset
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
    q = data['mixing_ratio']/(1 + data['mixing_ratio'])
    # Calculate virtual potential temperature, using a mixing ratio of 0.622
    data['virtual_potential_temperature'] = data['potential_temperature'] * (1 + 0.61*q)
    # Assign units
    data['virtual_potential_temperature'].attrs = {'units': 'K'}
    
    return data

def mixing_ratio(data):
    '''
    Calculates mixing ratio for given Dataset.

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset containing all observed data, water vapor density data, and pressure data.

    Returns
    -------
    data : xArray Dataset
        xArray Dataset containing mixing ratio in addition to pre-existing variables.

    '''
    
    # Calculate partial pressure of water vapor in hPa 
    # e = (1/100) * data['relative_humidity'] * (0.61 * np.exp(17.27*(data['temperature']-273.15)/((data['temperature']-273.15)+237.3))) * 10
    # Gas constant, water vapor (J K^-1 kg^-1)
    R_v = 461.51
    # Calculate partial pressure of water vapor (Wallace and Hobbs, 2006, Section 3.1.1)
    e = (data['vapor_density']/1000 * R_v * data['temperature'])/100
    # Calculate virtual potential temperature, using a mixing ratio of 0.622
    data['mixing_ratio'] = 0.622*e/(data['pressure'] - e)
    
    return data


def specific_humidity(data):
    
    '''
    Calculates specific humidity for given Dataset.
    Derived from Wallace and Hobbs, 2nd ed., Equation 3.57a.

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset containing all observed data and mixing ratio data.

    Returns
    -------
    data : xArray Dataset
        xArray Dataset containing specific_humidity in addition to pre-existing variables.
        
    '''
    
    data['specific_humidity'] = data['mixing_ratio'] / (data['mixing_ratio'] + 1)
    
    return data
    
    
def bulk_richardson_number(data, mode='surface', heights=[100, 200]):
    
    g = 9.81
    if mode == 'surface':
        buoyancy_param = g/data['virtual_potential_temperature'].sel(height=0)
        num = data['virtual_potential_temperature'] - data['virtual_potential_temperature'].sel(height=0)
        den = data['u']**2 + data['v']**2
        ri = buoyancy_param * data.height * num/den
    else:
        buoyancy_param = g/data['virtual_potential_temperature']
        dz = data['height'].diff(dim='height', n=1)
        du = data['u'].diff(dim='height', n=1)
        dv = data['v'].diff(dim='height', n=1)
        num = dz * data['virtual_potential_temperature'].diff(dim='height', n=1)
        den = du**2 + dv**2
        ri = buoyancy_param * num/den
        
    data['ri'] = ri
    
    return data

def bulk_richardson_number_stability(data, heights=[100, 150]):
    
    ''' Mode to calculate UBL stability based on a differential Ri number in the lower-level of the UBL. '''
    
    g = 9.81
    h_max, h_min = max(heights), min(heights)
    buoyancy_param = 2*g/(data['virtual_potential_temperature'].sel(height=h_max) + data['virtual_potential_temperature'].sel(height=h_min))
    num = data['virtual_potential_temperature'].sel(height=200) - data['virtual_potential_temperature'].sel(height=h_min)
    du = data['u'].sel(height=h_max) - data['u'].sel(height=h_max)
    dv = data['v'].sel(height=h_max) - data['v'].sel(height=h_min)
    den = du**2 + dv**2
    ri = buoyancy_param * (h_max-h_min) * num/den
    ri = np.dstack([ri]*len(data.height))
        
    data['ri_stability'] = (['time', 'site', 'height'], ri)
    
    return data

def stability_parameter(data):
    
    windows = bin.spectral.rolling_window(data, data.time, 10, 30, overlap=0)
    
    groups = []
    for group in windows:
        # Height of Marshak eddy covariance system above ground level (m)
        z = 50
        
        '''
        w_mask = ~np.isnan(group['w'])
        T_mask = ~np.isnan(group['sonic_temperature'])
        w = group['w'].values[w_mask][T_mask]
        T = group['sonic_temperature'].values[w_mask][T_mask]
        '''
        
        # Calculate w'T' covariance
        wT = np.cov(group['w'], group['sonic_temperature'])[0, 1]
        # Calculate friction velocity (see Stull, Section 2.10)
        u_star = (np.cov(group['u'], group['w'])[0, 1]**2 + np.cov(group['v'], group['w'])[0, 1]**2)**(1/4)
        # Calculate zeta (see Stull, Section 5.7, Equation 5.7b)
        zeta = -0.4*z*9.81*(wT)/(group['sonic_temperature'].mean() * u_star**3)
        groups.append(zeta)
    
    return groups