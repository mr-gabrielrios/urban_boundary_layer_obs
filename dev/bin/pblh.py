"""
Urban Boundary Layer Observation Data Processing
Script name:    Planetary Boundary Layer Height estimation
Path:           ~/bin/pblh.py
Description:    Estimate the planetary boundary layer height using various methods.
"""

import numpy as np, pandas as pd, scipy

# Note: all methods are derived from the methods listed in 
#       Seidel et al (2010), 10.1029/2009JD013680

def parcel(data):
    '''
    Parcel method for estimating mixed layer height, see Section 2.2.1.
    Objective: find height at which theta-v = theta-v at surface

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset containing microwave radiometer and lidar data.

    Returns
    -------
    data : xArray Dataset
        xArray Dataset with PBLH.

    '''
    
    # Note: offload numerical data from xArray format to save time
    # Load NumPy array with only theta_v values
    arr = data['virtual_potential_temperature'].values
    # Get NumPy array of heights
    heights = data.height.values
    # Initialize container NumPy array with shape of theta_v array to hold PBLH values
    mlh = np.empty(arr.shape)
    # Iterate over sites (axis = 1 in the NumPy array)
    for site in range(arr.shape[1]):
        # Iterate over timestamps (axis = 0 in the NumPy array)
        for time in range(arr.shape[0]):
            # Grab the surface theta_v value
            theta_v_0 = arr[time, site, 0]
            # Scan the theta_v values over all heights
            # If a value is found such that it is between two values, grab the floor. Else, 0.
            scan = np.array([heights[i] if arr[time, site, i] < theta_v_0 < arr[time, site, i+1] else 0 for i, value in enumerate(range(0, len(arr[time, site, :])-1))])
            # Filter out all zero values
            height = scan[np.nonzero(scan)]
            # If there are nonzero values, grab the minimum height and repeat it over all heights. Else, np.nan
            if height.size != 0:
                height = np.repeat(np.nanmin(height), arr.shape[2])
            else:
                height = np.repeat(np.nan, arr.shape[2])
            # Insert into the container NumPy array
            mlh[time, site] = height
    # Assign the container array into the Dataset
    data = data.assign(pblh=(['time', 'site', 'height'], mlh))
    
    return data

def gradient(data):
    '''
    Gradient method for estimating planetary boundary layer height, see Sections 2.2.2 and 4.
    Objective: find height at which d(theta_v)/dz is a minimum or maximum for a given quantity

    Currently using method described in Section 2.2.2 (maximum vertical gradient of theta).

    Parameters
    ----------
    data : xArray Dataset
        xArray Dataset containing microwave radiometer and lidar data.

    Returns
    -------
    data : xArray Dataset
        xArray Dataset with PBLH.

    '''
    
    # Note: offload numerical data from xArray format to save time
    # Load NumPy array with only theta_v values
    arr = data['potential_temperature'].values
    # Get differential of selected quantity (s) with respect to height (z)
    diff = np.diff(arr, axis=2)
    # Get NumPy array of heights
    heights = data.height.values
    # Initialize container NumPy array with shape of theta_v array to hold PBLH values
    pblh = np.empty(arr.shape)
    # Iterate over sites (axis = 1 in the NumPy array)
    for site in range(diff.shape[1]):
        # Iterate over timestamps (axis = 0 in the NumPy array)
        for time in range(diff.shape[0]):
            # Find the maximum value of ds/dz
            max_diff = np.amax(diff[time, site])
            # Find the index where max(ds/dz) is
            index = np.where(diff[time, site] == max_diff)
            # If there is such a value, grab the height. Else, np.nan
            if index[0].size != 0:
                height = np.repeat(heights[index[0][0]], arr.shape[2])
            else:
                height = np.repeat(np.nan, arr.shape[2])
            # Insert into the container NumPy array
            pblh[time, site] = height
    # Assign the container array into the Dataset
    data = data.assign(pblh=(['time', 'site', 'height'], pblh))      
    
    return data

if __name__ == '__main__':
    print('Run troubleshooting here...')