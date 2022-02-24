"""
Urban Boundary Layer Observation Data Processing
Script name:    Ceilometer Analysis
Path:           ~/bin/ceilometer.py
Description:    Process and analyze data from the Vaisala CL31 ceilometer atop Steinman Hall.
"""

import datetime, netCDF4 as nc, numpy as np, os, pandas as pd, xarray as xr

def processor(date_range, spectral_analysis=False, sites=['MANH']):
    '''
    Processes raw lidar data and returns an aggregate xArray over the dates and locations provided.

    Parameters
    ----------
    date_range : list
        Dates over which lidar data is desired.
    sites : list, optional
        List of sites at which data is desired. The default is ['BRON', 'MANH', 'QUEE', 'STAT'].

    Returns
    -------
    data : xArray Dataset
         xArray with data over the dates and locations provided.

    '''
    
    # Define file path to external hard drive
    fpath = '/Volumes/UBL Data/data/lidar'
    # Initialize empty dictionary for future use
    file_dict = {}
    # Build list of netCDF files for future concatenation into an xArray
    for root, dirs, files in os.walk(fpath):
        for directory in dirs:
            # Catch the loop and skip to next location if the site was not requested
            if directory not in sites:
                continue
            else:
                # Build temporary list of files for each location
                temp = []
                # Iterate through files for each location
                for file in os.listdir(os.path.join(root, directory)):
                    # Only select netCDF files
                    if file.split('.')[-1] == 'nc':
                        filename = file.split('.')[0]
                        # Handle date formatting for different lidar networks
                        # Manhattan: City College of New York
                        # Bronx, Queens, Staten Island: New York State Mesonet
                        if directory == 'MANH':
                            date = datetime.datetime.strptime(filename, '%Y%m%d%H%M%S')
                        else:
                            continue
                        # If file isn't within the specified date range, ignore it
                        # Note: "if date in date_range" being trialed to work with specific dates
                        if date_range[0] <= date < date_range[-1]:
                        # if date.date() in date_range:
                            temp.append(os.path.join(root, directory, filename + '.nc'))
                # Populate site-specific dictionary item with aggregated list
                file_dict[directory] = temp

