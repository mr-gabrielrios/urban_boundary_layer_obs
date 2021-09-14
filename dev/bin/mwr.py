"""
Urban Boundary Layer Observation Data Processing
Script name:    Microwave Radiometer Analysis
Path:           ~/bin/mwr.py
Description:    Process and analyze data from the microwave radiometer on top of Steinman Hall.
"""

import netCDF4 as nc, numpy as np, os, pandas as pd, xarray as xr

# Set local file path to pull data from
fpath_mwr = '/Users/gabriel/Downloads/2021-01-01_00-04-06_lv2.csv'
# Read in CSV and skip irrelevant rows
raw_data = pd.read_csv(fpath_mwr, sep=',', skiprows=2)
# Delete the first 3 rows
raw_data = raw_data.iloc[3:]
# Re-define column names to temporary values
raw_data.columns = raw_data.iloc[0]
# Remove columns only containing nans
raw_data = raw_data.iloc[1:].dropna(axis=1, how='all')
# Rename columns for consistency with NYS Mesonet variable naming conventions
raw_data = raw_data.rename(columns={'Date/Time': 'time'})
# Convert all Pandas time values to datetime values
raw_data['time'] = pd.to_datetime(raw_data['time'])
# Construct dictionary to dynamically assign data variable names for final xArray
param_dict = {'temperature': 401, 'vapor_density': 402, 'liquid': 403, 'relative_humidity': 404}
# Define list of times for use as a dimension and coordinate in final xArray
times = [i.to_pydatetime() for i in raw_data.loc[raw_data[400] == 401]['time']]
# Initialize empty list to hold temporary xArrays before merging to final xArray
params = []
# Iterate through keys and values to dynamically construct xArrays
for key, value in param_dict.items():
    # Format DataFrame corresponding to the current iterand by dropping certain columns and re-indexing
    df = raw_data.loc[raw_data[400].isin([value])].drop(columns=['Record', 400, 'LV2 Processor']).set_index('time').astype(float)
    # Create height range based on DataFrame column names
    heights = [float(i) for i in df.columns]
    # Construct temporary xArray using times and heights as dimensions and coordinates
    temp_xr = xr.Dataset(data_vars={key: (['time', 'height'], df.values)}, coords={'time': times, 'height': heights})
    # Append temporary xArray to list of xArrays for future merging
    params.append(temp_xr)
# Merge all xArrays into singular xArray
data = xr.merge(params)