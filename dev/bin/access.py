"""
Urban Boundary Layer Observation Data Processing
Script name:    NOAA-CREST Database Reader
Path:           ~/bin/access.py
Description:    Download data from the NOAA-CREST ORSL database for local storage for CCNY data.
"""

from bs4 import BeautifulSoup
import datetime
import matplotlib.pyplot as plt
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
import py7zr
import requests
import shutil
import ssl
import urllib
import warnings
import xarray as xr

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.captureWarnings(True)

def processor(data_type='flux', data_subtype='DBS'):
    '''
    Download all data from NOAA-CREST database given a data type. Downloads data for all days from Janury 2017 to September 2021, start-inclusive & end-exclusive. Meant to be a one-time run to port all data onto a local location.

    Parameters
    ----------
    data_type : str, optional
        Data type to access and download. The default is 'flux'.
    data_subtype : str, optional
        Manhattan-specific parameter to opt between fixed scan (FXD) and Doppler beam swinging (DBS) modes. FXD is used for vertical velocity at higher frequency, DBS is usedd for all 3 directions at a lower frequency.

    Returns
    -------
    None.

    '''
    
    # Trigger boolean only when requests want to be made - avoids getting shut out from server
    live = True
    
    # Define range of years for data access.
    date_range = [2017, 2018, 2019, 2020, 2021]
    
    # Access flux tower data
    if data_type == 'flux' or data_type == 'ts_data':
        # Define data type parent URL
        dirpath = 'https://datadb.noaacrest.org/public/ORSL/Archived_Loggernet/CCNY_Flux_Tower/'
        # Iterate through each year in the directory
        for year in date_range:
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
                # Build file path to save .csv contents to locally
                fpath = os.path.join(os.getcwd(), 'data', 'flux', 'MANH', href.split('/')[-1])
                # Only grab files that match the desired data type
                if href.split('MFT')[-1].split('.')[0][1:] == data_type:
                    print('Currently accessing: {0}'.format(href))
                    # Prevent re-reading and re-writing files
                    if live and not os.path.isfile(fpath):
                        # Use Pandas to read and save a .csv locally
                        df = pd.read_csv(href, sep=',', skiprows=2)
                        df.to_csv(fpath)
            
    # Access microwave radiometer data
    if data_type == 'mwr':
        # Define data type parent URL
        dirpath = 'https://datadb.noaacrest.org/public/ORSL/Archived_Microwave_Radiometer/'
        # Iterate through each year in the directory
        for year in date_range:
            print(year)
            # Certificate access workaround
            ssl._create_default_https_context = ssl._create_unverified_context
            # Define year-specific URLs
            if year == 2017 or year == 2018:
                # 2017 and 2018 have all .csv files in a single directory
                # Based on the year, access the corresponding folder
                if year == 2017:
                    url = os.path.join(dirpath, str(year), 'data-lv2')
                elif year == 2018:
                    url = os.path.join(dirpath, str(year), 'lv2')
                # Generate BeautifulSoup for this HTML page
                soup = BeautifulSoup(requests.get(url, verify=False).content, features='html.parser')
                # Iterate through each file
                for img in soup.find_all('img', alt='[TXT]'):
                    # Build path to relevant files
                    href = os.path.join(url, img.find_parent('a')['href'])
                    # Build file path to save .csv contents to locally
                    fpath = os.path.join(os.getcwd(), 'data', 'mwr', href.split('/')[-1].split('_')[0].replace('-', '') + '_lv2_PROF_MANH.csv')
                    print('Currently accessing: {0}'.format(href))
                    # Prevent re-reading and re-writing files
                    if live and not os.path.isfile(fpath):
                        # Use Pandas to read and save a .csv locally
                        df = pd.read_csv(href, sep=',', skiprows=2)
                        df.to_csv(fpath)
            else:
                # 2019, 2020, and 2021 have .csv files organized by day-specific directories
                if year == 2019:
                    url = os.path.join(dirpath, str(year))
                elif year == 2020:
                    url = os.path.join(dirpath, str(year))
                elif year == 2021:
                    # Note: 2021 has a 2021-specific subdirectory for some unknown reason
                    url = os.path.join(dirpath, str(year), str(year))
                # Generate BeautifulSoup for this HTML page
                soup = BeautifulSoup(requests.get(url, verify=False).content, features='html.parser')
                # Iterate through each directory
                for img in soup.find_all('img', alt='[DIR]'):
                    # Build path to relevant daily directory
                    url_day = os.path.join(url, img.find_parent('a')['href'])
                    # Generate BeautifulSoup for this day
                    soup_day = BeautifulSoup(requests.get(url_day, verify=False).content, features='html.parser')
                    # Iterate through each  in the daily directory
                    for img in soup_day.find_all('img', alt='[TXT]'):
                        # Select only the level 2 .csv files
                        if 'lv2.csv' in os.path.join(url_day, img.find_parent('a')['href']):
                            # Build path to relevant file
                            href = os.path.join(url_day, img.find_parent('a')['href'])
                            # Build file path to save .csv contents to locally
                            fpath = os.path.join(os.getcwd(), 'data', 'mwr', href.split('/')[-1].split('_')[0].replace('-', '') + '_lv2_PROF_MANH.csv')
                            print('Currently accessing: {0}'.format(href))
                            # Prevent re-reading and re-writing files
                            if live and not os.path.isfile(fpath):
                                # Use Pandas to read and save a .csv locally
                                df = pd.read_csv(href, sep=',', skiprows=2)
                                df.to_csv(fpath)

    # Access lidar data - fixed scan data only for vertical velocity due to temporal frequency requirements for turbulence spectral analysis. Assume only the Leosphere WindCube WLS200S is being used.
    if data_type == 'lidar':
        # Override date range - lidar data only currently available for 2021
        date_range = [2021]
        # Define data type parent URL
        dirpath = 'https://datadb.noaacrest.org/public/ORSL/Archived_Lidar/200S'
        # Hold container for temporary directory to be accessed later
        tmpdir = ''
        # Iterate through each year in the directory
        for year in date_range:
            # Certificate access workaround
            ssl._create_default_https_context = ssl._create_unverified_context
            # Define year-specific URL
            url = os.path.join(dirpath, str(year))
            # Generate BeautifulSoup for this HTML page
            soup = BeautifulSoup(requests.get(url, verify=False).content, features='html.parser')
            # Iterate through each directory
            for img in soup.find_all('img', alt='[DIR]'):
                # Build path to relevant daily directory
                url_day = os.path.join(url, img.find_parent('a')['href'])
                # Generate BeautifulSoup for this day
                soup_day = BeautifulSoup(requests.get(url_day, verify=False).content, features='html.parser')
                # Iterate through each folder in the daily directory
                for img in soup_day.find_all('img', alt='[   ]'):
                    # Select only the fixed scan files with substring 'FXD' in them
                    if data_subtype == 'FXD' and 'FXD' in os.path.join(url_day, img.find_parent('a')['href']):
                        # Build path to relevant file
                        href = os.path.join(url_day, img.find_parent('a')['href'])
                        # Create lidar directory reference
                        dirpath = os.path.join(os.getcwd(), 'data', 'lidar')
                        # Create Manhattan-specific directory to hold lidar data, if it doesn't exist
                        if not os.path.isdir(os.path.join(dirpath, 'PROF_MANH')):
                            os.mkdir(os.path.join(dirpath, 'PROF_MANH'))
                        # Build file path to save .7z to local directory
                        fpath = os.path.join(dirpath, 'PROF_MANH', href.split('/')[-1].split('_')[0])
                        # Extract file from URl and save to defined file path
                        # urllib.request.urlretrieve(url=href, filename=fpath)
                        print('Currently accessing: {0}'.format(href))
                        print('\t Saving file to {0}'.format(fpath))
                        # Define file path for a temporary directory to hold extracted archives
                        tmpdir = os.path.join(dirpath, 'PROF_MANH', 'tmp')
                        # Extract all files from .7z to the Manhattan lidar directory
                        # with py7zr.SevenZipFile(fpath, 'r') as archive:
                        #     # Extract .7z contents into a temporary directory
                        #     archive.extractall(path=tmpdir)
                        # # Remove archive
                        # os.remove(fpath)
                    # Select only the lidar files with substring 'DBS' in them
                    if data_subtype == 'DBS' and 'DBS' in os.path.join(url_day, img.find_parent('a')['href']):
                        # Filter by dates using numerical values
                        # 2021-07-14 to 2021-08-31 chosen to cover multiple heat waves within Steinman lidar data availability
                        if (int(url_day.split('/')[-2].split('_')[0]) <= 20210714) or  (int(url_day.split('/')[-2].split('_')[0]) > 20210831):
                            break
                        print(url_day)
                        # Build path to relevant file
                        href = os.path.join(url_day, img.find_parent('a')['href'])
                        # Create lidar directory reference
                        dirpath = os.path.join(os.getcwd(), 'data', 'lidar')
                        # Create Manhattan-specific directory to hold lidar data, if it doesn't exist
                        if not os.path.isdir(os.path.join(dirpath, 'PROF_MANH')):
                            os.mkdir(os.path.join(dirpath, 'PROF_MANH'))
                        # Build file path to save .7z to local directory with a DBS-specific identifier
                        fpath = os.path.join(dirpath, 'PROF_MANH', href.split('/')[-1].split('_')[0])
                        fpath = fpath.replace('.', '_dbs.')
                        # Extract file from URL and save to defined file path
                        urllib.request.urlretrieve(url=href, filename=fpath)
                        print('Currently accessing: {0}'.format(href))
                        print('\t Saving file to {0}'.format(fpath))
                        # Define file path for a temporary directory to hold extracted archives
                        tmpdir = os.path.join(dirpath, 'PROF_MANH', 'tmp')
                        # Extract all files from .7z to the Manhattan lidar directory
                        with py7zr.SevenZipFile(fpath, 'r') as archive:
                            # Extract .7z contents into a temporary directory
                            archive.extractall(path=tmpdir)
                        # Remove archive
                        os.remove(fpath)
        # Move all files to the top-level directory
        dirpath = os.path.join(dirpath, 'PROF_MANH')
        # Walk through all files and pull netCDF files
        file_list = [os.path.join(r,file) for r,d,f in os.walk(tmpdir) for file in f if file.split('.')[-1] == 'nc']
        # Move all the files to the top-level directory if the file doesn't already exist there
        [shutil.move(file, dirpath) for file in file_list if not os.path.isfile(os.path.join(dirpath, file.split('/')[-1]))]
        # Remove temporary directory
        shutil.rmtree(tmpdir)
            
    # Access ceilometer data from the Vaisala CL31 on top of Steinman
    # Note: data available in most recent format from 2020-06-15 onward
    if data_type == 'ceilometer':
        # Update date range to reflect data availability
        date_range = [2021]
        # Define data type parent URL
        dirpath = 'https://datadb.noaacrest.org/public/ORSL/Archived_Ceilometer/CL31/'
        # Create ceilometer directory reference
        storage_dir = '/Volumes/UBL Data/data/ceilometer'
        # Iterate through each year in the directory
        for year in date_range:
            # Certificate access workaround
            ssl._create_default_https_context = ssl._create_unverified_context
            # Define year-specific URL
            url = os.path.join(dirpath, str(year))
            # Generate BeautifulSoup for this HTML page
            soup = BeautifulSoup(requests.get(url, verify=False).content, features='html.parser')
            # Initialize string for directory where all temporary netCDF files are being kept
            dirpath = ''
            # Iterate through each directory
            for img in soup.find_all('img', alt='[DIR]'):
                break
                # Build path to relevant daily directory
                url_day = os.path.join(url, img.find_parent('a')['href'])
                # Generate BeautifulSoup for this day
                soup_day = BeautifulSoup(requests.get(url_day, verify=False).content, features='html.parser')
                # Iterate through each folder in the daily directory
                for img in soup_day.find_all('img', alt='[   ]'):
                    # Select only level 3 data (derived quantities from ceilometer)
                    if '/L3' in os.path.join(url_day, img.find_parent('a')['href']):
                        # Build path to relevant file
                        href = os.path.join(url_day, img.find_parent('a')['href'])
                        # Build file name for temporary save
                        tempname = href.split('/')[-2].split('_')[0] + '_raw.nc'
                        # Build file name for DataFrame save
                        fname = href.split('/')[-2].split('_')[0] + '.csv'
                        # Create temporary file path to the storage directory
                        temppath = os.path.join(storage_dir, tempname)
                        # Create temporary file path to the storage directory
                        fpath = os.path.join(storage_dir, fname)
                        # Extract file from URL and save to defined file path
                        # urllib.request.urlretrieve(url=href, filename=fpath)
                        print('Currently accessing: {0} \n'.format(href))
                        print('\t Saving temporary data to {0} \n -------'.format(temppath))
                        print('\t Saving DataFrame data to {0} \n -------'.format(fpath))
                        # Prevent re-reading and re-writing files for temporary netCDF4 files
                        if live and not os.path.isfile(fpath):
                            # Download netCDF4 file for temporary usage
                            urllib.request.urlretrieve(url=href, filename=temppath)
                            ceil = nc.Dataset(temppath)
                            # Save the boundary layer height to a Pandas DataFrame
                            df = pd.DataFrame(data=ceil['bl_height'][:, 0],
                                              columns=['bl_height'])
                            # Save the timestamp to a Pandas DataFrame (UTC time)
                            df['datetime'] = pd.to_datetime(ceil['date_stamp'][:, 0])
                            # Set the timestamp to be the index
                            df = df.set_index('datetime', drop=False)
                            # Get a 30-minute average
                            df = df.resample('30T').mean()
                            # Save to a CSV for future read-ins
                            df.to_csv(fpath)
                            # Remove the raw netCDF file
                            os.remove(temppath)
        # Collect all CSV files and place them into a .csv
        storage_dir = '/Volumes/UBL Data/data/ceilometer'
        files = [os.path.join(storage_dir, file) for file in os.listdir(storage_dir) if file.split('.')[-1] == 'csv']
        # Concatenate all DataFrames
        ceilometer_data = pd.concat([pd.read_csv(file) for file in files])
                            # Save the timestamp to a Pandas DataFrame (UTC time)
        ceilometer_data['datetime'] = pd.to_datetime(ceilometer_data['datetime'])
        # Set the timestamp to be the index
        ceilometer_data = ceilometer_data.set_index('datetime', drop=False)
        # Save the DataFrame to a CSV
        ceilometer_data.to_csv('/Volumes/UBL Data/data/storage/ceilometer_data_s20200615_e20211231.csv')
            
def create_file_from_source(file1, file2):
    with nc.Dataset(file1) as src, nc.Dataset(file2, "w", format='NETCDF4_CLASSIC') as dst:
        # copy global attributes all at once via dictionary
        dst.setncatts(src.__dict__)
        # copy dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
        # copy all file data except for the excluded
        for name, variable in src.variables.items():
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            dst[name][:] = src[name][:]
            # copy variable attributes all at once via dictionary
            dst[name].setncatts(src[name].__dict__)

def to_netcdf(fpath):
    '''
    Generates netCDF files from all .csv files in the given directory. Meant to be a one-time run for downloaded .csv files from the NOAA-CREST database.

    Returns
    -------
    None.

    '''
    
    # Define path to directory containing all .csv files to be converted to netCDF files
    fpath = '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/mwr'
    # Generate list of .csv files for future iteration and conversion
    files = [os.path.join(fpath, f) for f in os.listdir(fpath) if f.split('.')[-1] == 'csv']
    
    # Iterate through generated list of .csv files and convert them to netCDF files
    for file in sorted(files):
        print(file)
        raw_data = pd.read_csv(file)
        # Get number of rows to clip based on number of repeat values
        clip_num = raw_data['Date/Time'].str.count('Date*').sum()
        # Delete the first 3 rows
        raw_data = raw_data.iloc[(clip_num-1):]
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
            df = raw_data.loc[raw_data[400].isin([value])].drop(columns=['Record', 400, 'LV2 Processor']).set_index('time')
            # Eliminate any blank cells in the DataFrame (assume one blank cell per DataFrame)
            df.iloc[np.where(df.applymap(lambda x: x == '  '))[0], np.where(df.applymap(lambda x: x == '  '))[1]] = np.nan
            # Make columns with data types of 'object' to numeric types
            df = df.apply(pd.to_numeric)
            # Create height range based on DataFrame column names
            heights = [float(i) for i in df.columns]
            # Ensure no DataFrames with mismatching data values are incorporated to the xArray
            if df.values.shape[0] == len(times):
                temp_xr = xr.Dataset(data_vars={key: (['time', 'height'], df.values)}, coords={'time': times, 'height': heights})
            else:
                continue
            # Append temporary xArray to list of xArrays for future merging
            params.append(temp_xr)
        # Merge all xArrays into singular xArray
        data = xr.merge(params)
        # Write xArray to a netCDF file - allows for homogeneity with Mesonet data
        data.to_netcdf(file.split('.')[0] + '.nc')
        
if __name__ == '__main__':
    # Ceilometer testing
    '''
    processor(data_type='ceilometer')
    '''
    
    # Full data download
    # Define the data types to be downloaded
    data_types = ['lidar']
    # Pull all the data from the NOAA-CREST database. Note that lidar data is omitted because the file structure is terrible and incomplete. Note: data will be downloaded from 2017 to 2021 to match extent of research effort time period.
    for data_type in data_types:
        processor(data_type=data_type, data_subtype='DBS')
    # Convert any .csv files to .nc files for homogeneity with Mesonet data
    # to_netcdf()
    
