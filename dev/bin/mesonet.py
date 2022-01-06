"""
Urban Boundary Layer Observation Data Processing
Script name:    Mesonet Data Reader
Path:           ~/bin/mesonet.py
Description:    Process and analyze data from New York State Mesonet stations.
"""

# Library imports
import datetime, numpy as np, os, matplotlib.pyplot as plt, pandas as pd

def file_sort(fn):
    '''
    Returns filename string to use as key for filename list sorting.

    Parameters
    ----------
    fn : str
        Filename.

    Returns
    -------
    Truncated filename with only first 8 characters.

    '''
    
    return(fn[0:8])

def csv_reader(date_range, data_dir, site):
    
    # Define CSV columns to extract data from
    # Datetime, sensible heat flux, friction velocity, air temperature (deg C)
    cols = ['datetime', 'H', 'LE', 'USTAR', 'ZL', 'Tc']
    
    # Initialize empty DataFrame with predefined columns
    data = pd.DataFrame(columns=cols)
    
    # Sort files by date, assuming standard Mesonet filename convention
    # Example: YYYYMMDD_PARAMETER_LOCATION_Parameter_NYSMesonet.csv
    directory = os.path.join(data_dir, site)
    if os.path.isdir(directory):
        file_list = sorted(os.listdir(directory), key=file_sort)
        file_list = [os.path.join(directory, file) for file in file_list if 'Flux' in file and len(file.split('/')[-1]) == 38]
    else:
        return None
    
    # Specify date range of interest
    # Format: YYYYMMDDHHMM
    date_range = sorted(date_range)
    date_range = pd.date_range(start=date_range[0], end=date_range[-1], freq='30T')
    
    # Iterate through sorted file list and extract data within date range with daily resolution
    
    for i, file in enumerate(file_list):
        file_date = datetime.datetime.strptime(file.split('/')[-1][0:8], '%Y%m%d')
        # Reduce datetimes to daily resolution to work on down-filtering
        days = [datetime.datetime.strptime(date_range[0].strftime('%Y%m%d'), '%Y%m%d'),
                datetime.datetime.strptime(date_range[-1].strftime('%Y%m%d'), '%Y%m%d')]
        # Filter files by day - any day within the date range will have a corresponding file
        if days[0] <= file_date <= days[-1]:
            filename = os.path.join(data_dir, file)
            data = data.append(pd.read_csv(filename, usecols=cols))
    
    # Convert date strings to datetime data type
    data['datetime'] = pd.to_datetime(data['datetime'])
    # Filter data entries by full datetime (includes hours and minutes)
    data = data[(data['datetime'] >= date_range[0]) & (data['datetime'] <= date_range[-1])] 
    data = data.reset_index(drop=True)
    
    # Account for missing observation data by inserting nans
    for i, date in enumerate(date_range):
        nanrow = pd.DataFrame([[np.nan] * len(cols)], columns=cols)
        nanrow['datetime'] = date
        if data.loc[i, ['datetime']].item() != date:
            data = pd.concat([data.iloc[:i], nanrow, data.iloc[i:]]).reset_index(drop=True)
        
    # Re-cast numerical strings as floats
    data['H'] = data['H'].astype(float)
    data['LE'] = data['LE'].astype(float)
    data['USTAR'] = data['USTAR'].astype(float)
    data['ZL'] = data['ZL'].astype(float)
    data['Tc'] = data['Tc'].astype(float)
    data['Tc'] = [i + 273.15 for i in data['Tc']]
    # Match parameter names to model parameter names
    data.rename(columns = {'H': 'QH',
                           'LE': 'QE',
                           'USTAR': 'u_star',
                           'ZL': 'zeta',
                           'Tc': 'T_air'}, inplace=True) 
    # data = data.iloc[::2].reset_index(drop=True)
    data['site'] = site
    
    return data

def xr_merge(data, param):
    '''
    Integrate Mesonet data to xArray Dataset as a data variable for a given parameter.

    Parameters
    ----------
    data : xArray Dataset
    param : str
        Parameter in Mesonet data that will be integrated into the xArray Dataset.

    Returns
    -------
    data : xArray Dataset
    '''
    
    # Container NumPy array that will be used for Dataset variable assignment
    arr = np.full((len(data.site.values), len(data.height.values), len(data.time.values)), np.nan)
    # Iterate over all sites in Dataset
    for i, site in enumerate(data.site.values):
        # If site exists, extract corresponding Mesonet data
        try:
            # Grab Mesonet data from the iterand site
            mesonet_data = csv_reader(data.time.values, '/Volumes/UBL Data/data/flux', site=site)
            # Set index to datetime and re-index based on xArray Dataset
            mesonet_data = mesonet_data.set_index('datetime').reindex(data.time.values, fill_value=np.nan)
            # Expand Mesonet data along the height axis and repeat to fill. Then, insert to the container array
            arr[i, :, :] = np.repeat(np.expand_dims(np.array(mesonet_data[param]), axis=0), len(data.height.values), axis=0)
        # Else, skip this site
        except:
            continue
    # Assign container array data to the xArray Dataset as a data variable
    data = data.assign({param : (('site', 'height', 'time'), arr)})
    
    return data

def convert(raw_path, save_dir):
    '''
    Function to convert monthly Mesonet data to daily data to match existing formatting.

    Parameters
    ----------
    raw_path : str
        Absolute path to directory where raw Mesonet data is stored.
    save_path : str
        Absolute path to parent directory of all location-specific flux data files.

    Returns
    -------
    None.
    '''
    
    # Iterate through each raw file
    for file in os.listdir(raw_path):
        # Join directory and file paths
        filepath = os.path.join(raw_path, file)
        # Open the CSV
        raw_data = pd.read_csv(filepath)
        # Group by location
        raw_data_grouped_loc = raw_data.groupby(['stid'])
        # Iterate through location groups
        for label, data in raw_data_grouped_loc:
            data['datetime'] = pd.to_datetime(data['datetime'], format='%Y%m%d %H:%M:%S')
            # Create dedicated column for day of year
            data['day'] = data['datetime'].dt.strftime('%Y%m%d')
            # Group by day
            raw_data_grouped_day = data.groupby('day')
            # Iterate through day groups
            for sublabel, subdata in raw_data_grouped_day:
                print(label, sublabel)
                # Define file name
                save_name = sublabel + '_' + label + '_Flux_NYSMesonet.csv'
                # Define destination file path
                save_path = os.path.join(save_dir, label.split('_')[-1], save_name)
                print(save_path)
                # Save file to appropriate folder
                subdata.to_csv(save_path)
    

if __name__ == "__main__":
    
    '''
    # Enable if conversion of raw data is needed
    raw_path, save_path = '/Volumes/UBL Data/data/flux/raw', '/Volumes/UBL Data/data/flux'
    convert(raw_path, save_path)
    '''
    
    '''
    date_range = [datetime.datetime(year=2019, month=7, day=28, hour=5),
                  datetime.datetime(year=2019, month=7, day=29, hour=5)-datetime.timedelta(hours=1)]
    date_range = pd.date_range(start=date_range[0], end=date_range[1], freq='H') 
    data_dir = os.path.join(os.path.dirname(__file__), '/Volumes/UBL Data/data/flux/BKLN')
    data = csv_reader(date_range, data_dir)
    '''