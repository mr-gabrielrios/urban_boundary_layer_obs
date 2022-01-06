"""
Urban Boundary Layer Observation Data Processing
Script name:    ASOS Processing
Path:           ~/bin/asos.py
Description:    Use local ASOS data to identify heat wave events in New York City.
"""

import calendar, datetime, numpy as np, os, pandas as pd, re, shutil, sys, time, urllib.request
    
# Boolean to control print statements for script diagnostics.
str_switch = True

def KTtoMS(u):
    return u*0.51444

def CtoK(T):
    return T + 273.15 

def distance(crd, lat_asos, lon_asos):
    '''
    Calculate great circle distance between a point and an ASOS station.

    Parameters
    ----------
    crd : list or tuple
        Coordinates of point of interest.
    lat_asos : float
        Latitude of ASOS station.
    lon_asos : float
        Longitude of ASOS station.

    Returns
    -------
    Distance of point of interest to an ASOS station in meters.

    '''
    
    # GRS80 semi-major axis of Earth, per GOES-16 PUG-L2, Volume 5, Table 4.2.8
    R = 6378137 
    p = np.pi/180
    a = 0.5 - np.cos((lat_asos-crd[0])*p)/2 + np.cos(crd[0]*p) * np.cos(lat_asos*p) * (1-np.cos((lon_asos-crd[1])*p))/2
    return 2*R*np.arcsin(np.sqrt(a))



def asos_find(crd):
    '''
    Find closest ASOS station to a given coordinate.

    Parameters
    ----------
    crd : list or tuple
        Coordinates of a given point of interest.

    Returns
    -------
    station : str
        Station code for the closest ASOS station.

    '''
    
    t = time.time() 
    # Text file listing all ASOS stations with metadata.
    asos_station_fp = r'https://www.ncdc.noaa.gov/homr/file/asos-stations.txt'
    # GRS80 semi-major axis of Earth, per GOES-16 PUG-L2, Volume 5, Table 4.2.8
    R = 6378137
    # Define relevant columns for station location
    asos_cols = ['CALL', 'NAME', 'LAT', 'LON', 'ELEV', 'UTC']
    # Read data into ASOS DataFrame (adf)
    adf = pd.read_fwf(asos_station_fp, usecols=asos_cols).drop([0])
    # Filter stations with null data
    adf = adf[adf['ELEV'].astype(float) > -99999]
    # Read relevant parameters into lists for iterative purposes
    stations, lat_asos, lon_asos = [adf['CALL'].tolist(), adf['LAT'].astype(float).tolist(), adf['LON'].astype(float).tolist()]
    
    # Define arbitrarily large number as an initial condition (rouglhy equal to circumference of Earth)
    dists = 2*np.pi*R 
    # Initialize empty string for population
    station = '' 
    # Iterate over list of statons to find closest station
    for i in range(1, len(lat_asos)):
        dist = distance(crd, lat_asos[i], lon_asos[i])
        if dist < dists:
            dists = dist
            station = 'K' + stations[i]

    if str_switch:
        print("Closest station: %s, %.2f m away" % (station, dists))
        print('asos_find runtime: %.4f s' % (time.time() - t))
    
    # Manual override for work in New York City - KNYC is generally unreliable.
    if station == 'KNYC':
        station = 'KLGA'
        
    return station



def wfile(fpath, crd):
    '''
    Download ASOS data file for month corresponding to given date for the corresponding location.

    Parameters
    ----------
    fpath : str
        String containing file path of ASOS file.
    crd : list or tuple
        Coordinates of the point of interest

    Returns
    -------
    local_path : str
        File path to the data file.

    '''
    
    # Define path where ASOS data will be saved locally
    local_path = os.path.join(os.path.dirname(__file__), 'asos_data/' + '64010' + asos_find(crd) + fpath)
    
    print('Reading from {0} through {1}...'.format(asos_find(crd), local_path))
    # If path exists locally, use that. Else, download it and use locally moving forward.
    if os.path.isfile(local_path):
        return local_path
    else:
        # Create ASOS data folder if non-existent
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), local_path.split('/')[-2])):
            # Choose -4 to omit the 3-character file extension
            os.mkdir(os.path.join(os.path.dirname(__file__), local_path.split('/')[-2]))
        url = 'ftp://ftp.ncdc.noaa.gov/pub/data/asos-fivemin/6401-' + fpath[0:4] + '/64010' + asos_find(crd) + fpath
        with urllib.request.urlopen(url) as f:
            dat = f.read().decode('utf-8')
            file = open(local_path, 'w')
            file.write(dat)
            file.close()
        return local_path
         
##############################################################################################
# Method name:      data_read
# Method objective: Pull selected data from ASOS file for a given spatial domain and point.
# Input(s):         start_date [datetime], end_date [datetime], crd [list or tuple]
# Outputs(s):       df [Pandas DataFrame]
##############################################################################################

def data_read(date_range, crd, utc_offset):
    
    # Number of ASOS observations an hour
    interval = 12
    freq = ['5min', 'H']
    
    # Adjust datetimes to timezone corresponding to location of interest
    date_range = pd.date_range(start=date_range[0], end=date_range[-1], freq=freq[0]) 
    start_date, end_date = [date_range[0], date_range[-1]]
    
    # Generate strings from dates for comparison purposes
    date_str = [datetime.datetime.strftime(start_date, '%Y%m%d%H%M'),
                datetime.datetime.strftime(end_date, '%Y%m%d%H%M')]
    
    # Initialize DataFrame here to return nan DataFrame in case of failed FTP connection
    df = pd.DataFrame(np.nan, index=date_range, columns=['sky_cover', 'station'])
    # Set up URL to appropriate data file
    if str_switch:
        print(date_str)
    data_url = wfile(date_str[0][0:6] + '.dat', crd)
    
    # Import data to DataFrame 'df'
    try:
        asos_data = pd.read_table(data_url, header=None)
    except:
        print('FTP connection failed. Exiting program...')
        sys.exit()
        
    ## Regex patterns for data mining through the .dat file(s)
    # Air temperature regex: string of 6 characters "(0-9)(0-9)/(0-9)(0-9)" bounded by 2 spaces
    T_pattern = r'\s.?\d\d[+-/].?\d\d\s'
    # Wind speed regex: string of 6 characters "(0-9)(0-9)KT " bounded by 2 numbers and a space
    # Note: This definition ignores gusts
    u_pattern = r"\s\d\d\d\d\d\D"
    # Note: This definition allows the gust becomes the effective wind speed
    # u_pattern = r"\d\d[K][T]\s\d"
    # Air pressure regex: string of 6 characters "SLP(0-9)(0-9)"
    p_pattern = r"[S][L][P]\d\d\d"
    # Sky cover regex: string of variable number of characters "SM (variable number of characters and digits) (0-9)(0-9)/"
    cover_pattern = r"[S][M]\s(.*)\s\d\d(/)"
    
    # Iterate through all rows in ASOS data file. For dates in file that are within date range, extract data.
    for row in asos_data.iloc[:, 0]:
        if len(row) > 100 and datetime.datetime.strptime(row[13:23], '%Y%m%d%H') in df.index:
            # If sky cover pattern is found, extract data.
            if re.findall(cover_pattern, row):
                date = datetime.datetime.strptime(row[13:25], '%Y%m%d%H%M')
                
                # Store station name
                df.loc[date, 'station'] = row[5:9]
                # Extract sky cover information. If sky cover is clear, set to True, otherwise False.
                sky_cover_string = re.findall(cover_pattern, row)[0] 
                if 'CLR' in sky_cover_string:
                    df.loc[date, 'sky_cover'] = True
                else:
                    df.loc[date, 'sky_cover'] = np.nan
    
    # Average over all observations to produce hourly, then re-index to set dates to proper indices.
    dates = pd.date_range(start=date_range[0], end=date_range[-1], freq=freq[0])
    df = df.reindex(dates, fill_value=np.nan)
    
    # Delete ASOS data folder created locally
    shutil.rmtree(os.path.join(os.path.dirname(__file__), data_url.split('/')[-2]))

    return df

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

def clear_sky_finder(date_range, locs, threshold=0.5):
    '''
    Return list of dates with predominantly clear skies.

    Parameters
    ----------
    date_range : list
        List of datetime elements.
    locs : list
        List of lists or tuples containing coordinates of points of interest.
    threshold : float, optional
        Value denoting percentage (0 to 1 range) of clear sky readings required to pass.

    Returns
    -------
    days : list
        List of days that meet the clear sky threshold.

    '''
    
    # Generate list of "first of month" dates to allow for month to month indexing
    fom = [date for date in date_range if date.day == 1]
    
    # Capture all monthly ASOS data and concatenate into single DataFrame
    dfs = []
    locs = [(40.7769, -73.8740), (40.6413, -73.7781), (40.6895, -74.1745)]
    for loc in locs:
        for month in sorted(fom):
            # Get data from the entire month using a time offset equal to the number of days in the month.
            df = data_read([month, month+datetime.timedelta(days=calendar.monthrange(month.year, month.month)[1])], loc, -4)
            dfs.append(df)
    dfs = pd.concat(dfs)
    
    dfs['date'] = dfs.index
    grouped_df = dfs.groupby([dfs.date.dt.date]).count()
    grouped_df = grouped_df.where(grouped_df['sky_cover'] / grouped_df['station'] >= 0.5).dropna()
    days = grouped_df.index
    
    return days

def main(date_range, dirpath='/Volumes/UBL Data/data/ncei'):
    # Read in data from directory
    data = processor(dirpath)
    # Filter out data beyond date range
    heat_wave_days = [i for i in heat_wave_finder(data) if date_range[0] <= i.date() <= date_range[-1]] 
    
    return heat_wave_days
    
if __name__ == '__main__':
    date_range = pd.date_range(start='2021-06-01', end='2021-06-30', freq='MS')
    
    # Troubleshooting
    
    