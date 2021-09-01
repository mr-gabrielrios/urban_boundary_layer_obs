"""
Urban Boundary Layer Observation Data Processing
Script name:    Marshak Flux Tower Reader
Path:           ~/bin/scintillometer.py
Description:    Process, analyze, and visualize data collected by the Scintec BLS450 scintillometer.
"""

import datetime, pandas as pd

def scintillometer_data(fpath, date_range):
    data = pd.read_csv(fpath, delimiter='\t')
    data.columns = ['date', 'time', 'Cn2', 'CT2', 'H', 'pressure', 'temp', 'humidity', 'error']
    
    data['timestamp'] = data['date'] + ' ' + data['time']
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.drop(columns=['date', 'time'])
    
    # Format dates into datetime format.
    date_format = '%Y%m%d%H'
    if date_range and len(date_range) == 2:
        date_range = [datetime.datetime.strptime(date_range[0], date_format),
                      datetime.datetime.strptime(date_range[1], date_format)]
        print(date_range)
        # Filter data for timestamps within given date range
        mask = (data['timestamp'] > date_range[0]) & (data['timestamp'] <= date_range[1])
        data = data.loc[mask]
        
    data = data.set_index('timestamp', drop=False)
    
    for col in data.columns:
        if col != 'timestamp':
            data[col] = data[col].str.split(' ').str[0]
            data[col] = data[col].astype(float)
    
    data = data.resample('30T').mean()
    
    return data

if __name__ == '__main__':
    # Temporary file path  
    print('Hi')