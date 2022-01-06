"""
Urban Boundary Layer Observation Data Processing
Script name:    AMDAR ACARS Data Processing
Path:           ~/bin/amdar.py
Description:    Process, analyze, and visualize AMDAR data.
"""

import cartopy.crs as ccrs, gzip, matplotlib.pyplot as plt, numpy as np, os, xarray as xr

def processor():
    '''
    Read in and process data for given parameters.

    Returns
    -------
    None.

    '''
    
    fpath = '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/amdar/2018/20180101/acars.20180101_1400'
    
    
if __name__ == '__main__':
    
    # Rough attempt at plotting all flights from an origin airport
    flights = []
    origin = 'SCE'
    for subdir, dirs, files in os.walk('/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/amdar'):
        for file in files:
            # Define universal file path
            fpath = subdir + os.sep + file
            # Filter out tar files
            if fpath.split('.')[-1] == 'gz':
                g = gzip.open(fpath)
                g_data = g.read()
                x_data = xr.open_dataset(g_data)
                # Get location bytes
                locs = x_data['origAirport'].values
                # Convert all bytes to strings for filtering
                locs = [str(loc, 'latin-1')[:3] for loc in locs]
                print(origin in locs)
                hits = np.nansum(origin in locs)
                if hits > 0:
                    flights.append(x_data)
                g.close()
    c = 0
    for flight in flights:
        flight['origAirport'].values = [str(loc, 'latin-1')[:3] for loc in flight['origAirport'].values]
        flight = flight.where(flight['origAirport'] == origin)
        if c < 5:
            fig, ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})
            im = ax.scatter(flight['longitude'], flight['latitude'], c=flight['altitude'].values)
            ax.coastlines()
            # ax.set_extent([flight['longitude'].min(), flight['longitude'].max(), flight['latitude'].min(), flight['latitude'].max()])
            ax.set_extent([-120, -70, 20, 50])
            c += 1
        else:
            break
    cbar = fig.colorbar(im, ax=ax)
    
    data = processor()