"""
Urban Boundary Layer Observation Auxiliary Functions
Script name:    Auxiliary Functions
Path:           ~/bin/aux.py
Description:    Store functions used for miscellaneous and non-essential tasks - namely, manuscript supplemental information.
"""

import cartopy.crs as ccrs, cartopy.io.img_tiles as cimgt, cartopy.feature as cfeature, matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, os, pandas as pd

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def basemap(bound_box, pad=0.05, sat_img=False, high_res=False):
    '''
    Plot satellite imagery and labels given a spatial extent.

    Parameters
    ----------
    bound_box : list of tuples
        Pair of coordinates combining (N'most and W'most) and (S'most and E'most) extent of the city.
    pad : float, optional
        Padding on coordinates to extend spatial extent of map. The default is 0.1.
    sat_img : bool, optional
        Request satellite imagery. The default is False to save runtime during drafts.
    high_res : bool, optional
        Request high-resolution imagery from Google. The default is False to save runtime during drafts.

    Returns
    -------
    None.

    '''
    
    # Get coordinates assuming a NW to SE approach.
    (lat_max, lon_min), (lat_min, lon_max) = bound_box
    (lat_max, lon_min), (lat_min, lon_max) = (lat_max + pad, lon_min - pad), (lat_min - pad, lon_max + pad)
    
    ''' Plot formatting rules. '''
    # Define projection
    proj = ccrs.Orthographic(central_latitude=np.nanmean([lat_max, lat_min]),
                             central_longitude=np.nanmean([lon_max, lon_min]))
    # Initialize figure
    fig, ax = plt.subplots(dpi=300, subplot_kw={'projection': proj})
    # Set spatial extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    # Define gridlines
    gridlines = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gridlines.top_labels, gridlines.right_labels = False, False
    # Define longitude and latitude formatting rules
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ''' Add satellite or base imagery. '''
    if sat_img:
        # Define imagery resolution
        res = 12 if high_res else 8
        # Request imagery
        sat_img = cimgt.GoogleTiles(style='terrain')
        # Add satellite imagery to the map
        ax.add_image(sat_img, res)
    else:
        ax.add_feature(cfeature.LAND.with_scale('10m'))
        ax.coastlines('10m', zorder=0)
    
    ''' Add observation locations. '''
    # Location coordinates
    coords = {'Bronx': (40.87248, -73.89352),
              'Manhattan': (40.82044, -73.94836),
              'Queens': (40.73433, -73.81585),
              'Staten Island': (40.60401, -74.14850)}
    
    # Suppress plot output
    plt.draw()
    
    return fig, ax, proj

    
def nyc_building_heights(bound_box):
    '''
    Plot New York City building heights based on NYC Department of Planning information.
    Source: https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page
    '''
    
    ''' Building height data processing. '''
    # Load data. To get this dataset, use the link provided above.
    data = pd.read_csv('/Volumes/UBL Data/data/reference/nyc_pluto_21v4_csv/pluto_21v4.csv', 
                       usecols=['borough', 'latitude', 'longitude', 'numfloors', 'zipcode'])
    
    # Get spatial extent of the loaded data
    lat_max, lon_min, lat_min, lon_max = [data['latitude'].max(), 
                                          data['longitude'].min(), 
                                          data['latitude'].min(), 
                                          data['longitude'].max()]
    # Bin the coordinate data into pre-defined number of bins
    n_bins = 60
    lat_bins = np.linspace(lat_min, lat_max, n_bins)
    lon_bins = np.linspace(lon_min, lon_max, n_bins)
    # Create meshgrid based on binned coordinates. To be used for future plotting
    lons, lats = np.meshgrid(lon_bins, lat_bins)
    # Initialize empty grid that will be populated in the loop
    grid = np.full((len(lat_bins), len(lon_bins)), np.nan)
    i = 0
    for column, column_data in data.groupby(pd.cut(data['latitude'], lat_bins)):
        j = 0
        for elem, elem_data in column_data.groupby(pd.cut(column_data['longitude'], lon_bins)):
            grid[i, j] = np.nanmean(elem_data['numfloors'])
            j += 1
        i += 1
    
    ''' Plotting for spatial data. '''
    # Create figure for this data.
    fig, ax, proj = basemap(bound_box, sat_img=False, high_res=False);
    # Normalize plot values
    norm = mpl.colors.Normalize(vmin=0, vmax=150)
    # Choose colormap
    cmap = 'BuPu'
    # Plot the contour map
    im = ax.contourf(lons, lats, grid * 10, 
                     levels=20, norm=norm, extend='max', cmap=cmap, 
                     transform=ccrs.PlateCarree())
    ax.coastlines('10m')
    colorbar = fig.colorbar(im, ax=ax, extend='max')
    
    return grid
    
if __name__ == '__main__':
    # Print New York City
    grid = nyc_building_heights([(40.915, -74.256), (40.492, -73.699)])
    

