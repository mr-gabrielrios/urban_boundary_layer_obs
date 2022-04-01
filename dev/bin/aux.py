"""
Urban Boundary Layer Observation Auxiliary Functions
Script name:    Auxiliary Functions
Path:           ~/bin/aux.py
Description:    Store functions used for miscellaneous and non-essential tasks - namely, manuscript supplemental information.
"""

import cartopy.crs as ccrs, cartopy.io.img_tiles as cimgt, cartopy.feature as cfeature, datetime, matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, os, pandas as pd, PIL, xarray as xr, warnings

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle

# Ignore warnings
warnings.filterwarnings("ignore")
# Change font
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

def scale_bar(ax, length=None, location=(0.13, 0.97), linewidth=2):
    """
    By 'Siyh' - https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    # Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    # Make tmc horizontally centred on the middle of the map,
    # vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 # in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    # Plot the scalebar
    ax.plot(bar_xs, [sby, sby], linestyle=(0, (2, 2)), 
            transform=tmc, color='k', linewidth=linewidth)
    ax.plot(bar_xs, [sby, sby], linestyle=(2, (2, 2)), 
            transform=tmc, color='white', linewidth=linewidth)
    # Plot the scalebar label
    ax.text(sbx, -1000, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='top', fontsize=6)
    # Add north arrow
    # Add north arrow
    scale = 4
    xpos, lb, ub = 0.95, 0.94, 0.97
    ax.text(xpos - 0.05, (lb+ub)/2 - 0.0075, 'N', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=9)
    ax.annotate('',
            xy=(xpos, ub), xycoords='axes fraction',
            xytext=(xpos, lb),
            textcoords='axes fraction',
            arrowprops=dict(facecolor='black', width=0.1*scale, headwidth=1*scale, headlength=1.75*scale),
            horizontalalignment='left',
            verticalalignment='bottom')

def basemap(bound_box, pad=0.05, base_img='satellite', sat_type='RGB', high_res=False, boroughs=True, proj='PlateCarree'):
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
    if proj == 'PlateCarree':
        proj = ccrs.PlateCarree()
    elif proj == 'Orthographic':
        proj = ccrs.Orthographic(central_latitude=np.nanmean([lat_max, lat_min]),
                             central_longitude=np.nanmean([lon_max, lon_min]))
    # Initialize figure
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300, subplot_kw={'projection': proj})
    # Set spatial extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    # Define gridlines
    ax.set_xticks(np.linspace(lon_min, lon_max, 8), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(lat_min, lat_max, 8), crs=ccrs.PlateCarree())
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax.set_xlabel('Longitude [°E]', fontsize=8, labelpad=10)
    ax.set_ylabel('Latitude [°N]', fontsize=8, labelpad=10)
    
    ''' Add satellite or base imagery. '''
    if base_img == 'satellite':
        # Define imagery resolution
        res = 12 if high_res else 8
        # Request imagery
        sat_img = cimgt.GoogleTiles(desired_tile_form=sat_type, style='satellite')
        # Add satellite imagery to the map
        cmap = mpl.colors.LinearSegmentedColormap.from_list('bone_min', plt.get_cmap('gist_gray_r')(np.linspace(0.1, 0.4, 100)))
        ax.add_image(sat_img, res, cmap=cmap, alpha=0.5)
    elif base_img == 'nlcd':
        # Use NLCD land cover data for the basemap
        fname = '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/bin/assets/nlcd/NLCD_reprojected.tiff'
        nlcd = np.array(PIL.Image.open(fname))
        # Define land cover classes
        lookup = {'water': [11, 12], 
                  'low_density': [22, 31],
                  'wetlands': [90, 95],
                  'medium_density': [23],
                  'vegetated': [21, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82],
                  'high_density': [24]}
        # Remove erroneous data, if any exists
        nlcd = np.where(nlcd == 1, np.nan, nlcd)
        # Iterate through land cover classes and transform the NLCD array
        for i, values in enumerate(lookup.values()):
            for value in values:
                nlcd = np.where(nlcd == value, i, nlcd)
        # Define longitude and latitude vectors for basemap placement
        stretch = 0.16
        lon, lat = [np.linspace(lon_min-stretch, lon_max+stretch, nlcd.shape[1]), 
                    np.linspace(lat_max+stretch, lat_min-stretch, nlcd.shape[0])]
        
        # Create custom colormap
        colors = ['lightskyblue', 'snow', 'mediumaquamarine', 'silver',  'lightgreen', 'grey']
        nlcd_cmap = mpl.colors.ListedColormap(colors)
        # Plot the basemap
        ax.pcolormesh(lon-0.06, lat-0.025, nlcd, 
                      cmap=nlcd_cmap, alpha=0.5, transform=proj)
        
        custom_lines = []
        for i, lcz in enumerate(lookup.values()):
            label = Rectangle((0,0), 1, 1, facecolor=colors[i], linewidth=0.5, edgecolor='#222222')
            custom_lines.append(label)
        
        legend = ax.legend(custom_lines, 
                           [key.replace('_', ' ').capitalize()
                            for key in lookup.keys()], 
                           handlelength=0.9, handleheight=1, 
                           ncol=3, loc='lower center',
                           prop={'size': 6})
        legend.get_frame().set_facecolor((1, 1, 1, 0.8))
        legend.get_frame().set_edgecolor('none')
           
    else:
        boroughs = True
        # Add miscelleanous mapping features
        # ax.add_feature(cfeature.OCEAN, zorder=2)
        # ax.add_feature(cfeature.LAKES)
        # ax.add_feature(cfeature.LAND, facecolor='#eeeeee')
        # ax.coastlines('10m', zorder=3)
        
    ''' Add borough boundaries'''
    if boroughs:
        fname = '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/bin/assets/nyc_shp/geo_export_e33c300d-0bb8-4820-a4f3-a3fa4a356068.shp'
        boundaries = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='#666666', linewidth=1)
        ax.add_feature(boundaries, zorder=4, facecolor='none', alpha=0.75)
    # Suppress plot output
    plt.draw()
    
    ax.set_aspect(1.3)
    
    return fig, ax, proj

def observation_sites(bound_box, base_img='satellite', boroughs=False, high_res=False):
    '''
    Plot observation and other relevant locations on a map of New York City.

    Parameters
    ----------
    bound_box : list of tuples
        Pair of coordinates combining (N'most and W'most) and (S'most and E'most)

    '''
    
    sites = {'BRON': (40.87248, -73.89352, 'Lehman College'),
             'QUEE': (40.73433, -73.81585, 'Queens College'),
             'STAT': (40.60401, -74.14850, 'College of Staten Island')}
    sites = pd.DataFrame.from_dict(sites, orient='index', columns=['latitude', 'longitude', 'name'])
    
    ''' Plotting for spatial data. '''
    # Create figure for this data.
    fig, ax, proj = basemap(bound_box, base_img=base_img, high_res=high_res, boroughs=boroughs)
    
    # Plot coordinates
    for site, sitedata in sites.iterrows():
        ax.scatter(sitedata['longitude'], sitedata['latitude'], 
                   s=20, marker='o', color='r', zorder=5, transform=ccrs.PlateCarree())
        
        # Set horizontal alignment
        ha = 'right' if site in ['BRON', 'QUEE'] else 'left'
        # Set padding
        left, top = [-0.025, -0.0075] if site in ['BRON', 'QUEE'] else [0.025, -0.0075]
        # Add location label
        ax.text(sitedata['longitude'] + left, sitedata['latitude'] + top, sitedata['name'], ha=ha, transform=ccrs.PlateCarree(), zorder=6, fontsize=7)
    
    # Add scale to map
    scale_bar(ax, 10)
    
    return fig, ax
    
def nyc_building_heights(bound_box):
    '''
    Plot New York City building heights based on NYC Department of Planning information.
    Source: https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page
    '''
    
    ''' Building height data processing. '''
    # Load data. To get this dataset, use the link provided above.
    data = pd.read_csv('/Volumes/UBL Data/data/reference/nyc_pluto_21v4_csv/pluto_21v4.csv', usecols=['borough', 'latitude', 'longitude', 'numfloors', 'zipcode'])
    
    # Get spatial extent of the loaded data
    lat_max, lon_min, lat_min, lon_max = [data['latitude'].max(), 
                                          data['longitude'].min(), 
                                          data['latitude'].min(), 
                                          data['longitude'].max()]
    # Bin the coordinate data into pre-defined number of bins
    n_bins = 80
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
    # Adjust for floor to meters. Assume standard floor height is 3.05 m, or 10 ft.
    grid = np.abs(grid * 3.05)
    
    ''' Plotting for spatial data. '''
    # Create figure for this data.
    fig, ax, proj = basemap(bound_box, base_img='minimal', sat_type='RGB', high_res=False, boroughs=True)
    # Normalize plot values
    levels = np.arange(0, np.nanmax(grid), 20)
    # Plot the heat map
    data = data.sort_values('numfloors')
    values = data['numfloors'] * 3.05
    im = ax.pcolormesh(lons, lats, grid, cmap='plasma', transform=ccrs.PlateCarree())
    # im = ax.scatter(data['longitude'], data['latitude'], c=values, 
    #                 cmap='plasma', transform=ccrs.PlateCarree(), s=2)
    # Define colorbar
    colorbar = fig.colorbar(im, ax=ax, extend='max')
    colorbar.set_label('Building Height [m]', rotation=270, labelpad=20)
    
    output = xr.Dataset({'height': (['x', 'y'], grid)},
                        coords={'lat': (['x', 'y'], lats),
                                'lon': (['x', 'y'], lons)})
    
    # Define site coordinates
    obs_sites = {'BRON': (40.8729, -73.8945, 'Lehman College'),
                  'QUEE': (40.73433, -73.81585, 'Queens College'),
                  'STAT': (40.60401, -74.14850, 'College of Staten Island')}
    # Define search tolerance
    tol = 0.01
    # Print site-area building heights
    building_heights = {}
    # Iterate over each site to find the area-averaged building 
    for site, site_data in obs_sites.items():
        site_lat, site_lon = site_data[0], site_data[1]
        tol_val = output.where((output['lat'] > site_lat-tol) & 
                               (output['lat'] < site_lat+tol) & 
                               (output['lon'] > site_lon-tol) & 
                               (output['lon'] < site_lon+tol), drop=True)
        building_heights[site] = np.nanmin(tol_val['height'].values)
    
    print(building_heights)
    
    return fig, ax, output
    
def narr(date):
    '''
    Plot NARR data for a given date over the Northeast. 
    '''
    
    ''' Build preliminary map definitions. '''
    # Plot New York City central coordinates and define spatial extent parameter
    central_latitude, central_longitude, extent = [40.7128, -74.0060, 15]
    # Get spatial extent bounds
    lat_max, lon_min, lat_min, lon_max = [central_latitude + extent, 
                                          central_longitude - extent,
                                          central_latitude - extent, 
                                          central_longitude + extent]
    # Define the cartographic projection for the maps
    proj = ccrs.Orthographic(central_latitude=np.nanmean([lat_max, lat_min]),
                             central_longitude=np.nanmean([lon_max, lon_min]))
    
    ''' Load data for 2021. '''
    # Define directory path where NARR data is stored
    dir_ = '/Volumes/UBL Data/data/narr'
    # Build list of .nc files in the directory
    files = [os.path.join(dir_, file) for file in os.listdir('/Volumes/UBL Data/data/narr')]
    # Concatenate and load into a single xArray Dataset
    data = xr.open_mfdataset(files)
    
    # Clip data to spatial extent defined
    clipped = data.where(((data.lat > lat_min) & (data.lat < lat_max) 
                          & (data.lon > lon_min) & (data.lon < lon_max)), 
                         drop=True)
    # Limit temporal extent of data to prevent other dates from messing with the data range
    clipped = clipped.sel(time=date)
    # Get 2m air temperature
    temp_2m = clipped['air']
    # Get mean sea level pressure
    mslp = clipped['prmsl'] // 100
    
    ''' Plot data. '''
    fig, ax = plt.subplots(dpi=300, subplot_kw={'projection': proj})
    # Padding to prevent visibility of clipped spatial extent
    pad = 5
    # Set spatial extent
    ax.set_extent([lon_min+pad, lon_max-pad, lat_min+pad, lat_max-pad])
    # Set colormap for temperature
    cmap = 'Spectral_r'
    # Normalize temperature data
    norm = mpl.colors.Normalize(vmin=np.nanmin(clipped['air'].values),
                                vmax=np.nanmax(clipped['air'].values))
    # Plot temperature
    temp_2m_plot = ax.contourf(clipped.lon, clipped.lat, temp_2m.values, 
                               levels=32, norm=norm, cmap=cmap, transform=ccrs.PlateCarree())
    # Plot mean sea level pressure
    mslp_plot = ax.contour(clipped.lon, clipped.lat, mslp.values, 
                           levels=10, colors='k', transform=ccrs.PlateCarree(), zorder=4)
    # Add labels for the contours
    mslp_labels = ax.clabel(mslp_plot, mslp_plot.levels, 
                            fmt='%0.0f', inline=True, inline_spacing=40, fontsize=8, zorder=5)
    # Define the colorbar
    colorbar = fig.colorbar(temp_2m_plot, norm=norm, cmap=cmap, extend='both')
    # Add miscelleanous mapping features
    ax.add_feature(cfeature.OCEAN, zorder=2)
    ax.add_feature(cfeature.LAKES)
    ax.coastlines('10m', zorder=3)
    
    ax.set_title(date, loc='left')
    
    return data

def observation_sites_minimal(bound_box, base_img='satellite'):
    '''
    Plot observation and other relevant locations on a map of New York City.

    Parameters
    ----------
    bound_box : list of tuples
        Pair of coordinates combining (N'most and W'most) and (S'most and E'most)

    '''
    
    obs_sites = {'BRON': (40.8729, -73.8945, 'Lehman College'),
                  'QUEE': (40.73433, -73.81585, 'Queens College'),
                  'STAT': (40.60401, -74.14850, 'College of Staten Island')}
    obs_sites = pd.DataFrame.from_dict(obs_sites, 
                                       orient='index', 
                                       columns=['latitude', 'longitude', 'name'])
    
    ''' Plotting for spatial data. '''
    # Create figure for this data.
    fig, ax, proj = basemap(bound_box, base_img=base_img, high_res=False, boroughs=False)
    
    count = 0
    # Plot coordinates
    for site, sitedata in obs_sites.iterrows():
        if count == 0:
            label = 'Flux tower'
            count += 1
        else:
            label = '_nolegend_'
        ax.scatter(sitedata['longitude'], sitedata['latitude'], 
                   label=label, s=20, marker='s', color='r',
                   transform=ccrs.PlateCarree())
        plt.axis('off')
        
    # Add text for each borough
    ax.text(-74.02, 40.9, 'The Bronx', ha='center', transform=ccrs.PlateCarree())
    ax.text(-74.02, 40.54, 'Brooklyn', ha='center', transform=ccrs.PlateCarree())
    ax.text(-74.1, 40.75, 'Manhattan', ha='center', transform=ccrs.PlateCarree())
    ax.text(-73.7, 40.8, 'Queens', ha='center', transform=ccrs.PlateCarree())
    ax.text(-74.15, 40.669, 'Staten Island', ha='center', transform=ccrs.PlateCarree())
    
    
    # Add north arrow
    scale = 5
    xpos, lb, ub = 0.9, 0.91, 0.95
    ax.text(xpos, (lb+ub)/2 + 0.07, 'N', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.annotate('',
            xy=(xpos, ub), xycoords='axes fraction',
            xytext=(xpos, lb),
            textcoords='axes fraction',
            arrowprops=dict(facecolor='black', width=0.1*scale, headwidth=1*scale, headlength=1.75*scale),
            horizontalalignment='left',
            verticalalignment='bottom')
    
    return fig, ax

if __name__ == '__main__':
    # Pair of coordinates (NW, SE) to define bounding box
    bound_box = [(40.915, -74.256), (40.492, -73.699)]
    # Print New York City building heights
    # _, heights_ax, heights = nyc_building_heights([(40.915, -74.256), (40.492, -73.699)])
    # Show NARR data for a given date
    # narr(r'2021-08-12 15:00:00')
    # Print observation locations
    # _ , sites_ax = observation_sites_minimal(bound_box, base_img='minimal')
    _ , sites_ax = observation_sites(bound_box, base_img='nlcd', high_res=False)
    
    
    
