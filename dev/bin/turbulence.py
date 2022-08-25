"""
Urban Boundary Layer Observation Data Processing
Script name:    Turbulence Analysis
Path:           ~/bin/turbulence.py
Description:    Perform turbulence analysis on CCNY and NYS Mesonet sites.
"""

import datetime, math, matplotlib as mpl, matplotlib, matplotlib.pyplot as plt, numpy as np, os, pandas as pd, xarray as xr, scipy, seaborn as sns

import bin.functions, bin.lidar, bin.aux
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from cycler import cycler

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# Change font
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica Neue']})

# Define storage path for data
# storage_dir = '/Volumes/UBL Data'
storage_dir = '/Users/gabriel/Documents/urban_boundary_layer_obs/dev'

def rolling_mean(arr, n=10):
    '''
    Function to provide a rolling mean for lidar 
    Parameters
    ----------
    arr : NumPy array, 1D
        Array of data to be averaged.
    n : indices, int
        Number of positions over which the array will be averaged. The default is 10.

    Returns
    -------
    arr : NumPy array, 1D
        Averaged NumPy array.

    '''
    
    arr_ = np.cumsum(arr)
    arr_[n:] = arr_[n:] - arr_[:-n]
    return arr_[n - 1:] / n
    
    return arr

def quality_filter(data, param, sigma=2):
    ''' Adjust turbulence data distribution to adjust for outliers. '''
    
    mean, std = data[param].mean(), data[param].std()
    data = data.where((data[param] >= (mean - sigma*std)) & 
                      (data[param] <= (mean + sigma*std)), np.nan)
    
    return data

def processor(start_date='2021-05-31', end_date='2021-06-01', height=200, site='MANH', dbs=False, event='normal', stability_method='alt'):
    '''
    Perform spectral analysis for a given set of lidar and flux tower data.
    Note that data is pre-selected (lidar data from 2021-07-31 to 08-31, flux tower data from 2021-07-31 to 08-10)

    Parameters
    ----------
    height : int, optional
        Height at which the lidar data will be analyzed. The default is 200.

    Returns
    -------
    spectra : dict
        Dictionary containing stability-grouped lists containing stability grouping as the key, and the a list of normalized frequency, normalized spectra, frequency-averaged frequencies, and frequency-averaged spectra as the value.

    '''
    
    ''' Pull flux tower data from atop Marshak. '''
    # Boolean to switch Manhattan or Queens stability data.
    # Boolean to enable access to Marshak sonic anemometer data
    ccny_data, sonic_anemometer = False, False
    if site == 'MANH':
        ccny_data = True
        if height == 0:
            sonic_anemometer = True
            
    if ccny_data:
        # List pre-saved files containing relevant data
        files = ['{0}/data/flux/MANH/TOA5_10560_s20210707_e20210720_flux.dat'.format(storage_dir), 
                 '{0}/data/flux/MANH/TOA5_10560_s20210720_e20210802_flux.dat'.format(storage_dir), 
                 '{0}/data/flux/MANH/TOA5_10560_s20210802_e20210818_flux.dat'.format(storage_dir)]
        
        # Initialize list to hold DataFrames that will later be concatenated
        ts_data = []
        # Iterate through files to read the .dat files
        for file in files: 
            # Read in the .dat file. Keep 2nd row for column headings, skip the following rows (unnecessary metadata), replace all bad values with np.nan
            temp = pd.read_table(file, sep=',', header=[1], skiprows=[2, 3], na_values='NAN')
            ts_data.append(temp)
        # Concatenate the .dat file DataFrames
        ts_data = pd.concat(ts_data)
        # Adjust time index for daylight savings time (UTC-4)
        # ts_data['TIMESTAMP'] = pd.to_datetime(ts_data['TIMESTAMP']) - datetime.timedelta(hours=4)
        ts_data['TIMESTAMP'] = pd.to_datetime(ts_data['TIMESTAMP'])
        # Define Marshak building height.
        # See https://data.cityofnewyork.us/Housing-Development/Building-Footprints/nqwf-w8eh for building height data
        ts_data['z'] = bin.aux.site_info()[site]['height_agl']
        # Calculate Obukhov length. See Stull, 1988, Eq. 5.7c.
        ts_data['L'] = -ts_data['Ts_Avg'].to_numpy().flatten()*ts_data['u_star'].to_numpy().flatten()/(0.4*9.81*ts_data['Ts_Uz_cov'].to_numpy().flatten())
        # Calculate atmospheric stability parameter.
        ts_data['zeta'] = ts_data['z']/ts_data['L']
        # Calculate mean horizontal wind velocity. To be used for normalization.
        ts_data['U'] = np.sqrt(ts_data['Ux_Avg']**2 + ts_data['Uy_Avg']**2)
        ''' Pull high-frequency data - lidar or sonic anemometer. '''
        if sonic_anemometer:
            # Access pre-saved netCDF file containing an xArray of the sonic anemometer data
            sonic_data = xr.open_dataset('{0}/data/flux/MANH/ts_data_s202107310000_e202108110000_r202201180830.nc'.format(storage_dir))
            # Select data for dates of interest and convert to DataFrame
            df = sonic_data.sel(time=slice('2021-07-31', '2021-08-10')).to_dataframe()
            # Sort by time
            df = df.sort_values('time')
            # Remove MultiIndex to allow for time and height indices to become columns
            df = df.reset_index()
        else:    
            # Get Doppler-beam swinging data from the lidar files dedicated for turbulence analysis
            if dbs:
                dbs_dir_ = '{0}/data/storage/lidar'.format(storage_dir)
                # Get file list corresponding to DBS lidar data
                dbs_files = [os.path.join(dbs_dir_, file) for file in os.listdir(dbs_dir_) if 'MANH_turb_DBS' in file]
                # Get DBS lidar data
                df_dbs = xr.open_mfdataset(dbs_files).sel(site=site,
                                                          height=height).drop_vars(['ci']).to_dataframe()
                
                # Concatenate DataFrames
                df = df_dbs
            else:
                # Get lidar data from the August lidar file dedicated for spectral analysis
                lidar_data_fixed = xr.open_mfdataset(['{0}/data/storage/lidar/lidar_data_2021-08_spectral.nc'.format(storage_dir)]).sel(site='MANH', height=height)
                # Select lidar data for dates of interest and convert to DataFrame
                df = lidar_data_fixed.sel(time=slice('2021-08-01', '2021-08-10')).to_dataframe()
            # Remove MultiIndex to allow for time and height indices to become columns
            df = df.reset_index().sort_values('time')
        
        for col in df.columns:
            if col not in ['time', 'height', 'site', 'cov_u_w']:
                df = quality_filter(df, col)
    else:
        # List files containing Queens data
        date_range = [datetime.datetime.strptime(start_date, '%Y-%m-%d'),
                      datetime.datetime.strptime(end_date, '%Y-%m-%d')]
        
        file_dir = '{0}/data/flux/{1}'.format(storage_dir, site)
        files = [file for file in os.listdir(file_dir) 
                 if file.split('.')[-1] == 'csv']
        files = [file for file in files if '._' not in file]
        files = [os.path.join(file_dir, file) for file in files 
                 if (date_range[0] <= 
                     datetime.datetime.strptime(file[0:8], '%Y%m%d') <= 
                     date_range[-1])]
        # Initialize list to hold DataFrames that will later be concatenated
        ts_data = []
        # Iterate through files to read the .dat files
        for file in files: 
            # Read in the .dat file. Keep 2nd row for column headings, skip the following rows (unnecessary metadata), replace all bad values with np.nan
            temp = pd.read_table(file, sep=',', header=[0], na_values='NAN')
            ts_data.append(temp)
        # Concatenate the .dat file DataFrames
        ts_data = pd.concat(ts_data)
        # Adjust time index for daylight savings time (UTC-4)
        # ts_data['TIMESTAMP'] = pd.to_datetime(ts_data['datetime']) - datetime.timedelta(hours=4)
        # Note: DST matching removed during investigation of spectral analysis
        ts_data['TIMESTAMP'] = pd.to_datetime(ts_data['datetime'])
        # Define Queens building height.
        
        ts_data['z'] = bin.aux.site_info(site)['height_agl']
        # Drop zonal wind speed column
        ts_data = ts_data.drop(columns='U')
        # Rename wind direction column
        ts_data = ts_data.rename(columns={'WD': 'wnd_dir_compass', 'USTAR': 'u_star', 'ZL': 'zeta', 'WS': 'U', 'MO_LENGTH': 'L'})
        
        ''' Pull lidar data. '''
        # Get lidar files dedicated for turbulence analysis
        lidar_dir_ = '{0}/data/storage/lidar'.format(storage_dir)
        # Get file list corresponding to lidar data with temporal resolution sufficient for turbulence analysis
        lidar_files = [os.path.join(lidar_dir_, file) 
                       for file in os.listdir(lidar_dir_) if '{0}_turb'.format(site) in file]
        # Get lidar data from the lidar files dedicated for spectral analysis
        lidar_data = xr.open_mfdataset(lidar_files).sel(site=site, height=height)
        # Select lidar data for dates of interest and convert to DataFrame
        df = lidar_data.sel(time=slice(start_date, end_date)).to_dataframe()
        # Remove MultiIndex to allow for time and height indices to become columns
        df = df.reset_index()
        # Perform distribution check
        for col in df.columns:
            if col not in ['time', 'height', 'site']:
                df = quality_filter(df, col)
        
    # Group DataFrame into 30-minute intervals. This allows for matching with 30-minute-averaged ts_data
    df_grouped = df.groupby(pd.Grouper(key='time', freq='30T'))
    ''' Match the lidar and flux tower data to permit grouping by stability. '''
    # Initialize list of modified DataFrames to be concatenated.
    dfs = []
    # For the 30-minute binned DataFrames, add columns with data from the flux tower data.
    for label, data in df_grouped:
        # Match lidar and flux tower data by timestamp
        match = ts_data.loc[ts_data['TIMESTAMP'] == label].drop_duplicates()
        # Filter out necessary columns for spectra normalization
        match = match[['z', 'U', 'u_star', 'zeta', 'wnd_dir_compass', 'L']]
        # If the matched DataFrame column is empty, fill with np.nan
        for col in match.columns:
            if match[col].empty:
                match[col] = [np.nan]
        # Append the matching value to the DataFrame as a column for each parameter needed for normalization.
        data['z'] = np.repeat(match['z'].values, len(data))
        try:
            # For non-CCNY sites that have all 3 wind components
            data['U'] = np.sqrt(data['u']**2 + data['v']**2)
        except:
            # For CCNY
            data['U'] = np.repeat(match['U'].values, len(data))
        data['u_star'] = np.repeat(match['u_star'].values, len(data))
        data['zeta'] = np.repeat(match['zeta'].values, len(data))
        data['L'] = np.repeat(match['L'].values, len(data))
        data['wind_direction_surface'] = np.repeat(match['wnd_dir_compass'].values, len(data))
        data['w_prime'] = data['w'] - data['w'].mean()
        # Append the matched DataFrame to the list of DataFrames
        dfs.append(data)
    # Concatenate the matched DataFrames
    dfs = pd.concat(dfs).sort_values('time')
    # Filter out by wind direction - northerlies filtered out
    if site == 'MANH':
        dfs = dfs.where((dfs['wind_direction_surface'] >= 90) & (dfs['wind_direction_surface'] <= 270))
    elif site == 'QUEE':
        dfs = dfs.where((dfs['wind_direction_surface'] >= 180) & (dfs['wind_direction_surface'] < 360))
    # Remove null times to save data
    dfs = dfs.dropna(subset=['time'])
    
    return dfs

def spectral_analysis(dfs, param='w', sonic_anemometer=False):
    '''
    Perform spectral analysis on a DataFrame containing 

    Parameters
    ----------
    dfs : Pandas Dataframe
        Dataframe containing directional wind, the fluctuating quantity, height, and stability - at least.
    param : str, optional
        Name of the direction the analysis is in. The default is w.
    sonic_anemometer : bool, optional
        Boolean to control sampling frequency. The default is False.

    Returns
    -------
    spectra : NumPy arrays
        Fourier-transformed arrays containing spectral data from velocities.

    '''
    
    # Get averaged lidar data sampling frequency in Hz
    if sonic_anemometer:
        dt = 10
    else:
        dt = 1
    
    norm = 'u_star'
    
    # Define stability groups
    bins = [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf]
    # Group the matched DataFrames by stability classification
    dfs_grouped_zeta = dfs.groupby(pd.cut(dfs['zeta'], bins))
    
    ''' Perform spectral analysis for each stability grouping. '''
    # Initialize dictionary to hold all grouped spectra
    spectra = {}
    # Iterate over every stability grouping
    n = 0
    for label, data in dfs_grouped_zeta:
        # Filter out all nan data
        data = data[data['w_prime'].notna()]
        # Number of sample points 
        N = len(data['w'])
        # Get averaged lidar data sampling frequency
        T = 1/dt
        # Get fast Fourier transform frequencies (negative frequencies filtered out)
        if N <= 0:
            continue
        x = fftfreq(N, T)
        y = 2*(fft(data['w_prime'].values)**2)/N/dt
        # Get normalized frequencies. 
        # Indexing changes based on whether N is even or odd, see https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
        if N % 2 == 0:
            # Incorporate zero-plane displacement for comparison with Feigenwinter et al. (1999)
            x_norm = (x * data['z'] / data['U'])[:N//2]
        else:
            x_norm = (x * data['z'] / data['U'])[:(N-1)//2]
        # Perform fast Fourier transform with normalized data
        try:
            y_norm = np.abs(x * y / (data[norm]).values**2)[:N//2]
        except:
            y_norm = np.full(x_norm.shape, fill_value=np.nan)
        
        ''' Get timescale from normalized frequency peak. '''
        # Get index of peak normalized frequency
        index = np.argmax(y_norm)
        umax = data['U'][:N//2].values[index] if N % 2 == 0 else data['U'][:(N-1)//2].values[index]
        # Get timescale of peak normalized frequency
        timescale = data['height'].unique()[0]/(umax * x_norm.values[index])
        print('Stability group: {0} at height {1} m has a dominant mixing timescale of {2:.2f} s.'.format(label, data['height'].unique()[0], timescale))
        
        ''' Frequency band averaging. '''
        # Define number of frequency bands
        N_bands = 64
        try:
            # Define frequency bands
            freq_bands = np.logspace(np.log10(sorted(x_norm)[1]), np.log10(np.nanmax(x_norm)), N_bands)
            
            # Prepend a 0 to prevent a nonzero first element
            # freq_bands = np.insert(freq_bands, 0, 0)
            # Average frequency over defined frequency bands
            freq_avgd = [np.nanmean(x_norm[(x_norm >= freq_bands[i]) & (x_norm < freq_bands[i+1])]) for i in np.arange(0, len(freq_bands)-1)]
            # Average spectra over defined frequency bands
            arr_avgd = [np.nanmean(y_norm[(x_norm >= freq_bands[i]) & (x_norm < freq_bands[i+1])]) for i in np.arange(0, len(freq_bands)-1)]
            # Spectra std over defined frequency bands
            arr_std = [np.nanstd(y_norm[(x_norm >= freq_bands[i]) & (x_norm < freq_bands[i+1])]) for i in np.arange(0, len(freq_bands)-1)]
            
            # Remove outliers
            arr_avgd_ = np.array(arr_avgd)
            m, s = np.nanmean(arr_avgd_), np.nanstd(arr_avgd_)
            # Filter out elements outside of 2-sigma
            filter_ = np.array((arr_avgd < (m + 2*s)) & (arr_avgd > (m - 2*s)))
            # Filter all arrays by index
            freq_avgd = list(np.array(freq_avgd)[filter_])
            arr_avgd = list(np.array(arr_avgd)[filter_])
            arr_std = list(np.array(arr_std)[filter_])
            # Rolling mean for all arrays
            freq_avgd = rolling_mean(freq_avgd)
            arr_avgd = rolling_mean(arr_avgd)
            arr_std = rolling_mean(arr_std)
            
        except:
            freq_avgd = np.full(y_norm.shape, fill_value=32)
            arr_avgd = np.full(y_norm.shape, fill_value=32)
            arr_std = np.full(y_norm.shape, fill_value=32)
            
        
        spectra[label] = [x_norm, y_norm, freq_avgd, arr_avgd, arr_std, x, y]
        
        n += 1
    
    return spectra

def spectral_plotter(data, heights):
    '''
    Plot a list of spectral data for a series of heights.

    Parameters
    ----------
    data : list
        List of dictionaries containing spectral data from the proc() function.
    heights : list
        List of heights at which spectra were obtained for.
    '''
    
    # Check to make sure that the data input matches the heights analyzed
    if len(data) != len(heights):
        print('Check your inputs, the data list and height list should be equal...')
    
    # Interpolation function per Kaimal et al. (1972), Equation 7
    def interpolation(x, y):
        polyfit = np.polyfit(np.log(x), y, 2)
        return polyfit
    
    # Curve fitting - Kaimal spectrum per Larsen (2016) as written in Cheynet (2017)
    # def func(x, a, b, c, d):
    #     return a*x/((1+b*x)**(5/3)) + c*x/(1+d*x**(5/3))
    
    
    # Curve fitting - implementation
    def kaimal_spectrum(x, y):
        # Filter out nans from both arrays
        x_ = x[~np.isnan(x) & ~np.isnan(y)]
        y_ = y[~np.isnan(x) & ~np.isnan(y)]
        # Get curve fitting metadata
        popt, pcov = curve_fit(func, x_, y_, maxfev=10000)
        return x_, y_, popt, pcov
    
    # Define figure parameters
    ncols = 5
    # nrows = len(data[0]) // ncols
    nrows = 1
    # Initialize figure
    fig, axs = plt.subplots(figsize=(8, 3), dpi=300, nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    # Iterate over subplot to plot.
    # Outer loop iterates over stability groups.
    for i, ax in enumerate(axs.reshape(-1)):
        # Obtain name of stability group
        key = list(data[0].keys())[i]
        # Iterate over heights
        for j in range(0, len(data)):
        
            # Obtain frequency-averaged frequency bins and spectra
            x, y, s, f = data[j][key][2], data[j][key][3], data[j][key][4], data[j][key][-2]
            # Plot the data
            im = ax.loglog(x[:-1], y[:-1], lw=3, label='{0} m'.format(heights[j]))
            std = ax.fill_between(x, np.array(y)-0.75*np.array(s), np.array(y)+0.75*np.array(s), alpha=0.1, label='_nolegend_')
            
            # Plot spectral reference data for lowest level
            # x_, y_, popt, _ = kaimal_spectrum(x, y)
            # x_is, y_is = x_, func(x, popt[0], popt[1], key)
            # ax.loglog(x_is, y_is, color='k', linestyle='--', lw=2, 
            #           zorder=10, label='_nolegend_')
            
            xlim, ylim = [10e-4, 10e1], [10e-5, 10e0]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            locmaj = matplotlib.ticker.LogLocator(base=10.0, numticks=6) 
            locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.1), numticks=12)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_major_locator(locmaj)
            
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(which='major', linestyle=':')
            
            ax.set_title('$\zeta$ = {0}'.format(key), fontsize=10)
            # Control where y-label is set
            if i % ncols == 0:
                ax.set_ylabel('f S$_w$ / $\overline{U}^2$', labelpad=15, fontsize=10)
    
    hand, labl = ax.get_legend_handles_labels()
    fig.legend(labels=labl, loc='upper center', ncol=len(heights), bbox_to_anchor=(0.54, 1.15), fontsize=10, frameon=False)
    fig.supxlabel('f z / $\overline{U}$', fontsize=10, y=0.05)
        
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    
    # Post-script log-label removal
    for i, ax in enumerate(axs.reshape(-1)):
        n = 2
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]

    print([item.get_text() for item in ax.get_xticklabels()])

def stats(df, spectra):
    
    ''' Method to collect and visualize data statistics. '''
    
    # Simple histogram
    fig, ax = plt.subplots(dpi=300)
    param = 'U'
    im = sns.histplot(data=df, x=param, bins=50)
    ax.set_title('{0} distribution'.format(param))
    
    # Grouped histogram by stability
    df['zeta_bin'] = pd.cut(df['zeta'], [-np.inf, -2, -1, -0.1, 0, 0.1, 1, 2, np.inf])
    dfs = df.iloc[df['zeta_bin'].cat.codes.argsort()]
    len_ = len(dfs['zeta_bin'].unique())
    nrows, ncols = len_ // 3, len_ // 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    axs = ax.reshape(-1)
    i = 0
    for bin_ in dfs['zeta_bin'].unique():
        dfs_ = dfs.loc[dfs['zeta_bin'] == bin_]
        sns.histplot(data=dfs_, x='U', bins=20, ax=axs[i])
        axs[i].set_title(bin_)
        i += 1
    fig.tight_layout()
    
    # Grouped histogram by direction
    df['wind_direction_bin'] = pd.cut(df['wind_direction'], [0, 45, 90, 135, 180, 225, 270, 315, 360])
    dfs = df.iloc[df['wind_direction_bin'].cat.codes.argsort()]
    len_ = len(dfs['wind_direction_bin'].unique())
    nrows, ncols = 3, 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    axs = ax.reshape(-1)
    i = 0
    for bin_ in dfs['wind_direction_bin'].unique():
        dfs_ = dfs.loc[dfs['wind_direction_bin'] == bin_]
        sns.histplot(data=dfs_, x='zeta', bins=20, ax=axs[i])
        axs[i].set_title(bin_)
        i += 1
    fig.tight_layout()
    
    return dfs

def turbulence_stats(data):
    '''
    Calculate common turbulence statistics for a DataFrame of high-resolution data.
    '''
    
    # Clean the data by removing unnecessary columns
    try:
        data = data.drop(columns=['ci', 'u_std', 'v_std', 'w_std', 'wind_direction_std', 'ci_std'])
    except:
        pass
    
    # Define interval for time grouping
    interval = '30T'
    # Generate list of DataFrames to be rejoined
    dfs = []
    # Generate grouped data based on height
    height_groups = data.groupby('height')
    # Iterate through each group, calculatindata.gg defined turbulence statistics
    for height, height_data in height_groups:
        # Group by given time interval
        time_groups = height_data.groupby(pd.Grouper(key='time', freq=interval))
        # Iterate through each time interval
        for group, subdata in time_groups:
            # Try/except used to accommodate fixed-stare lidar data
            try:
                # Generate turbulent intensity for u, per Stull (1988)
                subdata['I_u'] = np.sqrt(subdata['var_u']) / np.nanmean(subdata['U'])
                # Generate turbulent intensity for v, per Stull (1988)
                subdata['I_v'] = np.sqrt(subdata['var_v']) / np.nanmean(subdata['U'])
                # Generate turbulent intensity for w, per Stull (1988)
                subdata['I_w'] = np.sqrt(subdata['var_w']) / np.nanmean(subdata['U'])
            except:
                # Generate turbulent intensity for w, per Stull (1988)
                subdata['I_w'] = np.sqrt(subdata['var_w']) / np.nanmean(subdata['U'])
            # Append to list
            dfs.append(subdata)
    # Concatenate all the groups
    data_ = pd.concat(dfs).sort_values('time')
    
    return data_

def stability_plots(datasets, sites=['QUEE', 'STAT'], stability_method='surface', param='I_u', norm=None, std_plot=True, exp=None, height_norm=True, comp=False):
    '''
    Function to plot data from a DataFrame as grouped by stability.

    Parameters
    ----------
    datasets : list
        List of Pandas DataFrames with pre-calculated turbulent properties. This input should come from output from 'turbulence_stats'.
    param : str, optional
        Parameter desired for plotting. The default is 'I_u'.
    std_plot : bool, optional
        Boolean to control whether or not standard deviations are plotted. The default is True.
        
    '''
    
    param_dict = {'u': {'long_name': 'Wind velocity, zonal',
                        'symbol': 'u'},
                  'v': {'long_name': 'Wind velocity, meridional',
                        'symbol': 'v'},
                  'w': {'long_name': 'Wind velocity, vertical',
                        'symbol': 'w'},
                  'I_u': {'long_name': 'Turbulent intensity, zonal',
                          'symbol': 'I$_u$'},
                  'I_v': {'long_name': 'Turbulent intensity, meridional',
                          'symbol': 'I$_v$'},
                  'I_w': {'long_name': 'Turbulent intensity, vertical',
                          'symbol': 'I$_w$'},
                  'var_u': {'long_name': 'Velocity variance, zonal',
                            'symbol': '$\sigma_u^2$'},
                  'var_v': {'long_name': 'Velocity variance, meridional',
                            'symbol': '$\sigma_v^2$'},
                  'var_w': {'long_name': 'Velocity variance, vertical',
                            'symbol': '$\sigma_w^2$'},
                  'w_star': {'long_name': 'Convective velocity scale',
                            'symbol': 'w*'},
                  'w_star_surf': {'long_name': 'Convective velocity scale',
                            'symbol': 'w$_s$*'},
                  'e': {'long_name': 'Turbulent kinetic energy',
                            'symbol': 'TKE'}}
        
    # Convert the site data to list form if it's not already
    if type(sites) is not list:
        sites = [sites]
    
    # Define stability group bins
    bins = [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf]
    # Define colors and markers for plotting
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', 'x']
    # Initialize figures
    fig, axs = plt.subplots(ncols=len(bins)-1, sharey=True, dpi=300)
    
    # Initialize plot x-limits
    vmin, vmax = 0, 0

    site_num = 0
    
    for site, site_data in datasets.groupby('site'):
        # Ensure data is only shown for the chosen sites
        if site not in sites:
            continue
        
        site_data = [site_data]
        for i, data in enumerate(site_data):
            # Initialize dictionaries for intensity mean, standard deviation values, and stability group names
            means, stds, groups = {}, {}, None
            
            # Initialize bins for normalized heights
            height_bins = []
            # Normalize heights either by (a) mixed layer height or (b) roughness height for comparison with Roth (2000)
            if height_norm:
                if comp and param in ['I_u', 'I_v', 'I_w']:
                    # Normalize height by area-averaged roughness height
                    zH = bin.aux.area_averaged_height(site)
                    data['height'] = data['height']/zH
                else:
                    data['height'] = data['height']/data['mixing_height']
                    
            # Group by height and iterate over each height.
            # Conditional statement handles binned normalized heights.
            if height_norm and not comp:
                height_groups = data.groupby('height')
            else:
                height_bins = np.linspace(0, 1, 21)
                height_groups = data.groupby(pd.cut(data['height'], height_bins))
                # height_groups = data.groupby('height')
                
            # Iterate over each height group.
            for height, height_data in height_groups:
                # Group by zeta
                stability_groups = height_data.groupby(pd.cut(height_data['zeta'], bins))
                # Save group labels for stability groupings
                groups = list(stability_groups.groups.keys())
                # Collect means and standard deviations at each height
                mean, std = [], []
                # Iterate over the stability groups
                for stability, subdata in stability_groups:
                    # Append mean and standard deviation to respective dictionaries
                    # If normalization is enabled, do so here. Else, use the non-normalized data.
                    if norm:
                        mean.append((subdata[param]/subdata[norm]**exp).mean())
                        # std.append((subdata[param]/subdata[norm]**exp).std())
                        std.append((subdata[param]/subdata[norm]**exp).quantile(q=0.25))
                    else:
                        mean.append((subdata[param]).mean())
                        # std.append((subdata[param]).std())
                        std.append((subdata[param]).quantile(q=0.25))
                # Append to dict
                if height_norm and comp:
                    means[height.right], stds[height.right] = mean, std
                else:
                    means[height], stds[height] = mean, std
            # Generate DataFrame for mean and standard deviations
            means = pd.DataFrame.from_dict(means, 
                                           orient='index', 
                                           columns=groups)
            stds = pd.DataFrame.from_dict(stds,
                                          orient='index', 
                                          columns=groups)
            
            # Get x-axis extrema
            if std_plot:
                if (np.nanmin(means - stds)) < vmin:
                    vmin = np.nanmin(means) - np.nanmin(stds)
                if (np.nanmax(means + stds)) > vmax:
                    vmax = np.nanmax(means) + np.nanmax(stds)
            else:
                if np.nanmin(means) < vmin:
                    vmin = np.nanmin(means)
                if np.nanmax(means) > vmax:
                    vmax = np.nanmax(means)
                
            # Plot the turbulent intensities by stability group
            for j, group in enumerate(groups):
                label = site if j == 0 else None
                # Plot the intensity mean values
                axs[j].scatter(means[group], means.index, 
                               color=colors[site_num], marker=markers[site_num], label=label)
                if std_plot:
                    # Plot the intensity standard deviations
                    axs[j].errorbar(means[group], 
                                    stds.index, 
                                    xerr=stds[group],
                                    color=colors[site_num],
                                    fmt='none',
                                    capsize=3)
                axs[j].set_xlim([vmin, vmax])
                axs[j].set_xscale('log')
                axs[j].set_title(group)
                if comp:
                    if param in ['I_u', 'I_v', 'I_w']:
                        y = np.arange(0, 200, 0.1)
                        if param == 'I_u':
                            x = 0.259 + 0.582*np.exp(-0.943*y)
                        elif param == 'I_v':
                            x = 0.163 + 0.391*np.exp(-0.563*y)
                        elif param == 'I_w':
                            x = 0.114 + 0.226*np.exp(-0.634*y)
                        axs[j].plot(x, y, c='k', lw=2)
                        # print(x, y)
                        axs[j].set_ylim([0, 25])
                    elif param in ['var_u', 'var_v', 'var_w']:
                        y = np.arange(0.05, 1, 0.01)
                        if param == 'var_u':
                            x = [(0.74**2)*(0.5*(2-np.sqrt(y)) + 0.3*np.sqrt(y))]
                        if param == 'var_w':
                            x = [1.17*(y*(1-y))**(2/3), 
                                 1.8*(y**(2/3))*(1 - 0.8*y)**2]
                            labels = ['Lenschow et al. (1980)', 'Sorbjan (1989)']
                            axs[j].set_xlim([1e-3, 1])
                            axs[j].set_xticks(np.logspace(-2, 0, 3))
                        for k, x_ in enumerate(x):
                            axs[j].plot(x[k], y, lw=2, ls='--', alpha=0.5)
                        # axs[j].set_xticks(np.linspace(0.1, 1, 10))
                        axs[j].grid(which='both', alpha=0.25, zorder=0)
                axs[j].set_ylim([0, 1])
                label_size = 12
                # Set the x-label
                if norm:
                    if j == 2:
                        axs[j].set_xlabel(r'{0} / {1}$^{2}$'.format(param_dict[param]['symbol'], param_dict[norm]['symbol'], exp), fontsize=label_size)
                else:
                    if j == 2:
                        axs[j].set_xlabel(r'{0}'.format(param_dict[param]['symbol']), labelpad=20, fontsize=label_size)
                # Only print the y-label on the first subplot
                if j == 0:
                    if height_norm:
                        if comp and param in ['I_u', 'I_v', 'I_w']:
                            axs[j].set_ylabel('z / z$_H$', labelpad=15, fontsize=label_size)
                        else:
                            axs[j].set_ylabel('z / z$_i$', labelpad=15, fontsize=label_size)
                    else:
                        axs[j].set_ylabel('Height [m]', labelpad=15, fontsize=label_size)
                        
        site_num += 1 

    if param_dict[param]:
        fig.suptitle(param_dict[param]['long_name'], y=1.05)
    else:
        fig.suptitle(param, y=1.05)
        
    fig.legend(ncol=len(sites), frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1))
        
    fig.tight_layout()
  
def data_merge(site='QUEE', heights=None, param='w'):
    ''' Generates fit and plots comparing normalized standard deviations versus stability. '''
    
    import scipy.optimize, scipy.stats, matplotlib
    
    # Get turbulence data
    dirname = '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/storage'
    ds = pd.concat([pd.read_csv(os.path.join(dirname, file)) for file in os.listdir(dirname) if 'turbulence_data_{0}'.format(site.lower()) in file])
    
    # Get flux data
    dirname = '/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/flux/{0}'.format(site)
    if site in ['BRON', 'QUEE', 'STAT']:
        files = [os.path.join(dirname, file) for file in os.listdir(dirname) 
                 if 'csv' in file]
        fluxes = []
        for file in files:
            df = pd.read_csv(file)
            fluxes.append(df)
        fluxes = pd.concat(fluxes).sort_values('datetime').rename(columns={'datetime': 'time'})
        fluxes['Ts'] = fluxes['Tc'] + 273.15
        fluxes['cov_w_T'] = fluxes['H']/1004
        #fluxes['w_star'] = ((9.81/fluxes['Ts'])*(fluxes['cov_w_T'])*44.6)**(1/3)
        fluxes = fluxes.loc[(fluxes['time'] >= '2020-09-01') & (fluxes['time'] < '2021-09-01')]
    else:
        files = [os.path.join(dirname, file) for file in os.listdir(dirname) 
                 if file.split('.')[-1] == 'dat']
        fluxes = []
        for file in files:
           temp = pd.read_csv(file, header=1, skiprows=[2, 3], na_values='NAN')
           fluxes.append(temp)
        fluxes = pd.concat(fluxes)[['TIMESTAMP', 'Ts_Avg', 'Ts_Uz_cov', 'wnd_dir_compass', 'u_star']] 
        fluxes['TIMESTAMP'] = pd.to_datetime(fluxes['TIMESTAMP'])
        fluxes['z'] = bin.aux.site_info()['MANH']['height_agl']
        fluxes['L'] = -fluxes['Ts_Avg'].to_numpy().flatten()*fluxes['u_star'].to_numpy().flatten()/(0.4*9.81*fluxes['Ts_Uz_cov'].to_numpy().flatten())
        fluxes['zeta'] = fluxes['z']/fluxes['L']
        fluxes = fluxes.rename(columns={'TIMESTAMP': 'time', 'Ts_Uz_cov': 'cov_w_T', 'Ts_Avg': 'Ts'})
    
    # Get mixing heights
    if site in ['BRON', 'QUEE', 'STAT']:
        mixing_height = pd.read_csv('/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/storage/mixing_heights_{0}_s20190901_e20210901.csv'.format(site.lower()))
        merged = ds.merge(fluxes, on=['time'], suffixes=('', '_flux')).merge(mixing_height, on=['time'], suffixes=('', '_mh'))
    else:
        # Average over all sites - MUST BE JUSTIFIED WITH LITERATURE REVIEW
        mixing_heights = []
        for profiler_site in ['BRON', 'QUEE', 'STAT']:
            mixing_height = pd.read_csv('/Users/gabriel/Documents/urban_boundary_layer_obs/dev/data/storage/mixing_heights_{0}_s20190901_e20210901.csv'.format(profiler_site.lower()))
            mixing_heights.append(mixing_height)
        # Perform the site-averaging for each timestamp
        mhs = []
        for group, group_data in pd.concat(mixing_heights).groupby('time'):
            mh = pd.DataFrame(columns=group_data.columns).drop(columns={'Unnamed: 0'})
            mh['mixing_height'] = [group_data['mixing_height'].mean()]
            mh['site'] = 'MANH'
            mh['time'] = group
            mhs.append(mh)
        mixing_height = pd.concat(mhs)
    
    # Convert time column to datetime
    ds['time'] = pd.to_datetime(ds['time'])
    fluxes['time'] = pd.to_datetime(fluxes['time'])
    mixing_height['time'] = pd.to_datetime(mixing_height['time'])
    
    # Drop the stability values in the original dataset, re-pull to accommodate sonic data
    if site == 'MANH':
        ds = ds.drop(columns=['zeta'])
    merged = ds.merge(fluxes, on=['time'], how='outer', suffixes=('', '_flux'))
    merged['time'] = pd.to_datetime(merged['time'])
    merged = merged.merge(mixing_height, on=['time'], suffixes=('', '_mh'))
    merged['time'] = pd.to_datetime(merged['time'])
    
    # Calculate normalized height relative to area-averaged roughness height
    merged['z_zh'] = merged['height']/bin.aux.site_info()[site]['height_agl']
    
    # Calculate convective velocity scale
    merged['w_star'] = ((9.81/merged['Ts'])*(merged['cov_w_T'])*merged['mixing_height'])**(1/3)
    # Calculate height-adjusted convective scale
    merged['cov_w_T_surface'] = merged['cov_w_T']/(1-1.2*merged['height']/merged['mixing_height'])
    merged['w_star_surf'] = ((9.81/merged['Ts'])*(merged['cov_w_T_surface'])*merged['mixing_height'])**(1/3)
    
    # Prevent irrational values during normalization
    merged['w_star'] = merged['w_star'].where(~np.isnan(merged['w_star']) &
                                          (merged['w_star'].abs() > 0), np.nan)
    merged['w_star_surf'] = merged['w_star_surf'].where(~np.isnan(merged['w_star_surf']) &
                                          (merged['w_star_surf'].abs() > 0), np.nan)
        
    return merged
      
def var_zeta(merged, site='QUEE', param='w', stability='unstable'):
    
    # Comparison dictionary
    # Holds coefficients for stability functions from various publications
    coeffs = pd.read_csv('bin/assets/similarity_coefficients.csv')
    
    # Narrow DataFrame to site-specific data
    merged = merged.loc[merged['site'] == site]
    
    # Define roughness-height normalized heights - arbitrary bounds
    heights = merged.groupby(pd.cut(merged['z_zh'], 
                                    bins=np.arange(merged['z_zh'].min(), merged['z_zh'].max(), 5), include_lowest=True))
        
    for height, data in heights:
        # Choose stability regime
        if stability == 'unstable':
            mask = np.where((data['zeta'] <= -0.2), True, False)
        else:
            mask = np.where((data['zeta'] > 0.1), True, False)
        
        # Get data
        x = data[mask]['zeta'].values
        y = np.sqrt(data[mask]['var_{0}'.format(param)].values)/data[mask]['u_star'].values
        
        ''' Prep curve fit. '''
        # Filter out nan values
        vector_mask = np.where(np.isnan(x) | np.isnan(y), False, True)
        x = x[vector_mask]
        y = y[vector_mask]
        
        if len(x) == 0:
            continue
        
        # Actual data
        fig, ax = plt.subplots(figsize=(5, 3))
        
        p = ax.scatter(x, y, s=5, c='lightgrey', marker='x')
        
        try:
            # Define function
            def func(x, a, b):
                return a*(1-b*x)**(1/3)
            # Curve fit
            popt, pcov = scipy.optimize.curve_fit(func, x, y, bounds=((-10, -10), (10, 100)))
            y_fit = popt[0]*(1-popt[1]*x)**(1/3)
            _, _, r, _, _ = scipy.stats.linregress(y, y_fit)
            # # Actual curve fit
            fit = ax.scatter(x, y_fit, c='k', s=5, 
                              label='Fit [a = {0:.2f}, b = {1:.2f}]'.format(*popt))
        except:
            continue
        
        ''' Load reference plots. '''
        # Load coefficients and filter by input arguments
        coeffs = coeffs.loc[(coeffs['z_zh'] >= height.left) &
                            (coeffs['z_zh'] <= height.right) &
                            (coeffs['param'] == param) &
                            (coeffs['stability'] == stability)]
        
        print(coeffs)
        
        for _, values in coeffs.iterrows():
            comp = ax.plot(np.sort(x), values['a']*(1-values['b']*np.sort(x))**(values['c']), lw=2, label=values['reference'])
        
        ax.set_xscale('symlog')
        ax.legend(frameon=False, loc='upper right')
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylabel('$\sigma_{0} \  / \ u_*$'.format(param), labelpad=15)
        ax.set_xlabel('$\zeta$', labelpad=10)
        ax.set_ylim([0, 5])
        ax.set_title('{0}, z/z$_H$ = {1}'.format(site, height))
        plt.gca()

def stability_distribution(ds, height=100):
    '''
    Plot the stability count per hour to show the distribution of the dataset when future data is grouped by stability bins.

    Parameters
    ----------
    ds : Pandas DataFrame
        List of DataFrames containing basic output data from the processor function.
    height : int, optional
        Height at which stability data is being counted. The default is 100.
    '''    
    
    if type(ds) is not list:
        ds = [ds]
    
    counts = {key: {} for key in ds.site.unique()}
    bins = [-np.inf, -0.1, 0.1, np.inf]
    for site_key, site in ds.groupby('site'):
        for hour_key, hour in site.groupby(site['time'].dt.hour):
            counts[site_key][hour_key] = {}
            for key, group in hour.groupby(pd.cut(hour['zeta'], bins=bins)):
                print(site_key, hour_key, key)
                counts[site_key][hour_key][str(key)] = group['zeta'].count()/hour['zeta'].count()
    cdfs = {key: pd.DataFrame.from_dict(counts[key], orient='index') for key in counts.keys()}
    
    ''' Plotting. '''
    fig, axs = plt.subplots(figsize=(4, 5), dpi=300, nrows=3, sharex=True)
    sites = sorted(ds.site.unique())
    hatches = ['//', 'xx', 'oo', '**']
    for i, ax in enumerate(fig.axes):
        im = cdfs[sites[i]].plot(kind='bar', ax=ax, stacked=True, cmap='coolwarm_r', legend=legend, edgecolor=(0, 0, 0, 0.5))
        bars = [bar for bar in im.containers]
        [patch.set_hatch(hatches[j]) for j, bar in enumerate(bars) for patch in bar]
        ax.set_title(sites[i], pad=10)
        ax.set_ylim([0, 1])
        ax.xaxis.set_tick_params(rotation=0)
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % 3 != 0]
    fig.supxlabel('Hour of day (UTC)', x=0.575)
    fig.supylabel('Occurrence fraction')
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[0:len(cdfs)], labels[0:len(cdfs)], frameon=False, bbox_to_anchor=(1.3, 0.94))

def main(start_date, end_date, site='MANH', heights=[200, 400, 600, 800], event='normal', plot_spectra=False):
    
    # Initialize list of DataFrames that will be concatenated
    data, spectra = [], {}
    # Iterate over list of heights
    for height in heights:
        # Load lidar and/or anemometer data
        df = processor(start_date, end_date, height=height, site=site, event=event)
        data.append(df)
        if plot_spectra:
            spectra[height] = spectral_analysis(df)
    
    # Concatenate data
    data = pd.concat(data)
    # Adjust for local time (UTC-4)
    data['time'] = data['time'] - datetime.timedelta(hours=4)
    # Plot spectra
    if plot_spectra:
        spectral_plotter(list(spectra.values()), list(spectra.keys()))
        
    # Define heat wave and normal dates for turbulence analysis
    normal_dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate turbulence data for the whole time range given
    # turbulence_normal = turbulence_stats(data[(data['time'] >= normal_dates[0]) & 
    #                                           (data['time'] <= normal_dates[-1])])
    
    return data, spectra


if __name__ == '__main__':
    print('Running...')
    
    # Boolean to determine whether data will be computed or not when script run
    compute = False
    
    if compute:
        # Define date rangebint
        start_date, end_date = ['2021-07-15', '2021-08-10']
        # Define height range
        heights, step = [200, 1500], 300
        height_arr = np.arange(heights[0], heights[-1]+step, step)
        # Collection list to be concatenated later
        # Generate data for a given site and heights
        output, turbulence_data = main(start_date, end_date, site='MANH', heights=[200, 300, 500, 1000, 1500], event='normal', plot_spectra=True)