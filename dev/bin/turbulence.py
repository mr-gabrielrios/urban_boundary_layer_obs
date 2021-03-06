"""
Urban Boundary Layer Observation Data Processing
Script name:    Turbulence Analysis
Path:           ~/bin/turbulence.py
Description:    Perform turbulence analysis on CCNY and NYS Mesonet sites.
"""

import datetime, math, matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, os, pandas as pd, xarray as xr, scipy, seaborn as sns

import bin.functions, bin.lidar
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from cycler import cycler

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# Change font
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# Define storage path for data
storage_dir = '/Volumes/UBL Data'

def rolling_mean(arr, n=10):
    '''
    Function to provide a rolling mean for lidar data.

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
        ts_data['z'] = 50.6
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
            # Get lidar data from the August lidar file dedicated for spectral analysis
            lidar_data_fixed = xr.open_mfdataset(['{0}/data/storage/lidar/lidar_data_2021-08_spectral.nc'.format(storage_dir)]).sel(site='MANH', height=height)
            # Select lidar data for dates of interest and convert to DataFrame
            df = lidar_data_fixed.sel(time=slice('2021-08-01', '2021-08-10')).to_dataframe()
            
            # Get Doppler-beam swinging data from the lidar files dedicated for turbulence analysis
            if dbs:
                dbs_dir_ = '{0}/data/storage/lidar'.format(storage_dir)
                # Get file list corresponding to DBS lidar data
                dbs_files = [os.path.join(dbs_dir_, file) for file in os.listdir(dbs_dir_) if 'MANH_DBS' in file]
                # Get DBS lidar data
                df_dbs = xr.open_mfdataset(dbs_files).sel(site=site,
                                                          height=height).drop_vars(['ci']).to_dataframe()
                
                # Concatenate DataFrames
                df = pd.concat([df, df_dbs])
                
            # Remove MultiIndex to allow for time and height indices to become columns
            df = df.reset_index().sort_values('time')
            
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
        ts_data['z'] = 44.6
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
        data['U'] = np.sqrt(data['u']**2 + data['v']**2)
        data['u_star'] = np.repeat(match['u_star'].values, len(data))
        data['zeta'] = np.repeat(match['zeta'].values, len(data))
        data['L'] = np.repeat(match['L'].values, len(data))
        data['wind_direction_surface'] = np.repeat(match['wnd_dir_compass'].values, len(data))
    
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
    
    # Define stability groups
    bins = [-np.inf, 0, 0.2, np.inf]
    # Group the matched DataFrames by stability classification
    dfs_grouped_zeta = dfs.groupby('stability')
    
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
            # x_norm = (x * data['z'] / data['U'])[:N//2]
            # Incorporate zero-plane displacement for comparison with Feigenwinter et al. (1999)
            x_norm = (x * data['z'] / data['U'])[:N//2]
        else:
            x_norm = (x * data['z'] / data['U'])[:(N-1)//2]
        # Perform fast Fourier transform with normalized data
        try:
            y_norm = np.abs(x * y / (data['U']).values**2)[:N//2]
        except:
            y_norm = np.full(x_norm.shape, fill_value=np.nan)
        
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
    def func(x, a, b, c, d):
        return a*x/((1+b*x)**(5/3)) + c*x/(1+d*x**(5/3))
    
    # Curve fitting - implementation
    def kaimal_spectrum(x, y):
        # Filter out nans from both arrays
        x_ = x[~np.isnan(x) & ~np.isnan(y)]
        y_ = y[~np.isnan(x) & ~np.isnan(y)]
        # Get curve fitting metadata
        popt, pcov = curve_fit(func, x_, y_, maxfev=100000)
        return x_, y_, popt, pcov
    
    # Define figure parameters
    ncols = 3
    nrows = len(data[0]) // ncols
    # Initialize figure
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300, nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    # Iterate over subplot to plot.
    # Outer loop iterates over stability groups.
    for i, ax in enumerate(ax.reshape(-1)):
        # Obtain name of stability group
        key = list(data[0].keys())[i]
        # Iterate over heights
        for j in range(0, len(data)):
            # Obtain frequency-averaged frequency bins and spectra
            x, y, s, f = data[j][key][2], data[j][key][3], data[j][key][4], data[j][key][-2]
            # Plot the data
            im = ax.loglog(x, y, lw=3, label='{0} m'.format(heights[j]))
            std = ax.fill_between(x, np.array(y)-0.75*np.array(s), np.array(y)+0.75*np.array(s), alpha=0.1)
            # Plot spectral reference data for lowest level
            if ((i+1) != nrows*ncols) and j == 0:
                # Inertial subrange frequency range
                x_is = np.linspace(10e-1, 10e0, 2)
                y_is = 0.6*(x_is ** (-5/3))
                im_is = ax.loglog(x_is, y_is, 
                                  color='k', linestyle='--', lw=2, 
                                  zorder=10, label='_nolegend_')
            
            xlim, ylim = [10e-4, 10e1], [10e-7, 10e1]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(which='both', linestyle=':')
            
            ax.set_title('$\zeta$ = {0}'.format(key))
            # Control where y-label is set
            if i % ncols == 0:
                ax.set_ylabel('$f S_w / \overline{U}^2$')
            if i // ncols > 0:
                ax.set_xlabel('$f z / \overline{U}$')
    
    hand, labl = ax.get_legend_handles_labels()
    fig.legend(labels=labl, loc='upper center', ncol=len(heights), bbox_to_anchor=(0.5, 1.1), fontsize=12, frameon=False)
        
    fig.tight_layout()

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
            # Generate turbulent intensity for u, per Stull (1988)
            subdata['I_u'] = np.sqrt(subdata['var_u']) / np.nanmean(subdata['U'])
            # Generate turbulent intensity for v, per Stull (1988)
            subdata['I_v'] = np.sqrt(subdata['var_v']) / np.nanmean(subdata['U'])
            # Generate turbulent intensity for w, per Stull (1988)
            subdata['I_w'] = np.sqrt(subdata['var_w']) / np.nanmean(subdata['U'])
            # Append to list
            dfs.append(subdata)
    # Concatenate all the groups
    data_ = pd.concat(dfs).sort_values('time')
    
    return data_

def stability_plots(datasets, stability_method='surface', param='I_u', norm=None, std_plot=True):
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
    
    # Convert the input data to list form if it's not in it already
    if type(datasets) is not list:
        datasets = [datasets]
        
    # Determine exponent for normalization
    # If normalization isn't performed, exponent to 0 so the normalization value is 1
    # Else, set to 3 for TKE normalization and 2 for all else
    if norm is None:
        exp = 0
    else:
        exp = 3 if '_e' in param else 2
    
    # Define stability group bins
    bins = [-np.inf, -0.1, 0.1, np.inf]
    # Define colors and markers for plotting
    colors = ['blue', 'red']
    markers = ['o', 's']
    # Initialize figures
    fig, axs = plt.subplots(ncols=len(bins), sharey=True)
    
    # Initialize plot x-limits
    vmin, vmax = 0, 0
    
    for i, data in enumerate(datasets):
        # Initialize dictionaries for intensity mean, standard deviation values, and stability group names
        means, stds, groups = {}, {}, None
        # Group by height and iterate over each height
        for height, height_data in data.groupby('height'):
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
                    std.append((subdata[param]/subdata[norm]**exp).std())
                else:
                    mean.append((subdata[param]).mean())
                    std.append((subdata[param]).std())
            # Append to dict
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
            if (np.nanmin(means + stds)) < vmin:
                vmin = np.nanmin(means) + np.nanmin(stds)
            if (np.nanmax(means + stds)) > vmax:
                vmax = np.nanmax(means) + np.nanmax(stds)
        else:
            if np.nanmin(means) < vmin:
                vmin = np.nanmin(means)
            if np.nanmax(means) > vmax:
                vmax = np.nanmax(means)
            
        # Plot the turbulent intensities by stability group
        for j, group in enumerate(groups):
            # Plot the intensity mean values
            axs[j].plot(means[group], means.index, 
                        marker=markers[i], color=colors[i])
            if std_plot:
                # Plot the intensity standard deviations
                axs[j].errorbar(means[group], 
                                stds.index, 
                                xerr=stds[group],
                                fmt='none',
                                capsize=3, 
                                ecolor=colors[i])
            axs[j].set_xlim([vmin, vmax])
            axs[j].set_title(group)
            axs[j].set_ylim([0, data.height.max() + 100])
            # Set the x-label
            if norm:
                axs[j].set_xlabel(r'{0} / {1}$^2$'.format(param, norm))
            else:
                axs[j].set_xlabel(r'{0}'.format(param))
            # Only print the y-label on the first subplot
            if j == 0:
                axs[j].set_ylabel('Height [m]')
    
    fig.suptitle(data['site'].unique()[0])
    fig.tight_layout()
    return means, stds

def stability_distribution(dataset, height=100):
    '''
    Plot the stability count per hour to show the distribution of the dataset when future data is grouped by stability bins.

    Parameters
    ----------
    dataset : Pandas DataFrame
        DataFrame containing basic output data from the processor function.
    height : int, optional
        Height at which stability data is being counted. The default is 100.
    '''    
    
    # Get hour number and append to corresponding entries
    dataset['hour'] = dataset['time'].dt.hour
    # Filter data for specific height to prevent duplicates
    dataset = dataset[dataset['height'] == height]
    
    # Initialize list of dictionaries that will be concatenated into a DataFrame
    counts = []
    # Define stability bins
    bins = [-np.inf, -0.1, 0.1, np.inf]
    # Get stability bin interval names
    bin_names = pd.cut(dataset['zeta'], bins=bins).unique()
    # Iterate through the stability groupings
    for group, group_data in dataset.groupby(pd.cut(dataset['stability'])):
        # Iterate over the hours in each stability group dataset
        # Note that value counts are used to count the number of stability grouping occurences for each given hour
        for hour, count in enumerate(group_data['hour'].value_counts()):
            # Initialize dictionary that will be used for DataFrame conversion later
            temp = {}
            # Add hour to dictionary
            temp['hour'] = group_data['hour'].value_counts().index[hour]
            # Iterate through the bins to append data where valid. Zero otherwise.
            for bin_ in bin_names:
                if bin_ == group:
                    temp[bin_] = count
                else:
                    temp[bin_] = 0
            counts.append(temp)
    # Convert the list of dictionaries to a DataFrame and compress it by summing hour counts and ordering the axes by increasing stability.
    counts = pd.DataFrame(counts).groupby('hour').sum().sort_index(axis=1)
    # Remove the unnecessary 'nan' column
    counts = counts.loc[:, (counts != 0).any(axis=0)]
    # Plot
    counts.plot(kind='bar', stacked=True, cmap='RdYlBu')

def main(start_date, end_date, site='MANH', heights=[200, 400, 600, 800], event='normal', plot_spectra=False):
    
    # Initialize list of DataFrames that will be concatenated
    data, spectra = [], {}
    # Iterate over list of heights
    for height in heights:
        # Load lidar and/or anemometer data
        df = processor(start_date, end_date, height=height, site=site, event=event)
        data.append(df)
        # spectra[height] = spectral_analysis(df)
    
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
    turbulence_normal = turbulence_stats(data[(data['time'] >= normal_dates[0]) & 
                                              (data['time'] <= normal_dates[-1])])
    
    # Plot turbulent intensities
    for component in ['u', 'v', 'w']:
        stability_plots([turbulence_normal], param='I_{0}'.format(component))
    
    return data, turbulence_normal


if __name__ == '__main__':

    # Define date range
    start_date, end_date = ['2021-08-12', '2021-08-15']
    # Define height range
    heights, step = [100, 1000], 300
    # Collection list to be concatenated later
    turbulence_data = []
    for day_ in clear_sky_days_2021[0:2]:
        print(day_)
        start, end = [day_.strftime('%Y-%m-%d'), 
                      (day_ + datetime.timedelta(days=1)).strftime('%Y-%m-%d')]
        # Generate data for a given site and heights
        output, turbulence_data_ = main(start, end, site='STAT', heights=np.arange(heights[0], heights[1]+step, step), event='normal', plot_spectra=False)
        turbulence_data.append(turbulence_data_)
    turbulence_data = pd.concat(turbulence_data)