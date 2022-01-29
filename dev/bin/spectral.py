"""
Urban Boundary Layer Observation Data Processing
Script name:    Spectral Analysis
Path:           ~/bin/spectral.py
Description:    Perform spectral analysis on data collected by the flux tower atop the Marshak Building at The City College of New York.
"""

import datetime, math, matplotlib.pyplot as plt, numpy as np, os, pandas as pd, xarray as xr, scipy, seaborn as sns

import bin.functions, bin.lidar
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from cycler import cycler

import warnings
warnings.filterwarnings("ignore")

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


def processor(height=200, site='MANH'):
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
        files = ['/Volumes/UBL Data/data/flux/MANH/TOA5_10560_s20210720_e20210802_flux.dat', '/Volumes/UBL Data/data/flux/MANH/TOA5_10560_s20210802_e20210818_flux.dat']
        
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
            sonic_data = xr.open_dataset('/Volumes/UBL Data/data/flux/MANH/ts_data_s202107310000_e202108110000_r202201180830.nc')
            # Select data for dates of interest and convert to DataFrame
            df = sonic_data.sel(time=slice('2021-07-31', '2021-08-10')).to_dataframe()
            # Sort by time
            df = df.sort_values('time')
            # Remove MultiIndex to allow for time and height indices to become columns
            df = df.reset_index()
        else:    
            # Get lidar data from the August lidar file dedicated for spectral analysis
            lidar_data_fixed = xr.open_mfdataset(['/Volumes/UBL Data/data/storage/lidar/lidar_data_2021-08_spectral.nc']).sel(site='MANH', height=height)
            # Select lidar data for dates of interest and convert to DataFrame
            df_fixed = lidar_data_fixed.sel(time=slice('2021-07-31', '2021-08-10')).to_dataframe()
            
            # Get Doppler-beam swinging data from the lidar files dedicated for turbulence analysis
            dbs_dir_ = '/Volumes/UBL Data/data/storage/lidar'
            # Get file list corresponding to DBS lidar data
            dbs_files = [os.path.join(dbs_dir_, file) for file in os.listdir(dbs_dir_) if 'MANH_DBS' in file]
            # Get DBS lidar data
            df_dbs = xr.open_mfdataset(dbs_files).sel(site=site, height=height).drop_vars(['ci']).to_dataframe()
            
            # Concatenate DataFrames
            df = pd.concat([df_fixed, df_dbs])
            # Remove MultiIndex to allow for time and height indices to become columns
            df = df.reset_index().sort_values('time')
            
    else:
        # List files containing Queens data
        date_range = [datetime.datetime(year=2021, month=8, day=1),
                      datetime.datetime(year=2021, month=9, day=1)]
        file_dir = '/Volumes/UBL Data/data/flux/QUEE'
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
        ts_data = ts_data.rename(columns={'WD': 'wnd_dir_compass', 'USTAR': 'u_star', 'ZL': 'zeta', 'WS': 'U'})
        
        ''' Pull lidar data. '''
        # Get lidar data from the August lidar file dedicated for spectral analysis
        lidar_data = xr.open_dataset('/Volumes/UBL Data/data/storage/lidar/lidar_data_2021-08_spectral_STAT.nc').sel(site='STAT', height=height)
        # Select lidar data for dates of interest and convert to DataFrame
        df = lidar_data.sel(time=slice('2021-08-01', '2021-09-01')).to_dataframe()
        # Remove MultiIndex to allow for time and height indices to become columns
        df = df.reset_index()
    
    ''' De-trend data to obtain velocity fluctuations '''
    # Initialize temporary container list
    dfs_ = []
    # De-trend period
    period = '3T'
    # Iterate over every de-trending period
    for group, data in df.groupby(pd.Grouper(key='time', freq=period)):
        # Calculate mean velocity quantity per period
        data['w_mean'] = data['w'].mean()
        # Calculate fluctuating velocity quantity
        data['w_prime'] = data['w'] - data['w_mean']
        # If u is in the DataFrame, get the mean and fluctuating quantities
        if 'u' in data.columns:
            data['u_mean'] = data['u'].mean()
            # Calculate fluctuating velocity quantity
            data['u_prime'] = data['u'] - data['u_mean']
        # If u is in the DataFrame, get the mean and fluctuating quantities
        if 'v' in data.columns:
            data['v_mean'] = data['v'].mean()
            # Calculate fluctuating velocity quantity
            data['v_prime'] = data['v'] - data['v_mean']
        dfs_.append(data)
    df = pd.concat(dfs_).sort_values('time')
    
    # Group DataFrame into 30-minute intervals. This allows for matching with 30-minute-averaged ts_data
    df_grouped = df.groupby(pd.Grouper(key='time', freq='30T'))
    
    ''' Match the lidar and flux tower data to permit grouping by stability. '''
    # Initialize list of modified DataFrames to be concatenated.
    dfs = []
    # For the 30-minute binned DataFrames, add columns with data from the flux tower data.
    for label, data in df_grouped:
        # Match lidar and flux tower data by timestamp
        match = ts_data.loc[ts_data['TIMESTAMP'] == label]
        # Filter out necessary columns for spectra normalization
        match = match[['z', 'U', 'u_star', 'zeta', 'wnd_dir_compass']]
        # If the matched DataFrame column is empty, fill with np.nan
        for col in match.columns:
            if match[col].empty:
                match[col] = [np.nan]
        # Append the matching value to the DataFrame as a column for each parameter needed for normalization.
        
        data['z'] = np.repeat(match['z'].values, len(data))
        data['U'] = np.repeat(match['U'].values, len(data))
        data['u_star'] = np.repeat(match['u_star'].values, len(data))
        data['zeta'] = np.repeat(match['zeta'].values, len(data))
        data['wind_direction'] = np.repeat(match['wnd_dir_compass'].values, len(data))
        # Calculate variance
        variance = np.nansum((data['w'] - data['w_mean']))**2/len(data['w'])
        data['w_var'] = variance
        # Append the matched DataFrame to the list of DataFrames
        dfs.append(data)
    # Concatenate the matched DataFrames
    dfs = pd.concat(dfs).sort_values('time')
    
    # Filter out by wind direction - northerlies filtered out
    if site == 'MANH':
        dfs = dfs.where((dfs['wind_direction'] >= 90) & (dfs['wind_direction'] <= 270))
    elif site == 'QUEE':
        dfs = dfs.where((dfs['wind_direction'] >= 180) & (dfs['wind_direction'] <= 360))
    
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
    bins = [-np.inf, -2, -1, -0.1, 0.1, 1, np.inf]
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

def plotter(data, heights):
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
    
    # Clean the data
    try:
        data = data.drop(columns=['ci', 'u_std', 'v_std', 'w_std', 'wind_direction_std', 'ci_std'])
        data = data.dropna()
    except:
        pass
    
    # Define interval for time grouping
    interval = '30T'
    # Generate grouped data based on interval
    grouped = data.groupby(pd.Grouper(key='time', freq=interval))
    # Generate list of DataFrames to be rejoined
    dfs = []
    # Iterate through each group, calculating defined turbulence statistics
    for group, subdata in grouped:
        # Calculate std for zonal wind
        subdata['u_std'] = np.sqrt(np.nansum((subdata['u'] - subdata['u_mean'])**2)/len(subdata))
        # Generate turbulent intensity, per Stull (1988)
        subdata['I_u'] = subdata['u_std'] / np.nanmean(subdata['U'])
        # Calculate std for meridional wind
        subdata['v_std'] = np.sqrt(np.nansum((subdata['v'] - subdata['v_mean'])**2)/len(subdata))
        # Generate turbulent intensity, per Stull (1988)
        subdata['I_v'] = subdata['v_std'] / np.nanmean(subdata['U'])
        # Calculate std for vertical wind
        subdata['w_std'] = np.sqrt(np.nansum((subdata['w'] - subdata['w_mean'])**2)/len(subdata))
        # Generate turbulent intensity, per Stull (1988)
        subdata['I_w'] = subdata['w_std'] / np.nanmean(subdata['U'])
        # Append to list
        dfs.append(subdata)
    # Concatenate all the groups
    data = pd.concat(dfs).sort_values('time')
    
    # Store dictionary of turbulent intensity values
    intensity_means, intensity_stds = {}, {}
    # Group by zeta
    bins = [-np.inf, -2, -1, -0.1, 0.1, 1, np.inf]
    # Group the matched DataFrames by stability classification
    stability_groups = data.groupby(pd.cut(data['zeta'], bins))
    # Iterate over the DataFrames and get averages by stability group
    for group, data_ in stability_groups:
        # Group by height
        height_groups = data_.groupby(['height'])
        for subgroup, subdata in height_groups:
            if group in intensity_means.keys():
                intensity_means[group].append(np.nanmean(subdata['I_u']))
            else:
                intensity_means[group] = [np.nanmean(subdata['I_u'])]
                
            if group in intensity_stds.keys():
                intensity_stds[group].append(np.nanstd(subdata['I_u']))
            else:
                intensity_stds[group] = [np.nanstd(subdata['I_u'])]
    
    return data

def main(heights=[200, 500], plot_spectra=False):
    
    # Initialize list of DataFrames that will be concatenated
    data, spectra = [], {}
    # Iterate over list of heights
    for height in heights:
        # Load lidar and/or anemometer data
        df = processor(height=height)
        data.append(df)
        spectra[height] = spectral_analysis(df)
        
    # Concatenate data
    data = pd.concat(data)
    # Plot spectra
    if plot_spectra:
        plotter(list(spectra.values()), list(spectra.keys()))
        
    turbulence_analysis = turbulence_stats(data)
    
    return data, turbulence_analysis

if __name__ == '__main__':
    
    data, turbulence_analysis = main()
    
    # MATLAB analysis
    save_mat = False
    if save_mat:
        dfs_ = dfs.copy().reset_index()
        df_ = dfs_[['time', 'w', 'w_prime', 'zeta']]
        scipy.io.savemat('/Users/gabriel/Documents/urban_boundary_layer_obs/dev/bin/matlab/sonic_data_ccny_extremely_unstable.mat', {name: col.values for name, col in df_.items()})
    