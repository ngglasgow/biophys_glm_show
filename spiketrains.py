import os
import pandas as pd
import scipy.io as sio
import scipy
import set_paths
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import conversions
from colors import bluered10


def corr_psths(SpikeTrain_1, SpikeTrain_2, lambda_1, lambda_2, xlim=(None, None)):
    xmin = xlim[0]
    xmax = xlim[1]

    psth_1 = SpikeTrain_1.psths.loc[xmin:xmax, (slice(None), slice(None), lambda_1)].copy()
    psth_1.columns = psth_1.columns.droplevel('lambda')

    psth_2 = SpikeTrain_2.psths.loc[xmin:xmax, (slice(None), slice(None), lambda_2)].copy()
    psth_2.columns = psth_2.columns.droplevel('lambda')

    if len(psth_1.columns) > len(psth_2.columns):
        psth_1 = psth_1[psth_2.columns]
    elif len(psth_2.columns) > len(psth_1.columns):
        psth_2 = psth_2[psth_1.columns]

    corr = psth_1.corrwith(psth_2)
    corr.sort_index(level='channel', inplace=True)

    return corr


def plot_psth_corr(ax, corr, channels, legend=True):
    if channels is 'all':
        channels = corr.index.levels[0].tolist()
    elif isinstance(channels, list):
        channels = channels
    else:
        raise TypeError('channels must be a list of channels or str(all)')
    sns.set_palette('tab10')

    for channel in channels:
        channel_corr = corr.loc[channel, slice(None)]
        index_str = channel_corr.index.remove_unused_levels().levels[1].tolist()
        index_float = np.array(index_str, dtype=float)
        ax.plot(index_float, channel_corr, marker='.', label=channel)

    ax.set_xlabel('Conductance Scaling Factor')
    ax.set_ylabel('Correlation')

    minimum = round(corr.min(), 1)
    if minimum > corr.min():
        ymin = minimum - 0.1
    else:
        ymin = minimum

    ax.set_ylim(ymin, 1.0)

    if legend is True:
        ax.legend(bbox_to_anchor=(0, 1.05, 1, 1), fontsize=10, mode='expand', loc='lower left', ncol=3, borderaxespad=0., columnspacing=1.0, handlelength=1.5)


    return ax


class SpikeTrains:
    '''
    Opens the coherence for the model with all channels, scales, and optimal and no lambda penalty values
    with methods for plotting or pulling out bands or channels
    Parameters:
        -----------
        paths_object: set_paths.Paths object
            the path object from set_paths.py
        data_type: str
            'biophys' or 'glm'
            str describing what kind of data it is to determine the appropriate source

        Returns:
        self.indices: pd.DataFrame
            the DataFrame of all spike indices for a given biophysical model with all channels, scales, and lambdas
        self.spiketimes: pd.DataFrame
            the DataFrame of all spike times for a given biophysical model with all channels, scales, and lambdas
    '''
    def __init__(self, paths_object, data_type):
        if data_type == 'glm':
            data_path = paths_object.glm_sims
        elif data_type == 'biophys':
            data_path = paths_object.biophys_output
        else:
            raise ValueError('data_type can only be glm or biophys')
        self.data_type = data_type
        self.data_path = data_path
        self.paths_object = paths_object

        spiketimes_path = os.path.join(data_path, 'spiketimes.csv')
        indices_path = os.path.join(data_path, 'spikeindices.csv')

        self.model = paths_object.model

        if os.path.exists(spiketimes_path):
            print('spiketimes file exists')
        
        else:
            if data_type == 'glm':
                conversions.convert_glm_spiketrains(paths_object)

            elif data_type == 'biophys':
                conversions.convert_biophys_spiketrains(paths_object)

        self.indices = pd.read_csv(indices_path, index_col=[0], header=[0, 1, 2])
        self.spiketimes = pd.read_csv(spiketimes_path, index_col=[0], header=[0, 1, 2])

        print(f'Spiketimes dfs for {self.model} model opened')

        self.channels = self.indices.columns.levels[0].tolist()
        self.scales = self.indices.columns.levels[1].tolist()
        self.lambdas = self.indices.columns.levels[2].tolist()


    def open_psth(self, trial_length=3200, sd=2, fs=1000):
        '''
        Checks the data_path directory for psths.csv, and opens if it exists, prints
        message to creat it if not.
        trial_length: int
            the index of the length of the actual trial
            default = 3200 ms
        sd: int, float
            standard deviation to use for the gaussian window for the PSTH kernel
            default = 2 ms
        fs: int
            sampling frequency in Hz
            default = 1000 Hz
        '''
        psths_path = os.path.join(self.data_path, 'psths.csv')

        if os.path.exists(psths_path):
            print('PSTH value file exists')
            print(f'PSTH df for {self.model} model opened')
        else:
            conversions.create_psths(self.paths_object, self.data_type, trial_length, sd, fs)
            print('May need to rerun conversions.create_psths() with non-default args.')

        self.psths = pd.read_csv(psths_path, index_col=[0], header=[0, 1, 2])

    def plot_single_rasters(self, ax, channel, scale, lambda_value, xlim=None):
        '''
        Plots the rasters of all trials given a channel, scale, and lambda_value
        with adjustable xlim.

        Parameters:
        -----------
        ax: plt.axes
            the axes on which to supply the scatter
        channel: str
            a channel present in the model; one of self.channels
        scales: str
             a single scale as a str from the list of scales self.scales
        lambda_value: str
            'none' or 'optimal'
        xlim: None or tuple
            tuple where index (start, stop) is beginning and end of xlim to plot
            default = None to plot whole xlim
        legend: bool
            whether to plot a legend with the figure or not. If True, will plot all
            coded channel values as legend)
            default=True

        Return:
        -------
        axs: plt.axes
            the axes instances that belong to the figure
        '''
        indices = self.indices.loc[:, (channel, scale, lambda_value)]
        spiketimes = self.spiketimes.loc[:, (channel, scale, lambda_value)]

        color_index = self.scales.index(scale)
        color = bluered10[color_index]

        indices_spikes = np.array((indices, spiketimes))
        trial_counter = np.arange(1, max(indices) + 1, 1)
        spikes_list = [indices_spikes[1][indices_spikes[0] == trial] for trial in trial_counter]
        
        spikes_df = pd.DataFrame(spikes_list)

        x = np.arange(0, spikes_df.shape[1], 1)
        y = np.arange(0, spikes_df.shape[0], 1)
        _, ys = np.meshgrid(x, y)
        
        if xlim is None:
            ax.scatter(spikes_df, ys, s=1, marker='.', c=color)

        elif isinstance(xlim, tuple):
            start = xlim[0]
            stop = xlim[1]
            lim_spikes_df = spikes_df[(spikes_df >= start) & (spikes_df <= stop)]
            ax.scatter(lim_spikes_df, ys, s=1, marker='.', c=color)
            ax.set_xlim(start, stop)

        else:
            return TypeError('xlim must be None or tuple')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Trial')

        return ax


    def plot_all_rasters(self, ax, channel, lambda_value, xlim=None):
        '''
        Plots the rasters of all trials given a channel, scale, and lambda_value
        with adjustable range.

        Parameters:
        -----------
        ax: plt.axes
            the axes on which to supply the scatter
        channel: str
            a channel present in the model; one of self.channels
        lambda_value: str
            'none' or 'optimal'
        xlim: None or tuple
            tuple where index (start, stop) is beginning and end of range to plot
            default = None to plot whole range
        legend: bool
            whether to plot a legend with the figure or not. If True, will plot all
            coded channel values as legend)
            default=True

        Return:
        -------
        axs: plt.axes
            the axes instances that belong to the figure
        '''

        for meta_count, scale in enumerate(self.scales):
            indices = self.indices.loc[:, (channel, scale, lambda_value)]
            spiketimes = self.spiketimes.loc[:, (channel, scale, lambda_value)]

            color_index = self.scales.index(scale)
            color = bluered10[color_index]

            indices_spikes = np.array((indices, spiketimes))

            trial_counter = np.arange(1, max(indices) + 1, 1)
            spikes_list = [indices_spikes[1][indices_spikes[0] == trial] for trial in trial_counter]
        
            scale_trial_break = meta_count * (max(indices) + 1)
            scale_trial_start = scale_trial_break + 1
            
            spikes_df = pd.DataFrame(spikes_list)

            x = np.arange(0, spikes_df.shape[1], 1)
            y = np.arange(scale_trial_start, scale_trial_start + spikes_df.shape[0], 1)
            _, ys = np.meshgrid(x, y)

            if xlim is None:
                ax.scatter(spikes_df, ys, s=1, marker='.', c=color)

            elif isinstance(xlim, tuple):
                start = xlim[0]
                stop = xlim[1]
                lim_spikes_df = spikes_df[(spikes_df >= start) & (spikes_df <= stop)]
                ax.scatter(lim_spikes_df, ys, s=1, marker='.', c=color)
                ax.set_xlim(start, stop)
            
            else:
                return TypeError('xlim must be None or tuple')

            ax.axhline(y=scale_trial_break, color='gray', linewidth=0.5)

        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Trial')

        return ax


    def plot_psth(self, ax, channel, scale, lambda_value, xlim=(None, None)):
        '''
        Plots the rasters of all trials given a channel, scale, and lambda_value
        with adjustable range.

        Parameters:
        -----------
        ax: plt.axes
            the axes on which to supply the scatter
        channel: str
            a channel present in the model; one of self.channels
        scales: str
             a single scale as a str from the list of scales self.scales
        lambda_value: str
            'none' or 'optimal'
        xlim: tuple
            tuple where index (start, stop) is beginning and end of range to plot
            default = (None, None) to plot whole range
        legend: bool
            whether to plot a legend with the figure or not. If True, will plot all
            coded channel values as legend)
            default=True

        Return:
        -------
        axs: plt.axes
            the axes instances that belong to the figure
        '''
        start = xlim[0]
        stop = xlim[1]

        psth = self.psths.loc[start:stop, (channel, scale, lambda_value)]

        color_index = self.scales.index(scale)
        color = bluered10[color_index]
        
        if self.data_type is 'glm':
            linestyle = '--'
            
        elif self.data_type is 'biophys':
            linestyle = '-'

        ax.plot(psth, color=color, linestyle=linestyle)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing Rate (Hz)')

        return ax
        

    def __repr__(self):
        return f'Spiketimes and spikeindices for {self.model} model for {data_type} simulations'
