import os
import pandas as pd
import scipy.io as sio
import set_paths
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import conversions
from colors import bluered10

# set paths
home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')

bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# mean frequency bands

def theta(data):
    theta_mean = data.loc[2:10].mean().values
    return theta_mean

def beta(data):
    beta_mean = data.loc[10:30].mean().values
    return beta_mean

def gamma(data):
    gamma_mean = data.loc[30:100].mean().values
    return gamma_mean


class Coherence:
    '''
    Opens the coherence for the model with all channels, scales, and optimal and no lambda penalty values
    with methods for plotting or pulling out bands or channels
    Parameters:
        -----------
        paths_object: set_paths.Paths object
            the path object from set_paths.py

        Returns:
        self.coherence: pd.DataFrame
            the DataFrame of all coherences for a given biophysical model with all channels, scales, and lambdas
    '''
    def __init__(self, paths_object):       
        coherence_path = os.path.join(paths_object.coherence, 'coherence.csv')
        self.model = paths_object.model

        if os.path.exists(coherence_path):
            print('Coherence file exists')
        
        else:
            conversions.convert_coherences(paths_object)

        self.coherence = pd.read_csv(coherence_path, index_col=[0], header=[0, 1, 2])
        print('Coherence df for {} model opened'.format(self.model))

        self.channels = self.coherence.columns.levels[0].tolist()
        self.scales = self.coherence.columns.levels[1].tolist()
        self.lambdas = self.coherence.columns.levels[2].tolist()


    def plot_channel_coherence(self, axs, channel, scales, lambda_value='optimal', frequency_cutoff=100, legend=False):
        '''
        Plots coherence from  0 to frequency_cutoff in Hz for a given channel, scales and lambda penalty value.

        Parameters:
        -----------
        axs: plt.axes
            axis on which to plot the traces
        channel: str
            a channel present in the model; one of self.channels
        scales: str or list of str
             'all' for all scales, or a list of str of which scales to plot from self.scales
        lambda_value: str
            'none', 'optimal', or 'both'
            default='optimal'
        frequency_cutoff: int, float
            frequency cutoff (in Hz) for slicing the dataset
            default=100 (Hz)
        legend: bool; default = False
            whether or not to return the handles for a legend
        Return:
        -------
        axs: plt.axes omstamce
            an axs instance of the channel and lambda value 
        hand
        '''
        if scales == 'all':
            cohr_none = self.coherence.loc[:frequency_cutoff, (channel, slice(None), 'none')]
            cohr_optimal = self.coherence.loc[:frequency_cutoff, (channel, slice(None), 'optimal')]

            scales_set = set(cohr_optimal.columns.remove_unused_levels().levels[1].tolist())

        else:
            cohr_none = self.coherence.loc[:frequency_cutoff, (channel, scales, 'none')]
            cohr_optimal = self.coherence.loc[:frequency_cutoff, (channel, scales, 'optimal')]

            scales_set = set(scales)

        scales_list = list(scales_set)
        scales_list.sort(reverse=True)
        
        color_indices = [i for i, scale in enumerate(self.scales) if scale in scales_set]
        colors = [bluered10[i] for i in color_indices]
        sns.set_palette(colors)
        '''
        if lambda_value == 'all':
            fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True, constrained_layout=True)

            axs[0].plot(cohr_none, label=scales_list)
            axs[0].text(0.98, 0.98, r'$\lambda = 0$', transform=axs[0].transAxes, horizontalalignment='right', verticalalignment='top')
            axs[1].plot(cohr_optimal)
            axs[1].text(0.98, 0.98, r'$\lambda = \lambda*$', transform=axs[1].transAxes, horizontalalignment='right', verticalalignment='top')
            axs[0].set_ylabel('Coherence')
            handles, labels = axs[0].get_legend_handles_labels()

            for ax in axs.flat:
                ax.set_xlabel('Frequency (Hz)')
        '''
        if lambda_value == 'none':
            axs.plot(cohr_none, label=scales_list)
            # axs.text(0.98, 0.98, r'$\lambda = 0$', transform=axs.transAxes, horizontalalignment='right', verticalalignment='top')
            axs.set_ylabel('Coherence')
            axs.set_xlabel('Frequency (Hz)')
            handles, labels = axs.get_legend_handles_labels()

        elif lambda_value == 'optimal':
            axs.plot(cohr_optimal, label=scales_list)
            # axs.text(0.98, 0.98, r'$\lambda = \lambda*$', transform=axs.transAxes, horizontalalignment='right', verticalalignment='top')
            axs.set_ylabel('Coherence')
            axs.set_xlabel('Frequency (Hz)')
            handles, labels = axs.get_legend_handles_labels()

        handles_rev = handles[::-1]

        if legend is True:
            return axs, handles_rev

        else:
            return axs

    def plot_coherence_subtraction(self, axs, channel, scales, lambda_value='optimal', frequency_cutoff=100, legend=False):
        if scales == 'all':
            cohr_none = self.coherence.loc[:frequency_cutoff, (channel, slice(None), 'none')]
            cohr_optimal = self.coherence.loc[:frequency_cutoff, (channel, slice(None), 'optimal')]

            scales_set = set(cohr_optimal.columns.remove_unused_levels().levels[1].tolist())

        else:
            cohr_none = self.coherence.loc[:frequency_cutoff, (channel, scales, 'none')]
            cohr_optimal = self.coherence.loc[:frequency_cutoff, (channel, scales, 'optimal')]

            scales_set = set(scales)

        cont_none = pd.DataFrame(cohr_none.loc[:, (channel, '1.0', 'none')])
        cont_opt = pd.DataFrame(cohr_optimal.loc[:, (channel, '1.0', 'optimal')])

        sub_none = np.subtract(cohr_none, cont_none)
        sub_opt = np.subtract(cohr_optimal, cont_opt)

        scales_list = list(scales_set)
        scales_list.sort(reverse=True)
        
        color_indices = [i for i, scale in enumerate(self.scales) if scale in scales_set]
        colors = [bluered10[i] for i in color_indices]
        sns.set_palette(colors)
        '''
        if lambda_value == 'all':
            fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True, constrained_layout=True)

            axs[0].plot(cohr_none, label=scales_list)
            axs[0].text(0.98, 0.98, r'$\lambda = 0$', transform=axs[0].transAxes, horizontalalignment='right', verticalalignment='top')
            axs[1].plot(cohr_optimal)
            axs[1].text(0.98, 0.98, r'$\lambda = \lambda*$', transform=axs[1].transAxes, horizontalalignment='right', verticalalignment='top')
            axs[0].set_ylabel('Coherence')
            handles, labels = axs[0].get_legend_handles_labels()

            for ax in axs.flat:
                ax.set_xlabel('Frequency (Hz)')
        '''
        if lambda_value == 'none':
            axs.plot(sub_none, label=scales_list)
            # axs.text(0.98, 0.98, r'$\lambda = 0$', transform=axs.transAxes, horizontalalignment='right', verticalalignment='top')
            axs.set_ylabel(r'$\Delta$ Coherence')
            axs.set_xlabel('Frequency (Hz)')
            handles, labels = axs.get_legend_handles_labels()

        elif lambda_value == 'optimal':
            axs.plot(sub_opt, label=scales_list)
            # axs.text(0.98, 0.98, r'$\lambda = \lambda*$', transform=axs.transAxes, horizontalalignment='right', verticalalignment='top')
            axs.set_ylabel(r'$\Delta$ Coherence')
            axs.set_xlabel('Frequency (Hz)')
            handles, labels = axs.get_legend_handles_labels()

        handles_rev = handles[::-1]

        if legend is True:
            return axs, handles_rev

        else:
            return axs

    
    def plot_coherence_bands(self, axs, channels='all', lambda_value='optimal', legend=True):
        '''
        Plots the mean coherence across theta, beta, and gamma bands broadly construed. 
        Theta = 2-10 Hz
        Beta = 10-30 Hz
        Gamma = 30-100 Hz

        Parameters:
        -----------
        channels: list(str)
            override default with a list of channels to plot, default=all channels in self.channels
        lambda_value: str
            'none', 'optimal', or 'both'
            default='optimal'
        legend: bool
            whether to plot a legend with the figure or not. If True, will plot all
            coded channel values as legend)
            default=True

        Return:
        -------
        fig: plt.figure
            a figure instance of the channel and lambda value 
        axs: plt.axes
            the axes instances that belong to the figure
        '''
        band_labels = ['2-10 Hz', '10-30 Hz', '30-100 Hz']

        if channels == 'all':
            channels = self.channels
            
        elif isinstance(channels, list):
            channels = channels
            
        else:
            raise TypeError('channels must be a list of channels or str(all).')

        sns.set_palette('tab10')
               
       
        if lambda_value == 'optimal':
            for channel in channels:
                opt = self.coherence.loc[:, (channel, slice(None), 'optimal')]
                index_str = opt.columns.remove_unused_levels().levels[1].tolist()
                index_float = np.array(index_str, dtype=float)
                axs[0].plot(index_float, theta(opt), marker='.', label=channel)
                axs[1].plot(index_float, beta(opt), marker='.')
                axs[2].plot(index_float, gamma(opt), marker='.')

        elif lambda_value == 'none':
            for channel in channels:
                none = self.coherence.loc[:, (channel, slice(None), 'none')]
                index_str = none.columns.remove_unused_levels().levels[1].tolist()
                index_float = np.array(index_str, dtype=float)
                axs[0].plot(index_float, theta(none), marker='.', label=channel)
                axs[1].plot(index_float, beta(none), marker='.')
                axs[2].plot(index_float, gamma(none), marker='.')

        elif lambda_value == 'both':
            for channel in channels:
                none = self.coherence.loc[:, (channel, slice(None), 'none')]
                opt = self.coherence.loc[:, (channel, slice(None), 'optimal')]
                index_str = none.columns.remove_unused_levels().levels[1].tolist()
                index_float = np.array(index_str, dtype=float)
                axs[0].plot(index_float, theta(none), marker='.', label=channel)
                axs[0].plot(index_float, theta(opt), marker='.', label=channel + '*')
                axs[1].plot(index_float, beta(none), marker='.')
                axs[1].plot(index_float, beta(opt), marker='.')
                axs[2].plot(index_float, gamma(none), marker='.')
                axs[2].plot(index_float, gamma(opt), marker='.')
            
        for i, ax in enumerate(axs.flat):
            ax.set_ylabel('Coherence')
            ax.set_ylim(0, 1.0)
            ax.axvline(x=1, color='gray', alpha=0.5, linestyle='--')
            ax.text(0.95, 0.98, band_labels[i], transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')

        axs[2].set_xlabel('Conductance Scaling Factor')
        
        if legend is True:
            axs[0].legend(bbox_to_anchor=(0, 1.05, 1, 1), fontsize=10, mode='expand', loc='lower left', ncol=3, borderaxespad=0., columnspacing=1.0, handlelength=1.5)

        return axs

    def __repr__(self):
        return(f'Coherence df for {self.model} model')
