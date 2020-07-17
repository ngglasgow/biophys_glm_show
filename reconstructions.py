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
'''
# set paths
home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')

bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')
'''


class Reconstructions:
    '''
    Opens the reconstructions and actual test stimuli for a model with all channels, scales, and optimal and no 
    lambda penalty values with methods for plotting and pulling scales and channels.
    '''
    def __init__(self, paths_object):
        '''
        Opens reconstruction.csv and test_stimuli.csv with multiIndexing
        If the files are not present, it will run convert_reconstructions
        
        Parameters:
        -----------
        paths_object: set_paths.Paths_object
            the path object from set_paths.py
        
        Returns:
        ---------
        self.reconstructions: pd.DataFrame
            the DataFrame of all the reconstructions for a given biophysical model with all channels, scales, and lambda penalties
        self.test_stims: pd.DataFrame
            the DataFrame of all the testing stimuli used for reconstruction
        self.channels: list
            list of channels present in the index
        self.scales: list
            list of scales present in the index
        self.lambdas: list
            list of available lambda penalty values present in the index
        self.model: string
            model identifier taken from paths_object.model
        '''
        reconstruction_path = os.path.join(paths_object.reconstructed, 'reconstruction.csv')
        test_stim_path = os.path.join(paths_object.reconstructed, 'test_stimuli.csv')
        self.model = paths_object.model

        if os.path.exists(reconstruction_path):
            print('Reconstructions file exists')

        else:
            conversions.convert_reconstructions(paths_object)
        
        self.reconstructions = pd.read_csv(reconstruction_path, index_col=[0], header=[0, 1, 2]) * 1000
        print(f'Reconstruction df for {self.model} model opened')

        self.test_stims = pd.read_csv(test_stim_path, index_col=[0]) * 1000
        self.test_stims.columns = self.test_stims.columns.astype(int)
        print(f'Test stimuli df for {self.model} model opened')

        self.channels = self.reconstructions.columns.levels[0].tolist()
        self.scales = self.reconstructions.columns.levels[1].tolist()
        self.lambdas = self.reconstructions.columns.levels[2].tolist()


    def plot_channel_reconstructions(self, ax, channel, scales, xlim=(None, None), lambda_value='optimal', test_stim='mean', legend=False, title=False):
        '''
        Plots
        Parameters:
        -----------
        channel: str
            a channel present in the model; one of self.channels
        scales: str or list of str
             'all' for all scales, or a list of str of which scales to plot from self.scales
        range: str: 'all' or list(int)
            the x-range of the values in time. Either 'all' for the entire range, or a list
            of ints where [xmin, xmax] equals the range.
            default: 'all'
        lambda_value: str
            'none', 'optimal', or 'both'
            default='optimal'
        test_stim: str
            choose whether test_stim is 'mean', 'all', or 'none'
        legend: bool
            whether or not to plot the legend
            default: True
        title: bool
            whether or not to plot the channel in the title
            default: True

        Return:
        -------
        fig: plt.figure
            a figure instance of the channel and lambda value 
        '''
        xmin = xlim[0]
        xmax = xlim[1]

        if scales == 'all':
            recon_none = self.reconstructions.loc[xmin:xmax, (channel, slice(None), 'none')]
            recon_optimal = self.reconstructions.loc[xmin:xmax, (channel, slice(None), 'optimal')]

            scales_set = set(recon_optimal.columns.remove_unused_levels().levels[1].tolist())

        else:
            recon_none = self.reconstructions.loc[xmin:xmax, (channel, scales, 'none')]
            recon_optimal = self.reconstructions.loc[xmin:xmax, (channel, scales, 'optimal')]

            scales_set = set(scales)

        if test_stim is 'mean': 
            offset = self.test_stims.mean().mean()
            test_stim = self.test_stims.mean(axis=1).loc[xmin:xmax] - offset
            test_stim_label = 'mean'
        
        elif test_stim is 'all':
            offset = self.test_stims.mean().mean()
            test_stim = self.test_stims.loc[xmin:xmax] - offset
            test_stim_label = 'all'

        elif test_stim is 'none':
            test_stim = np.nan
            test_stim_label = None
        
        scales_list = list(scales_set)
        scales_list.sort()

        color_indices = [i for i, scale in enumerate(self.scales) if scale in scales_set]
        colors = [bluered10[i] for i in color_indices]
        
        
        '''
        if lambda_value == 'both':
            fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True, constrained_layout=True)
            axs[0].plot(test_stim, color='lightgray', linewidth=4, label=test_stim_label)
            axs[0].plot(recon_none, label=scales_list)
            axs[0].text(0.98, 0.98, r'$\lambda = 0$', transform=axs[0].transAxes, horizontalalignment='right', verticalalignment='top')
            axs[1].plot(test_stim, color='lightgray', linewidth=4)
            axs[1].plot(recon_optimal)
            axs[1].text(0.98, 0.98, r'$\lambda = \lambda*$', transform=axs[1].transAxes, horizontalalignment='right', verticalalignment='top')
            axs[0].set_ylabel('Current (pA)')
            handles, labels = axs[0].get_legend_handles_labels()

            for ax in axs.flat:
                ax.set_xlabel('Time (ms)')
        '''
        if lambda_value == 'none':
            ax.plot(test_stim, color='lightgray', linewidth=4, label=test_stim_label)
            sns.set_palette(colors)
            ax.plot(recon_none, label=scales_list, linewidth=2)
            ax.text(0.98, 0.98, r'$\lambda = 0$', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
            ax.set_ylabel('Current (pA)')
            ax.set_xlabel('Time (ms)')
            handles, _ = ax.get_legend_handles_labels()

        elif lambda_value == 'optimal':
            ax.plot(test_stim, color='lightgray', linewidth=4, label=test_stim_label)
            sns.set_palette(colors)
            ax.plot(recon_optimal, label=scales_list, linewidth=2)
            # ax.text(0.98, 0.98, r'$\lambda = \lambda*$', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
            ax.set_ylabel('Current (pA)')
            ax.set_xlabel('Time (ms)')
            handles, _ = ax.get_legend_handles_labels()
        
        if legend is True:
            handles_rev = handles[::-1]
            scales_list.append(test_stim_label)
            ax.legend(handles_rev, scales_list, bbox_to_anchor=(1.35, 0.53), loc='center right', handlelength=1.5, borderaxespad=0)
        
        if title is True:
            ax.suptitle(channel)

        return ax


    def __repr__(self):
        return(f'Reconstruction df for {self.model} model')


'''
bhalla_recon = reconstructions(bhalla_paths)
bhalla_recon.plot_channel_reconstructions('kA', ['0.5', '1', '1.5'])
'''