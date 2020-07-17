# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

'''################### Set direcotories and open files #####################'''
# set system type as macOS or linux; set project directory; set downstream dirs
# set all these directory variables first

home_dir = os.path.expanduser("~")
project_path = 'projects/biophys_glm_show'
project_dir = os.path.join(home_dir, project_path)
bhalla_dir = os.path.join(project_dir, 'data/beta_slopes', 'bhalla')
alon_dir = os.path.join(project_dir, 'data/beta_slopes', 'alon')
figure_dir = os.path.join(project_dir, 'analysis/figures/')
table_dir = os.path.join(project_dir, 'analysis/tables/')

'''
organization of yu chen's data from matlab

There are 21 total components, 10 history, 10 stimulus, 1 bias, with 22 values
of slope and of the maximum liklihood estimator

beta_slopes is an array of shape (21, 22) where axis=0 is each component, and
axis=1 are the values at each of 22 lambda values.

lambda_plot is the array of 22 values where each value is the maximum likelihood
estimator value as a function of the 22 lambda values

'''
'''
# grab file names from my output directory and sort by name
file_list = os.listdir(alon_dir)
file_list.sort()
file_list
'''

class beta_slopes_ml_to_np:
    def __init__(self, slopes_dir, file):
        matlab = sio.loadmat(os.path.join(slopes_dir, file))
        beta_slopes = matlab['beta_slopes']
        self.stim_slopes = beta_slopes[:10]
        self.hist_slopes = beta_slopes[11:]
        self.bias_slopes = beta_slopes[10]
        self.lambda_plot = matlab['lambda_plot']
        self.lambda_id = matlab['lambd_id'][0][0] - 1
        self.channel = file.split('_')[0]
        self.model = slopes_dir.split('/')[-1]
        self.ll_train = matlab['log_likelihood_train']
        self.ll_test = matlab['log_likelihood_test']

    
    def get_optimal_values(self):
        lambda_stim = self.stim_slopes.T[self.lambda_id]
        lambda_hist = self.hist_slopes.T[self.lambda_id]
        lambda_bias = self.bias_slopes.T[self.lambda_id]
        lambda_value = self.lambda_plot[0][self.lambda_id]
        lambda_star = pd.DataFrame({'model': self.model,
                                    'channel': self.channel,
                                    'lambda index': self.lambda_id,
                                    'stimulus': lambda_stim,
                                    'history': lambda_hist,
                                    'bias': lambda_bias,
                                    'lambda value': lambda_value})
        return lambda_star

    def __repr__(self):
        return 'Object containing beta slopes from the {} model {} channel'.format(self.model, self.channel)


def channels_lambda_to_pandas(slopes_dir):
    channel_file_list = os.listdir(slopes_dir)
    channel_file_list.sort()

    model_slopes = pd.DataFrame()

    for file in channel_file_list:
        channel_beta_slopes = beta_slopes_ml_to_np(slopes_dir, file)
        channel_slopes_lambda_star = channel_beta_slopes.get_optimal_values()
        model_slopes = pd.concat([model_slopes, channel_slopes_lambda_star], ignore_index=True)

    model_slopes.set_index(['model', 'channel'], inplace=True)

    return model_slopes

def get_channel_list(model_slopes):
    model_name = model_slopes.index[0][0]
    channel_list = model_slopes.loc[model_name].index.unique().tolist()

    return channel_list

class optimal_slopes:
    def __init__(self, slopes_dir):
        optimal_slopes_df = channels_lambda_to_pandas(slopes_dir)
        self.df = optimal_slopes_df
        self.channels = get_channel_list(optimal_slopes_df)
        self.model = optimal_slopes_df.index[0][0]
    

    def get_column_data(self, data_tag):
        '''
        data_tag = str, 'stimulus', 'history', etc.
        returns a df of the desired data from all channels
        '''
        model_data = pd.DataFrame()

        for channel in self.channels:
            channel_data = self.df.loc[self.model, channel][data_tag].values
            model_data = pd.concat([model_data, pd.DataFrame(channel_data)], axis=1, ignore_index=True)
        
        model_data.columns = self.channels
        return model_data


    def plot_slopes(self, ax, data_tag, s=None, marker=None, cmap=None, vmin=None, vmax=None):
        xt = np.arange(0, 10, 1)
        yt = np.arange(0, len(self.channels), 1)
        xst, yst = np.meshgrid(xt, yt)

        norm = Normalize(vmin, vmax)

        data = self.get_column_data(data_tag).values.T

        ax.scatter(xst, yst, s=s, marker=marker, c=data, cmap=cmap, norm=norm)
        ax.set_ylim(-1, len(self.channels))
        ax.set_yticks(np.arange(0, len(self.channels), 1))
        ax.set_yticklabels(self.channels)
        ax.set_xticks([])

        return ax

    def plot_colorbar(self, fig, axs, cmap=None, vmin=None, vmax=None):
        norm = Normalize(vmin, vmax)

        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs)

        return cb

# TODO
'''
- make good choices for max and min
    - make stimulus and history on different scales, still need to choose vmin, vmax for each, but shouldn't be same
    - this is because they are normalized differently, so can't reasonably compare between the two
    - we will still be able to compare across channels on each and assign which are more or less important within type of filter, but not between
- make sure norm is doing what I think it is
- add a color bar
- combine with other parts of figure
    - stimulus filters, lambda=0 and optimal
    - history filters, lambda=0 and optimal
    - example of how slopes change for a given channel as lambda increases
    - what basis functions represent
    - what the summed slopes represent 
        - e.g. the slope of the weights of a given basis function as conductance changes

    - the change in log likelihood as a function of increasing lambda
    - subtracted optimal stimulus and history filters
- decide on colors for stimulus and history filter colors - use throughout
- should weighting of stimulus filter and history filter differ
- should weigthing be on a log scale?
'''

alon = optimal_slopes(alon_dir)
fig, axs = plt.subplots(figsize=(8, 3), nrows=1, ncols=2, sharey=True, constrained_layout=True)
stim_vmin = 1
stim_vmax = 50
hist_vmin = 1
hist_vmax = 150
stim_plot = alon.plot_slopes(axs[0], 'stimulus', cmap=cm.Purples, vmin=stim_vmin, vmax=stim_vmax)
hist_plot = alon.plot_slopes(axs[1], 'history', cmap=cm.Greens, vmin=hist_vmin, vmax=hist_vmax)
stim_cb = alon.plot_colorbar(fig, stim_plot, cmap=cm.Purples, vmin=stim_vmin, vmax=stim_vmax)
hist_cb = alon.plot_colorbar(fig, hist_plot, cmap=cm.Greens, vmin=hist_vmin, vmax=hist_vmax)

# stim_cb.set_ticks([1., 20., 40., 60., 80., 100.])
# stim_cb.set_ticklabels(['< 1', 20, 40, 60, 80, '> 100'])
hist_cb.set_label('Summed Slopes')

stim_plot.set_title('Stimulus Bases')
hist_plot.set_title('History Bases')
fig.suptitle('{} Model Optimal Summed Slopes'.format('Alon'))



fig.savefig(figure_dir + 'optimal_summed_slopes2.png', format='png', dpi=300)


######### bhalla 
bhalla = optimal_slopes(bhalla_dir)
optimal_bhalla, bhalla_axs = plt.subplots(figsize=(8, 3), nrows=1, ncols=2, sharey=True, constrained_layout=True)
stim_vmin = 1
stim_vmax = 50
hist_vmin = 1
hist_vmax = 150
stim_col = cm.Purples
hist_col = cm.Greens
b_stim_plot = bhalla.plot_slopes(bhalla_axs[0], 'stimulus', cmap=stim_col, vmin=stim_vmin, vmax=stim_vmax)
b_hist_plot = bhalla.plot_slopes(bhalla_axs[1], 'history', cmap=hist_col, vmin=hist_vmin, vmax=hist_vmax)
stim_cb = bhalla.plot_colorbar(optimal_bhalla, b_stim_plot, cmap=stim_col, vmin=stim_vmin, vmax=stim_vmax)
hist_cb = bhalla.plot_colorbar(optimal_bhalla, b_hist_plot, cmap=hist_col, vmin=hist_vmin, vmax=hist_vmax)

# stim_cb.set_ticks([1., 20., 40., 60., 80., 100.])
# stim_cb.set_ticklabels(['< 1', 20, 40, 60, 80, '> 100'])
hist_cb.set_label('Summed Slopes')

b_stim_plot.set_title('Stimulus Bases')
b_hist_plot.set_title('History Bases')
optimal_bhalla.suptitle('{} Model Optimal Summed Slopes'.format('Bhalla'))



optimal_bhalla.savefig(figure_dir + 'bhalla_optimal_summed_slopes2.png', format='png', dpi=300)