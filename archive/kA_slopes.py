# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.io as sio
from matplotlib import cm
from matplotlib.colors import Normalize


'''################### Set direcotories and open files #####################'''
# set system type as macOS or linux; set project directory; set downstream dirs
# set all these directory variables first

home_dir = os.path.expanduser("~")
project_path = 'projects/biophys_glm_show'
project_dir = os.path.join(home_dir, project_path)
bhalla_dir = os.path.join(project_dir, 'data/beta_slopes', 'bhalla')
data_dir = os.path.join(project_dir, 'data/example_slopes_kA')
figure_dir = os.path.join(project_dir, 'analysis/figures/')
table_dir = os.path.join(project_dir, 'analysis/tables/')
glm_dir = os.path.join(project_dir, 'data/glm_survival')
glm_bhalla_dir = os.path.join(glm_dir, 'bhalla')


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
        self.ll_train = matlab['log_likelihood_train']
        self.ll_test = matlab['log_likelihood_test']

    
    def get_optimal_values(self):
        lambda_stim = self.stim_slopes.T[self.lambda_id]
        lambda_hist = self.hist_slopes.T[self.lambda_id]
        lambda_bias = self.bias_slopes.T[self.lambda_id]
        lambda_value = self.lambda_plot[0][self.lambda_id]
        lambda_star = pd.DataFrame({'stimulus': lambda_stim,
                                    'history': lambda_hist,
                                    'bias': lambda_bias,
                                    'channel': self.channel,
                                    'lambda index': self.lambda_id,
                                    'lambda value': lambda_value})
        return lambda_star
    
    def __repr__(self):
        return 'Absolute summed change in slopes for {} '.format(self.channel)

'''####################### pull out lambda=0 filters #######################'''
# grab no penalty glm filters
stim_filter = pd.read_csv(os.path.join(glm_bhalla_dir, 'stim_filters.csv'), index_col=[0, 1])
hist_filter = pd.read_csv(os.path.join(glm_bhalla_dir, 'history_filters.csv'), index_col=[0, 1])
bias = pd.read_csv(os.path.join(glm_bhalla_dir, 'bias_terms.csv'), index_col=[0, 1])

# pull out kA lambda=0 filters
stim_ka = stim_filter.loc['kA']
hist_ka = hist_filter.loc['kA']
bias_ka = bias.loc['kA']

'''####################### pull out lambda=* filters########################'''
opt_filters = pd.read_csv(os.path.join(glm_bhalla_dir, 'kA_glm_lambda*.csv'), index_col=[0, 1])
opt_stim_ka = opt_filters.loc['k']
opt_hist_ka = opt_filters.loc['h']
opt_bias_ka = opt_filters.loc['dc']

'''####################### pull out beta slopes for kA #####################'''
bhalla_files = [file for file in os.listdir(bhalla_dir)]
bhalla_files.sort()
kA = beta_slopes_ml_to_np(bhalla_dir, bhalla_files[0])

# pull out test and train data and shape, and make index for plotting
test = kA.ll_test.reshape(-1)
train = kA.ll_train.reshape(-1)
index = np.arange(0, len(test), 1)

# transposed xs and ys for grid of stim or history components
xt = np.arange(0, 10, 1)
yt = np.arange(0, 22, 1)

xst, yst = np.meshgrid(xt, yt)

# plots not normalized or clipped at all
stim_cmap = cm.Purples
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

hist_cmap = cm.Greens
hist_vmin = 1
hist_vmax = 150
hist_norm = Normalize(hist_vmin, hist_vmax)

# uniform normalization parameters
vmin = 1
vmax = 50
norm = Normalize(vmin, vmax)

# stim_plot = alon.plot_slopes(axs[0], 'stimulus', cmap=cm.Purples, vmin=stim_vmin, vmax=stim_vmax)
# hist_plot = alon.plot_slopes(axs[1], 'history', cmap=cm.Greens, vmin=hist_vmin, vmax=hist_vmax)
# stim_cb = alon.plot_colorbar(fig, stim_plot, cmap=cm.Purples, vmin=stim_vmin, vmax=stim_vmax)
# hist_cb = alon.plot_colorbar(fig, hist_plot, cmap=cm.Greens, vmin=hist_vmin, vmax=hist_vmax)

fig, axs = plt.subplots(nrows=1, ncols=3)
axs[0].scatter(xst, yst, c=kA.stim_slopes.T, cmap=stim_cmap, norm=stim_norm, )    
axs[1].scatter(xst, yst, c=kA.hist_slopes.T, cmap=hist_cmap, norm=hist_norm)
axs[2].plot(train, index, c='k')
axs[2].plot(test, index, c='grey')

for ax in axs.flat:
    ax.set_ylim(23, -1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(y=kA.lambda_id, color='lightgrey', alpha=1, linestyle='--', linewidth=1.0)
