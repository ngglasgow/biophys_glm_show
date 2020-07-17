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

'''
organization of yu chen's data from matlab

There are 21 total components, 10 history, 10 stimulus, 1 bias, with 22 values
of slope and of the maximum liklihood estimator

beta_slopes is an array of shape (21, 22) where axis=0 is each component, and
axis=1 are the values at each of 22 lambda values.

lambda_plot is the array of 22 values where each value is the maximum likelihood
estimator value as a function of the 22 lambda values

'''

# grab file names from my output directory and sort by name
file_list = os.listdir(alon_dir)
file_list.sort()
file_list


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


test1 = beta_slopes_ml_to_np(slopes_dir, file_list[0])

#TODO
'''
- make loops that include saving values to file
- use saved files to make plots
- make seperate method to save log likehood values
'''
# xs and ys for grid of stim or history components
x = np.arange(0, 22, 1)
y = np.arange(0, 10, 1)

xx, yy = np.meshgrid(x, y)

'''########################## not transposed ###############################'''
# plots not normalized or clipped at all
cmap = cm.Purples
vmin = None
vmax = None
norm = Normalize(vmin, vmax)
# normalized or clipped plots
cmap = cm.Purples
vmin = 1
vmax = 50
norm = Normalize(vmin, vmax)

fig, axs = plt.subplots(nrows=2, ncols=1)
axs[0].scatter(xx, yy, c=stim_slopes, cmap=cmap, norm=norm)
axs[1].scatter(xx, yy, c=hist_slopes, cmap=cmap, norm=norm)
fig.suptitle(cmap.name + ' | vmin={}:vmax={}'.format(vmin, vmax))


'''########################### all transposed below ########################'''
cmap = cm.Purples
vmin = 1
vmax = 50
norm = Normalize(vmin, vmax)

fig, axs = plt.subplots(nrows=2, ncols=1)
axs[0].scatter(xx, yy, c=stim_slopes, cmap=cmap)
axs[1].scatter(xx, yy, c=hist_slopes, cmap=cmap)
fig.suptitle(cmap.name + ' | vmin={}:vmax={}'.format(vmin, vmax))


# transposed xs and ys for grid of stim or history components
xt = np.arange(0, 10, 1)
yt = np.arange(0, 22, 1)

xst, yst = np.meshgrid(xt, yt)

# mask slope values below 1
slope_mask = 1

stim_masked = np.ma.masked_where(stim_slopes < slope_mask, stim_slopes)
hist_masked = np.ma.masked_where(hist_slopes < slope_mask, hist_slopes)


''' not normalized or clipped with sequential color maps '''
# plots not normalized or clipped at all
cmap = cm.Purples
vmin = None
vmax = None
norm = Normalize(vmin, vmax)
# normalized or clipped plots
cmap = cm.Purples
vmin = 1
vmax = 50
norm = Normalize(vmin, vmax)
# transposed, not normalized
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].scatter(xst, yst, c=stim_slopes.T, cmap=cmap)
axs[1].scatter(xst, yst, c=hist_slopes.T, cmap=cmap)
axs[0].set_ylim(23, -1)
axs[1].set_ylim(23, -1)
fig.suptitle(cmap.name + ' | vmin={}:vmax={}'.format(vmin, vmax))
# transposed, clipped
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].scatter(xst, yst, c=stim_slopes.T, cmap=cmap, norm=norm)
axs[1].scatter(xst, yst, c=hist_slopes.T, cmap=cmap, norm=norm)
axs[0].set_ylim(23, -1)
axs[1].set_ylim(23, -1)
fig.suptitle(cmap.name + ' | vmin={}:vmax={}'.format(vmin, vmax))

''' normalized and masked plots with sequential colors '''
cmap = cm.viridis
vmin = 1
vmax = 30
norm = Normalize(vmin, vmax)
# transposed, clipped
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].scatter(xst, yst, c=stim_masked.T, cmap=cmap, norm=norm)
axs[1].scatter(xst, yst, c=hist_masked.T, cmap=cmap, norm=norm)
axs[0].set_ylim(23, -1)
axs[1].set_ylim(23, -1)
fig.suptitle(cmap.name + ' | vmin={}:vmax={}'.format(vmin, vmax))

''' loop through a bunch of colormaps to see what looks best '''

cmaps_all = [cm.viridis, cm.viridis_r,
             cm.magma, cm.magma_r,
             cm.plasma, cm.plasma_r,
             cm.inferno, cm.inferno_r,
             cm.cividis, cm.cividis_r,
             cm.Greys, cm.Purples, cm.Blues, cm.Greens, cm.Oranges, cm.Reds,
             cm.YlOrBr, cm.YlOrRd, cm.OrRd, cm.PuRd, cm.RdPu, cm.BuPu,
             cm.GnBu, cm.PuBu, cm.YlGnBu, cm.PuBuGn, cm.BuGn, cm.YlGn,
             cm.binary, cm.gist_yarg, cm.gist_gray, cm.gray, cm.bone, cm.pink,
             cm.spring, cm.summer, cm.autumn, cm.winter, cm.cool, cm.Wistia,
             cm.hot, cm.afmhot, cm.gist_heat, cm.copper, cm.gist_earth,
             cm.gist_heat, cm.gist_rainbow]

cmaps_reduced = []

vmin = 0
vmax = 50
norm = Normalize(vmin, vmax)
sns.set_style('white')

for cmap in cmaps:
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(4, 7.5), constrained_layout=True)
    axs[0, 0].scatter(xst, yst, c=stim_slopes.T, cmap=cmap, norm=norm)
    axs[0, 1].scatter(xst, yst, c=hist_slopes.T, cmap=cmap, norm=norm)

    axs[1, 0].scatter(xst, yst, c=stim_masked.T, cmap=cmap, norm=norm)
    axs[1, 1].scatter(xst, yst, c=hist_masked.T, cmap=cmap, norm=norm)

    for ax in axs.flat:
        ax.set_ylim(23, -1)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(cmap.name + ' | vmin={}:vmax={}'.format(vmin, vmax))
    filename = '{}_{}_{}.png'.format(cmap.name, vmin, vmax)
    fig.savefig(os.path.join(figure_dir, filename), format='png', dpi=150)

cmap = cm.Blues
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(4, 7.5), constrained_layout=True)

for ax in axs.flat:
    ax.set_ylim(23, -1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(y=12, color='lightgrey', alpha=0.5, linestyle='--', linewidth=1.0)

axs[0, 0].scatter(xst, yst, c=stim_slopes.T, cmap=cmap, norm=norm)
axs[0, 1].scatter(xst, yst, c=hist_slopes.T, cmap=cmap, norm=norm)
axs[0, 0].set_title('Stimulus Bases')
axs[0, 1].set_title('History Bases')

axs[1, 0].scatter(xst, yst, c=stim_masked.T, cmap=cmap, norm=norm)
axs[1, 1].scatter(xst, yst, c=hist_masked.T, cmap=cmap, norm=norm)
axs[1, 0].set_title('Stimulus Bases')
axs[1, 1].set_title('History Bases')

fig.suptitle('Summed Slopes | {} | vmin={}:vmax={}'.format(cmap.name, vmin, vmax))
