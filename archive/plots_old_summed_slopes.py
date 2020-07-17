import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import set_paths
import conversions
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# set paths
home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')

bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')

# open all data
filters_0 = conversions.filters(bhalla_paths, 'kA', '0')
filters_opt = conversions.filters(bhalla_paths, 'kA', '*')
bases = conversions.bases(bhalla_paths)
slopes_0 = conversions.example_weight_slopes(bhalla_paths, 'kA_beta_curves_lambda_0.mat')
slopes_max = conversions.example_weight_slopes(bhalla_paths, 'kA_beta_curves_lambda_largest.mat')
slopes_opt = conversions.example_weight_slopes(bhalla_paths, 'kA_beta_curves_lambda_star.mat')
ka_slopes = conversions.beta_slopes(bhalla_paths, 'kA')
bhalla_slopes = conversions.optimal_slopes(bhalla_paths)

''' ############ plots of basis functions, peaks, and filters ###################'''
# pull out control glm filters
control_stim = filters_0.stim['1']
control_hist = filters_0.hist['1']

# pull out peaks of basis functions
x_stim_bases_peaks = bases.stim.idxmax()
y_stim = bases.stim.max()
y_stim_bases_peaks = np.ones_like(x_stim_bases_peaks)

x_hist_bases_peaks = bases.hist.idxmax()
y_hist = bases.hist.max()
y_hist_bases_peaks = np.ones_like(x_hist_bases_peaks)

# create aribitrarily spaced basis function peaks for weights
x_spread = np.arange(1, 11, 1)
y_spread = np.zeros_like(x_spread)

stim_color = 'blue'
hist_color = 'green'
spread_colors = cm.tab10.colors

gs_kw = dict(height_ratios=[1, 0.3, 1])
fig, axs = plt.subplots(figsize=(6, 4), nrows=3, ncols=2, gridspec_kw=gs_kw, constrained_layout=True)

# stimulus plots
axs[0, 0].plot(bases.stim)
axs[0, 0].scatter(x_stim_bases_peaks, y_stim, c=spread_colors)
axs[0, 0].set_title('Stimulus Basis Functions')
axs[0, 0].set_xticklabels([])

# axs[1, 0].scatter(x_stim_bases_peaks, y_stim_bases_peaks, c=spread_colors)
axs[1, 0].scatter(x_spread, y_spread, c=spread_colors)
axs[1, 0].set_yticks([])
axs[1, 0].set_xticks(x_spread)
axs[1, 0].set_xlabel('Stimulus Coefficient Index')

axs[2, 0].plot(control_stim, c=stim_color)
axs[2, 0].set_xlabel('Time (ms)')
axs[2, 0].set_ylabel('logit FR (spikes/ms)')
axs[2, 0].set_title('Stimulus Filter')

# history plots
axs[0, 1].plot(bases.hist)
axs[0, 1].scatter(x_hist_bases_peaks, y_hist, c=spread_colors)
axs[0, 1].set_title('History Basis Functions')
axs[0, 1].set_xticklabels([])

# axs[1, 1].scatter(x_hist_bases_peaks, y_hist_bases_peaks, c=spread_colors)
axs[1, 1].scatter(x_spread, y_spread, c=spread_colors)
axs[1, 1].set_yticks([])
axs[1, 1].set_xticks(x_spread)
axs[1, 1].set_xlabel('History Coefficient Index')

axs[2, 1].plot(control_hist, c=hist_color)
axs[2, 1].set_ylim(-15, 5)
axs[2, 1].set_xlabel('Time (ms)')
axs[2, 1].set_title('History Filter')

'''####################### plots of example slopes with lambda ################'''
stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

ymin = -10
ymax = 25

x_minor = MultipleLocator(0.5)

gs_kw = dict(height_ratios=[1, 0.3])
fig, axs = plt.subplots(figsize=(9, 3), nrows=2, ncols=3, gridspec_kw=gs_kw, constrained_layout=True)

axs[0, 0].plot(slopes_0.stim_weights)
axs[0, 1].plot(slopes_opt.stim_weights)
axs[0, 2].plot(slopes_max.stim_weights)

axs[0, 0].set_ylabel('Coefficient Values')
axs[0, 0].set_title('\u03BB = 0')
axs[0, 1].set_title('\u03BB = *')
axs[0, 2].set_title('\u03BB = Max')

for ax in axs.flat[:3]:
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Scaling Factor')
    ax.xaxis.set_minor_locator(x_minor)

for ax in axs.flat[1:3]:
    ax.set_yticklabels([])

for ax in axs.flat[3:]:
    ax.set_xticks(x_spread)
    ax.set_yticks([])
    ax.set_xlabel('Stimulus Coefficient Index')

lambda_0 = ka_slopes.stim_slopes[:, 0]
lambda_opt = ka_slopes.stim_slopes[:, ka_slopes.lambda_id]
lambda_max = ka_slopes.stim_slopes[:, -1]

axs[1, 0].scatter(x_spread, y_spread, c=lambda_0, cmap=stim_cmap, norm=stim_norm)
axs[1, 1].scatter(x_spread, y_spread, c=lambda_opt, cmap=stim_cmap, norm=stim_norm)
axs[1, 2].scatter(x_spread, y_spread, c=lambda_max, cmap=stim_cmap, norm=stim_norm)

cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs, orientation='vertical', aspect=40)
cb.set_label('Abs. Summed Slopes')

'''####################### Bhalla kA slopes all lambda plot ###################'''
'''stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)
stim_data = ka_slopes.stim_slopes.T

hist_cmap = cm.Greens
hist_vmin = 1
hist_vmax = 150
hist_norm = Normalize(hist_vmin, hist_vmax)
hist_data = ka_slopes.hist_slopes.T

test = ka_slopes.ll_test.reshape(-1)
train = ka_slopes.ll_train.reshape(-1)
index = np.arange(0, len(test), 1)

xt = np.arange(0, 10, 1)
yt = np.arange(0, 22, 1)

xst, yst = np.meshgrid(xt, yt)

fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)

axs[0].scatter(xst, yst, c=stim_data, cmap=stim_cmap, norm=stim_norm)
axs[1].scatter(xst, yst, c=hist_data, cmap=hist_cmap, norm=hist_norm)
axs[2].plot(train, index, c='k', label='train')
axs[2].plot(test, index, c='grey', label='test')
axs[2].legend()

for ax in axs.flat:
    ax.set_ylim(23, -1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(y=ka_slopes.lambda_id, color='grey', alpha=0.5, linestyle='--', linewidth=1.0)

axs[0].set_title('Stimulus Slopes')
axs[1].set_title('History Slopes')
axs[2].set_title('Log Likelihood')

axs[0].set_ylabel('\u03BB Increasing \u2192', rotation=270, labelpad=16)
'''

'''################### Bhalla kA filters and slops with lambda ###############'''
bluered10 = ['#08509b', '#2070b4', '#4191c6', '#6aaed6', '#9dcae1', 'k', '#fca082', '#fb694a', '#e32f27', '#b11218']
colors = bluered10
scales = filters_0.stim.columns

stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)
stim_data = ka_slopes.stim_slopes.T

hist_cmap = cm.Greens
hist_vmin = 1
hist_vmax = 150
hist_norm = Normalize(hist_vmin, hist_vmax)
hist_data = ka_slopes.hist_slopes.T

test = ka_slopes.ll_test.reshape(-1)
train = ka_slopes.ll_train.reshape(-1)
index = np.arange(0, len(test), 1)
llmin = ka_slopes.ll_test.min()
llmax = ka_slopes.ll_train.max()

xt = np.arange(1, 11, 1)
yt = np.arange(0, 22, 1)

xst, yst = np.meshgrid(xt, yt)

gs_kw = dict(height_ratios=[0.5, 1, 0.5], width_ratios=[1, 1, 0.5])
fig, axs = plt.subplots(figsize=(7.5, 8), nrows=3, ncols=3, gridspec_kw=gs_kw, constrained_layout=True)

i = 0
for scale in scales:
    axs[0, 0].plot(filters_0.stim[scale], color=colors[i], linewidth=2, label=scale)
    i += 1
axs[0, 0].set_ylim(-1, 8)
axs[0, 0].set_title('Stimulus')
axs[0, 0].set_ylabel('logit FR (spikes/ms)')

i = 0
for scale in scales:
    axs[0, 1].plot(filters_0.hist[scale], color=colors[i], linewidth=2, label=scale)
    i += 1
axs[0, 1].set_ylim(-15, 5)
axs[0, 1].set_title('History')

i = 0
for scale in scales:
    axs[2, 0].plot(filters_opt.stim[scale], color=colors[i], linewidth=2, label=scale)
    i += 1
axs[2, 0].set_ylim(-1, 8)
axs[2, 0].set_ylabel('logit FR (spikes/ms)')

i = 0
for scale in scales:
    axs[2, 1].plot(filters_opt.hist[scale], color=colors[i], linewidth=2, label=scale)
    i += 1
axs[2, 1].set_ylim(-15, 5)

axs[0, 0].set_xlabel('Time (ms)')
axs[0, 1].set_xlabel('Time (ms)')
axs[2, 0].set_xlabel('Time (ms)')
axs[2, 1].set_xlabel('Time (ms)')

axs[0, 2].set_axis_off()
axs[2, 2].set_axis_off()

axs[1, 0].scatter(xst, yst, c=stim_data, cmap=stim_cmap, norm=stim_norm)
stim_cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs[1, 0], orientation='horizontal')
stim_cb.set_label('Abs. Summed Slopes')

axs[1, 1].scatter(xst, yst, c=hist_data, cmap=hist_cmap, norm=hist_norm)
hist_cb = fig.colorbar(cm.ScalarMappable(hist_norm, hist_cmap), ax=axs[1, 1], orientation='horizontal')
hist_cb.set_label('Abs. Summed Slopes')

axs[1, 2].plot(train, index, c='k', label='train')
axs[1, 2].plot(test, index, c='grey', label='test')
axs[1, 2].legend(loc='upper left')

for ax in axs.flat[3:6]:
    ax.set_ylim(22, -1)
    ax.set_yticks([0, ka_slopes.lambda_id, 21])
    ax.set_yticklabels([])
    ax.axhline(y=ka_slopes.lambda_id, color='grey', alpha=0.5, linestyle='--', linewidth=1.0)

axs[1, 0].set_yticklabels(['0', '*', 'max'])
axs[1, 0].set_ylabel('\u03BB Value')
axs[1, 0].set_xlabel('Stimulus Coefficient Index')
axs[1, 0].set_xticks(x_spread)

axs[1, 1].set_xlabel('History Coefficient Index')
axs[1, 1].set_xticks(x_spread)

axs[1, 2].set_xticks([llmin, llmax])
axs[1, 2].set_xticklabels(['min', 'max'])
axs[1, 2].set_xlabel('Log Likelihood')

axs[0, 2].text(0.5, 0.5, '\u03BB = 0', horizontalalignment='center', verticalalignment='center')
axs[2, 2].text(0.5, 0.5, '\u03BB = *', horizontalalignment='center', verticalalignment='center')


'''####################### all optimal slopes plots ###########################'''
stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

hist_cmap = cm.Greens
hist_vmin = 1
hist_vmax = 150
hist_norm = Normalize(hist_vmin, hist_vmax)

fig, axs = plt.subplots(figsize=(8, 3), nrows=1, ncols=2, sharey=True, constrained_layout=True)
stim_plot = bhalla_slopes.plot_slopes(axs[0], 'stim', cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax)
stim_cb = fig.colorbar(cm.ScalarMappable(norm=stim_norm, cmap=stim_cmap), ax=stim_plot, aspect=40)

hist_plot = bhalla_slopes.plot_slopes(axs[1], 'hist', cmap=hist_cmap, vmin=hist_vmin, vmax=hist_vmax)
hist_cb = fig.colorbar(cm.ScalarMappable(norm=hist_norm, cmap=hist_cmap), ax=hist_plot, aspect=40)

stim_plot.set_title('Stimulus Summed Slopes')
hist_plot.set_title('History Summed Slopes')
hist_cb.set_label('Abs. Summed Slopes')

