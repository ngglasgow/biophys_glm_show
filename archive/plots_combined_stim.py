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
alon_slopes = conversions.optimal_slopes(alon_paths)

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

stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

ymin = -10
ymax = 25
y_minor = MultipleLocator(10)

k_equation = r'$k = \sum_{i=1}^n k_i\beta^K_i$'
slope_equation = r'$ss = \sum_{i=1}^{n-1} \|\|\beta(g_i) - \beta(g_{i+1})\|\|$'
beta_k = r'$\beta^K_i$'

x_minor = MultipleLocator(0.5)

gs_kw = dict(height_ratios=[0.3, 1, 1, 0.3])
fig, axs = plt.subplots(figsize=(6, 6.5), nrows=4, ncols=2, gridspec_kw=gs_kw, constrained_layout=True)

# stimulus plots
axs[1, 1].plot(bases.stim)
axs[1, 1].scatter(x_stim_bases_peaks, y_stim, c=spread_colors)
axs[1, 1].set_title('Stimulus Basis Functions ($k_i$)')
axs[1, 1].set_ylim(-0.03, 0.62)
axs[1, 1].set_yticks([])
axs[1, 1].set_xlabel('Time (ms)')

# axs[1, 0].scatter(x_stim_bases_peaks, y_stim_bases_peaks, c=spread_colors)
axs[0, 1].scatter(x_spread, y_spread, c=spread_colors)
axs[0, 1].set_yticks([])
axs[0, 1].set_xticks(x_spread)
axs[0, 1].set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

axs[1, 0].plot(control_stim, c=stim_color)
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_ylabel('logit FR (spikes/ms)')
axs[1, 0].set_title('Stimulus Filter ($k$)')

axs[0, 0].set_axis_off()
axs[0, 0].text(0.5, -0.5, k_equation, fontsize=14, transform=axs[0, 0].transAxes, horizontalalignment='center')

# do weights here
axs[2, 0].plot(slopes_0.stim_weights)
axs[2, 1].plot(slopes_opt.stim_weights)

axs[2, 0].set_ylabel('Coefficient Values')
axs[2, 0].set_title(r'$\lambda = 0$')
axs[2, 1].set_title(r'$\lambda = *$')
axs[2, 1].set_yticklabels([])

for ax in axs[2, :]:
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_minor_locator(y_minor)
    ax.set_xlabel('Scaling Factor ($g_i$)')
    ax.xaxis.set_minor_locator(x_minor)

for ax in axs[3, :]:
    ax.set_xticks(x_spread)
    ax.set_yticks([])
    ax.set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

axs[3, 0].set_title(r'$\lambda = 0$')
axs[3, 1].set_title(r'$\lambda = *$')
axs[3, 0].text(0.5, 2.5, slope_equation, fontsize=12, transform=axs[3, 0].transAxes, horizontalalignment='center')

lambda_0 = ka_slopes.stim_slopes[:, 0]
lambda_opt = ka_slopes.stim_slopes[:, ka_slopes.lambda_id]
lambda_max = ka_slopes.stim_slopes[:, -1]

axs[3, 0].scatter(x_spread, y_spread, c=lambda_0, cmap=stim_cmap, norm=stim_norm)
axs[3, 1].scatter(x_spread, y_spread, c=lambda_opt, cmap=stim_cmap, norm=stim_norm)

cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs[3, 1], aspect=30, location='top')
cb.set_label('Summed Slopes ($ss$)')

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_a_horiz.png')
fig.savefig(fig_path, format='png', dpi=300)

'''##############################################################################'''
''' ############ plots of basis functions, peaks, and filters ###################'''
'''############################### VERTICAL #####################################'''
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

xs = np.stack((x_spread, x_spread))
ys = np.stack((y_spread + 1, y_spread))

lambda_0 = ka_slopes.stim_slopes[:, 0]
lambda_opt = ka_slopes.stim_slopes[:, ka_slopes.lambda_id]
lambdas = np.stack((lambda_0, lambda_opt))

stim_color = 'blue'
hist_color = 'green'
spread_colors = cm.tab10.colors

stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

ymin = -10
ymax = 25
y_minor = MultipleLocator(10)

k_equation = r'$k = \sum_{i=1}^n k_i\beta^K_i$'
slope_equation = r'$ss = \sum_{i=1}^{n-1} \|\|\beta(g_i) - \beta(g_{i+1})\|\|$'
beta_k = r'$\beta^K_i$'

x_minor = MultipleLocator(0.5)

gs_kw = dict(height_ratios=[0.3, 1, 1, 0.6])
fig, axs = plt.subplots(figsize=(6, 6.5), nrows=4, ncols=2, gridspec_kw=gs_kw, constrained_layout=True)

# stimulus plots
axs[1, 1].plot(bases.stim)
axs[1, 1].scatter(x_stim_bases_peaks, y_stim, c=spread_colors)
axs[1, 1].set_title('Stimulus Basis Functions ($k_i$)')
axs[1, 1].set_ylim(-0.03, 0.62)
axs[1, 1].set_yticks([])
axs[1, 1].set_xlabel('Time (ms)')

# axs[1, 0].scatter(x_stim_bases_peaks, y_stim_bases_peaks, c=spread_colors)
axs[0, 1].scatter(x_spread, y_spread, c=spread_colors)
axs[0, 1].set_yticks([])
axs[0, 1].set_xticks(x_spread)
axs[0, 1].set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

axs[1, 0].plot(control_stim, c=stim_color)
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_ylabel('logit FR (spikes/ms)')
axs[1, 0].set_title('Stimulus Filter ($k$)')

axs[0, 0].set_axis_off()
axs[0, 0].text(0.5, -0.5, k_equation, fontsize=14, transform=axs[0, 0].transAxes, horizontalalignment='center')

# do weights here
axs[2, 0].plot(slopes_0.stim_weights)
axs[2, 1].plot(slopes_opt.stim_weights)

axs[2, 0].set_ylabel('Coefficient Values')
axs[2, 0].set_title(r'$\lambda = 0$')
axs[2, 1].set_title(r'$\lambda = *$')
axs[2, 1].set_yticklabels([])

for ax in axs[2, :]:
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_minor_locator(y_minor)
    ax.set_xlabel('Scaling Factor ($g_i$)')
    ax.xaxis.set_minor_locator(x_minor)

axs[3, 0].set_axis_off()
axs[3, 0].text(0.5, 0.5, slope_equation, fontsize=12, transform=axs[3, 0].transAxes, horizontalalignment='center')

axs[3, 1].scatter(xs, ys, c=lambdas, cmap=stim_cmap, norm=stim_norm)
axs[3, 1].set_ylabel('$\lambda$')
axs[3, 1].set_ylim(-0.5, 1.5)
axs[3, 1].set_yticks([1, 0])
axs[3, 1].set_yticklabels(['0', '*'])
axs[3, 1].set_xticks(x_spread)
axs[3, 1].set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs[3, 0], aspect=30, orientation='horizontal', pad=-0.175)
cb.set_label('Summed Slopes ($ss$)')

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_a_vertical.png')
fig.savefig(fig_path, format='png', dpi=300)



'''##############################################################################'''
''' ############ plots of basis functions, peaks, and filters ###################'''
'''###################### VERTICAL VERTICAL #####################################'''
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

xs = np.stack((x_spread, x_spread))
ys = np.stack((y_spread + 1, y_spread))

lambda_0 = ka_slopes.stim_slopes[:, 0]
lambda_opt = ka_slopes.stim_slopes[:, ka_slopes.lambda_id]
lambdas = np.stack((lambda_0, lambda_opt))

stim_color = 'blue'
hist_color = 'green'
spread_colors = cm.tab10.colors

stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

ymin = -10
ymax = 25
y_minor = MultipleLocator(10)

k_equation = r'$k = \sum_{i=1}^n k_i\beta^K_i$'
slope_equation = r'$ss = \sum_{i=1}^{n-1} \|\|\beta(g_i) - \beta(g_{i+1})\|\|$'
beta_k = r'$\beta^K_i$'

x_minor = MultipleLocator(0.5)

gs_kw = dict(height_ratios=[0.4, 1, 1, 1])
fig, axs = plt.subplots(figsize=(6, 6.5), nrows=4, ncols=2, gridspec_kw=gs_kw, constrained_layout=True)

# stimulus plots
axs[1, 1].plot(bases.stim)
axs[1, 1].scatter(x_stim_bases_peaks, y_stim, c=spread_colors)
axs[1, 1].set_title('Stimulus Basis Functions ($k_i$)')
axs[1, 1].set_ylim(-0.03, 0.62)
axs[1, 1].set_yticks([])
axs[1, 1].set_xlabel('Time (ms)')

# axs[1, 0].scatter(x_stim_bases_peaks, y_stim_bases_peaks, c=spread_colors)
axs[0, 1].scatter(x_spread, y_spread, c=spread_colors)
axs[0, 1].set_yticks([])
axs[0, 1].set_xticks(x_spread)
axs[0, 1].set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

axs[1, 0].plot(control_stim, c=stim_color)
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_ylabel('logit FR (spikes/ms)')
axs[1, 0].set_title('Stimulus Filter ($k$)')

axs[0, 0].set_axis_off()
axs[0, 0].text(0.5, -0.3, k_equation, fontsize=14, transform=axs[0, 0].transAxes, horizontalalignment='center')

# do weights here
axs[2, 0].plot(slopes_0.stim_weights)
axs[3, 0].plot(slopes_opt.stim_weights)

axs[2, 0].set_title(r'$\lambda = 0$')
axs[2, 0].set_xticklabels([])
axs[3, 0].set_title(r'$\lambda = *$')
axs[3, 0].set_xlabel('Scaling Factor ($g_i$)')

for ax in axs[2:, 0]:
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Coefficient Values')
    ax.yaxis.set_minor_locator(y_minor)
    ax.xaxis.set_minor_locator(x_minor)

axs[2, 1].set_axis_off()
axs[2, 1].text(0.5, 0.75, slope_equation, fontsize=12, transform=axs[2, 1].transAxes, horizontalalignment='center')
cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs[2, 1], aspect=30, orientation='horizontal', pad=-0.3)
cb.set_label('Summed Slopes ($ss$)')

axs[3, 1].scatter(xs, ys, c=lambdas, cmap=stim_cmap, norm=stim_norm)
axs[3, 1].set_ylabel('$\lambda$')
axs[3, 1].set_ylim(-0.5, 1.5)
axs[3, 1].set_yticks([1, 0])
axs[3, 1].set_yticklabels(['0', '*'])
axs[3, 1].set_xticks(x_spread)
axs[3, 1].set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_a_vertical_vertical.png')
fig.savefig(fig_path, format='png', dpi=300)



'''##############################################################################'''
''' ############ plots of basis functions, peaks, and filters ###################'''
'''###################### VERTICAL VERTICAL FLIP   ##############################'''
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

xs = np.stack((x_spread, x_spread))
ys = np.stack((y_spread + 1, y_spread))

lambda_0 = ka_slopes.stim_slopes[:, 0]
lambda_opt = ka_slopes.stim_slopes[:, ka_slopes.lambda_id]
lambdas = np.stack((lambda_0, lambda_opt))

stim_color = 'blue'
hist_color = 'green'
spread_colors = cm.tab10.colors

stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

ymin = -10
ymax = 25
y_minor = MultipleLocator(10)

k_equation = r'$k = \sum_{i=1}^n k_i\beta^K_i$'
slope_equation = r'$ss = \sum_{i=1}^{n-1} \|\|\beta(g_i) - \beta(g_{i+1})\|\|$'
beta_k = r'$\beta^K_i$'

x_minor = MultipleLocator(0.5)

gs_kw = dict(height_ratios=[0.4, 1, 1, 1])
fig, axs = plt.subplots(figsize=(6, 6.5), nrows=4, ncols=2, gridspec_kw=gs_kw, constrained_layout=True)

# stimulus plots
axs[1, 1].plot(bases.stim)
axs[1, 1].scatter(x_stim_bases_peaks, y_stim, c=spread_colors)
axs[1, 1].set_title('Stimulus Basis Functions ($k_i$)')
axs[1, 1].set_ylim(-0.03, 0.62)
axs[1, 1].set_yticks([])
axs[1, 1].set_xlabel('Time (ms)')

# axs[1, 0].scatter(x_stim_bases_peaks, y_stim_bases_peaks, c=spread_colors)
axs[0, 1].scatter(x_spread, y_spread, c=spread_colors)
axs[0, 1].set_yticks([])
axs[0, 1].set_xticks(x_spread)
axs[0, 1].set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

axs[1, 0].plot(control_stim, c=stim_color)
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_ylabel('logit FR (spikes/ms)')
axs[1, 0].set_title('Stimulus Filter ($k$)')

axs[0, 0].set_axis_off()
axs[0, 0].text(0.5, -0.3, k_equation, fontsize=14, transform=axs[0, 0].transAxes, horizontalalignment='center')

# do weights here
axs[2, 1].plot(slopes_0.stim_weights)
axs[3, 1].plot(slopes_opt.stim_weights)

axs[2, 1].set_title(r'$\lambda = 0$')
axs[2, 1].set_xticklabels([])
axs[3, 1].set_title(r'$\lambda = *$')
axs[3, 1].set_xlabel('Scaling Factor ($g_i$)')

for ax in axs[2:, 1]:
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Coefficient Values')
    ax.yaxis.set_minor_locator(y_minor)
    ax.xaxis.set_minor_locator(x_minor)

axs[2, 0].set_axis_off()
axs[2, 0].text(0.5, 0.75, slope_equation, fontsize=12, transform=axs[2, 0].transAxes, horizontalalignment='center')
cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs[2, 0], aspect=30, orientation='horizontal', pad=-0.3)
cb.set_label('Summed Slopes ($ss$)')

axs[3, 0].scatter(xs, ys, c=lambdas, cmap=stim_cmap, norm=stim_norm)
axs[3, 0].set_ylabel(r'$\lambda$')
axs[3, 0].set_ylim(-0.5, 1.5)
axs[3, 0].set_yticks([1, 0])
axs[3, 0].set_yticklabels(['0', '*'])
axs[3, 0].set_xticks(x_spread)
axs[3, 0].set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_a_vertical_vertical_flip.png')
fig.savefig(fig_path, format='png', dpi=300)

'''##############################################################################'''
''' ############ plots of basis functions, peaks, and filters ###################'''
'''###################### VERTICAL VERTICAL FLIP FLIP ###########################'''
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

xs = np.stack((x_spread, x_spread))
ys = np.stack((y_spread + 1, y_spread))

lambda_0 = ka_slopes.stim_slopes[:, 0]
lambda_opt = ka_slopes.stim_slopes[:, ka_slopes.lambda_id]
lambdas = np.stack((lambda_0, lambda_opt))

stim_color = 'blue'
hist_color = 'green'
spread_colors = cm.tab10.colors

stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

ymin = -10
ymax = 25
y_minor = MultipleLocator(10)

k_equation = r'$k = \sum_{i=1}^n k_i\beta^K_i$'
slope_equation = r'$ss = \sum_{i=1}^{n-1} \|\|\beta(g_i) - \beta(g_{i+1})\|\|$'
beta_k = r'$\beta^K_i$'

x_minor = MultipleLocator(0.5)

gs_kw = dict(height_ratios=[0.4, 1, 1, 1])
fig, axs = plt.subplots(figsize=(6, 6.5), nrows=4, ncols=2, gridspec_kw=gs_kw, constrained_layout=True)

# stimulus plots
axs[1, 0].plot(bases.stim)
axs[1, 0].scatter(x_stim_bases_peaks, y_stim, c=spread_colors)
axs[1, 0].set_title('Stimulus Basis Functions ($k_i$)')
axs[1, 0].set_ylim(-0.03, 0.62)
axs[1, 0].set_yticks([])
axs[1, 0].set_xlabel('Time (ms)')

# axs[1, 0].scatter(x_stim_bases_peaks, y_stim_bases_peaks, c=spread_colors)
axs[0, 0].scatter(x_spread, y_spread, c=spread_colors)
axs[0, 0].set_yticks([])
axs[0, 0].set_xticks(x_spread)
axs[0, 0].set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

axs[1, 1].plot(control_stim, c=stim_color)
axs[1, 1].set_xlabel('Time (ms)')
axs[1, 1].set_ylabel('logit FR (spikes/ms)')
axs[1, 1].set_title('Stimulus Filter ($k$)')

axs[0, 1].set_axis_off()
axs[0, 1].text(0.5, -0.3, k_equation, fontsize=14, transform=axs[0, 1].transAxes, horizontalalignment='center')

# do weights here
axs[2, 0].plot(slopes_0.stim_weights)
axs[3, 0].plot(slopes_opt.stim_weights)

axs[2, 0].set_title(r'$\lambda = 0$')
axs[2, 0].set_xticklabels([])
axs[3, 0].set_title(r'$\lambda = *$')
axs[3, 0].set_xlabel('Scaling Factor ($g_i$)')

for ax in axs[2:, 0]:
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Coefficient Values')
    ax.yaxis.set_minor_locator(y_minor)
    ax.xaxis.set_minor_locator(x_minor)

axs[2, 1].set_axis_off()
axs[2, 1].text(0.5, 0.75, slope_equation, fontsize=12, transform=axs[2, 1].transAxes, horizontalalignment='center')
cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs[2, 1], aspect=30, orientation='horizontal', pad=-0.3)
cb.set_label('Summed Slopes ($ss$)')

axs[3, 1].scatter(xs, ys, c=lambdas, cmap=stim_cmap, norm=stim_norm)
axs[3, 1].set_ylabel(r'$\lambda$')
axs[3, 1].set_ylim(-0.5, 1.5)
axs[3, 1].set_yticks([1, 0])
axs[3, 1].set_yticklabels(['0', '*'])
axs[3, 1].set_xticks(x_spread)
axs[3, 1].set_xlabel('Stimulus Coefficient Index (' + beta_k + ')')

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_a_vertical_vertical_flip_flip.png')
fig.savefig(fig_path, format='png', dpi=300)

'''###########################################################################'''
'''################### Bhalla kA filters and slops with lambda ###############'''
'''###########################################################################'''
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

# create aribitrarily spaced basis function peaks for weights
x_spread = np.arange(1, 11, 1)
y_spread = np.zeros_like(x_spread)

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

# draw legends
handles, labels = axs[0, 0].get_legend_handles_labels()
handles_rev = handles[::-1]
labels_rev = labels[::-1]

axs[0, 2].set_axis_off()
axs[0, 2].legend(handles_rev, labels_rev, fontsize=10, frameon=False, markerscale=0.5, ncol=2, title='Scaling Factor', handlelength=1.0, columnspacing=1.0)

axs[2, 2].set_axis_off()
axs[2, 2].legend(handles_rev, labels_rev, fontsize=10, frameon=False, markerscale=0.5, ncol=2, title='Scaling Factor', handlelength=1.0, columnspacing=1.0)

axs[1, 0].scatter(xst, yst, c=stim_data, cmap=stim_cmap, norm=stim_norm)
stim_cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs[1, 0], orientation='horizontal')
stim_cb.set_label('Abs. Summed Slopes')

axs[1, 1].scatter(xst, yst, c=hist_data, cmap=hist_cmap, norm=hist_norm)
hist_cb = fig.colorbar(cm.ScalarMappable(hist_norm, hist_cmap), ax=axs[1, 1], orientation='horizontal')
hist_cb.set_label('Abs. Summed Slopes')

axs[1, 2].plot(train, index, c='k', label='train')
axs[1, 2].plot(test, index, c='grey', label='test')
axs[1, 2].legend(loc='upper left', fontsize=10, markerscale=0.5, frameon=False, handlelength=1.0)

for ax in axs.flat[3:6]:
    ax.set_ylim(22, -1)
    ax.set_yticks([0, ka_slopes.lambda_id, 21])
    ax.set_yticklabels(['0', '*', 'max'])
    ax.axhline(y=ka_slopes.lambda_id, color='grey', alpha=0.5, linestyle='--', linewidth=1.0)

axs[1, 0].set_ylabel(r'$\lambda$ Value')
axs[1, 0].set_xlabel('Stimulus Coefficient Index')
axs[1, 0].set_xticks(x_spread)

axs[1, 1].set_xlabel('History Coefficient Index')
axs[1, 1].set_xticks(x_spread)

axs[1, 2].set_xticks([llmin, llmax])
axs[1, 2].set_xticklabels(['min', 'max'])
axs[1, 2].set_xlabel('Log Likelihood')

axs[0, 0].text(0.7, 0.8, r'$\lambda = 0$', transform=axs[0, 0].transAxes)
axs[0, 1].text(0.1, 0.8, r'$\lambda = 0$', transform=axs[0, 1].transAxes)

axs[2, 0].text(0.7, 0.8, r'$\lambda = *$', transform=axs[2, 0].transAxes)
axs[2, 1].text(0.1, 0.8, r'$\lambda = *$', transform=axs[2, 1].transAxes)

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_b.png')
fig.savefig(fig_path, format='png', dpi=300)

'''####################### all optimal slopes plots ###########################'''
stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

hist_cmap = cm.Greens
hist_vmin = 1
hist_vmax = 150
hist_norm = Normalize(hist_vmin, hist_vmax)

fig, axs = plt.subplots(figsize=(8, 6), nrows=2, ncols=2, sharey=True, constrained_layout=True)
stim_plot_b = bhalla_slopes.plot_slopes(axs[0, 0], 'stim', cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax)
stim_cb_b = fig.colorbar(cm.ScalarMappable(norm=stim_norm, cmap=stim_cmap), ax=stim_plot_b, aspect=40)

hist_plot_b = bhalla_slopes.plot_slopes(axs[0, 1], 'hist', cmap=hist_cmap, vmin=hist_vmin, vmax=hist_vmax)
hist_cb_b = fig.colorbar(cm.ScalarMappable(norm=hist_norm, cmap=hist_cmap), ax=hist_plot_b, aspect=40)

stim_plot_b.set_title('Stimulus Summed Slopes Bhalla')
hist_plot_b.set_title('History Summed Slopes Bhalla')
hist_cb_b.set_label('Summed Slopes ($ss$)')

stim_plot_a = alon_slopes.plot_slopes(axs[1, 0], 'stim', cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax)
stim_cb_a = fig.colorbar(cm.ScalarMappable(norm=stim_norm, cmap=stim_cmap), ax=stim_plot_a, aspect=40)

hist_plot_a = alon_slopes.plot_slopes(axs[1, 1], 'hist', cmap=hist_cmap, vmin=hist_vmin, vmax=hist_vmax)
hist_cb_a = fig.colorbar(cm.ScalarMappable(norm=hist_norm, cmap=hist_cmap), ax=hist_plot_a, aspect=40)

stim_plot_a.set_title('Stimulus Summed Slopes Alon')
hist_plot_a.set_title('History Summed Slopes Alon')
hist_cb_a.set_label('Summed Slopes ($ss$)')

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_c.png')
fig.savefig(fig_path, format='png', dpi=300)

test, tax = plt.subplots(1, 2, constrained_layout=True)
i = 0
for scale in scales:
    tax[0].plot(filters_0.stim[scale], color=colors[i], linewidth=2, label=scale)
    i += 1
tax[0].set_ylim(-1, 8)
tax[0].set_title('Stimulus')
tax[0].set_ylabel('logit FR (spikes/ms)')

handles, labels = tax[0].get_legend_handles_labels()
handles_rev = handles[::-1]
labels_rev = labels[::-1]

tax[1].set_axis_off()
tax[1].legend(handles_rev, labels_rev, fontsize=8, frameon=False, markerscale=0.5, ncol=2, title='Scaling Factor')

test