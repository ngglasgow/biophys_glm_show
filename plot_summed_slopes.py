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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# pull out relevant data for plotting
# pull out control glm filters
control_stim = filters_0.stim['1']
control_hist = filters_0.hist['1']

# pull out control stim filter weights
control_stim_weights = filters_0.stim_weights['1']

# pull out peaks of basis functions
x_stim_bases_peaks = bases.stim.idxmax()
y_stim = bases.stim.max()
y_stim_bases_peaks = np.ones_like(x_stim_bases_peaks)

x_hist_bases_peaks = bases.hist.idxmax()
y_hist = bases.hist.max()
y_hist_bases_peaks = np.ones_like(x_hist_bases_peaks)

# pull out slope values
lambda_0 = ka_slopes.stim_slopes[:, 0]
lambda_opt = ka_slopes.stim_slopes[:, ka_slopes.lambda_id]
lambda_max = ka_slopes.stim_slopes[:, -1]

# pull out data for all stim and hist and training
stim_data = ka_slopes.stim_slopes.T
hist_data = ka_slopes.hist_slopes.T

test = ka_slopes.ll_test.reshape(-1)
train = ka_slopes.ll_train.reshape(-1)
index = np.arange(0, len(test), 1)
llmin = ka_slopes.ll_test.min()
llmax = ka_slopes.ll_train.max()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# general plot parameters and variables
stim_cmap = cm.Blues
stim_vmin = 1
stim_vmax = 50
stim_norm = Normalize(stim_vmin, stim_vmax)

hist_cmap = cm.Greens
hist_vmin = 1
hist_vmax = 150
hist_norm = Normalize(hist_vmin, hist_vmax)

beta_k_title = r'Stimulus Coefficient Index ($\beta^K_i$)'
beta_h_title = r'History Coefficient Index ($\beta^H_i$)'
beta_k_value = r'$\beta^K_i$ Values'
ss_k_title = r'Summed Slopes ($ss^K$)'
ss_h_title = r'Summed Slopes ($ss^H$)'

x_spread = np.arange(1, 11, 1)
y_spread = np.zeros_like(x_spread)

bases_colors = sns.color_palette('viridis', 10)
sns.set_palette(bases_colors)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot a: plot detailing how filters are made with basis functions and coefficients slopes
# plot a parameters and variables
stim_color = sns.color_palette('Blues')[-1]
ymin = -10
ymax = 25
y_minor = MultipleLocator(10)
x_minor = MultipleLocator(0.5)

k_equation = r'$k = \sum_{i=1}^n k_i\beta^K_i$'
slope_equation = r'$ss^K_i = \sum_{s=1}^{n-1} \frac{\|\beta^K_i(g_s) - \beta^K_i(g_{s+1})\|}{g_{s+1} - g_s}$'

xs = np.stack((x_spread, x_spread))
ys = np.stack((y_spread + 1, y_spread))
lambdas = np.stack((lambda_0, lambda_opt))

# plot
gs_kw = dict(height_ratios=[1, 0.15, 1, 0.4])
fig, axs = plt.subplots(figsize=(6, 6.5), nrows=4, ncols=2, gridspec_kw=gs_kw, constrained_layout=True)

# define axes
axs_bases = axs[0, 1]
axs_coeff = axs[1, 1]
# axs_bar = axs_coeff.twinx()
axs_stim = axs[0, 0]
axs_k_eq = axs[1, 0]
axs_lam_0 = axs[2, 0]
axs_lam_opt = axs[2, 1]
axs_ss_eq = axs[3, 1]
axs_ss = axs[3, 0]

# stimulus bases
axs_bases.plot(bases.stim)
axs_bases.scatter(x_stim_bases_peaks, y_stim, c=bases_colors)
axs_bases.set_title('Stimulus Basis Functions ($k_i$)')
axs_bases.set_ylim(-0.03, 0.62)
axs_bases.set_yticks([])
axs_bases.set_xlabel('Time (ms)')

# coefficients
axs_coeff.scatter(x_spread, y_spread, c=bases_colors)
axs_coeff.set_yticks([])
axs_coeff.set_xticks(x_spread)
axs_coeff.set_xlabel(beta_k_title)
# with values
# axs_bar.bar(x_spread, control_stim_weights, color=bases_colors, alpha=0.3)
# axs_bar.set_ylim(-20, 20)
# axs_bar.set_yticks([])

# stimulus
axs_stim.plot(control_stim, c=stim_color)
axs_stim.set_xlabel('Time (ms)')
axs_stim.set_ylabel('logit FR (spikes/ms)')
axs_stim.set_title('Stimulus Filter ($k$)')
# with bars
widths = x_stim_bases_peaks.diff()
widths[0] = 1
axs_bar2 = axs_stim.twinx()
# uniform bar width
# axs_bar2.bar(x_stim_bases_peaks, control_stim_weights, width=3, color=bases_colors, alpha=0.6)
# make widths scale with width of component
axs_bar2.bar(x_stim_bases_peaks, control_stim_weights, width=widths, color=bases_colors, alpha=0.6)
axs_bar2.set_ylim(-10, 40)
axs_bar2.set_yticks([0])
# axs_bar2.set_yticklabels([])
axs_bar2.set_ylabel(r'$\beta^K_i$ Value')
axs_bar2.hlines(y=0, xmin=-1, xmax=150, alpha=0.6, color='gray', linewidth=1)

# equation
axs_k_eq.set_axis_off()
#axs_stim.text(0.45, 0.65, k_equation, fontsize=13, transform=axs_stim.transAxes) # horizontalalignment='center', verticalalignment='center')
axs_k_eq.text(0.5, -0.8, k_equation, fontsize=14, transform=axs_k_eq.transAxes, horizontalalignment='center', verticalalignment='center')
# do weights here
axs_lam_0.plot(slopes_0.stim_weights)
axs_lam_opt.plot(slopes_opt.stim_weights)

axs_lam_0.set_ylabel(beta_k_value)
axs_lam_0.set_title(r'$\lambda = 0$')
axs_lam_opt.set_title(r'$\lambda = \lambda*$')
axs_lam_opt.set_yticklabels([])

for ax in [axs_lam_0, axs_lam_opt]:
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_minor_locator(y_minor)
    ax.set_xlabel('Conductance Scaling Factor ($g_s$)')
    ax.xaxis.set_minor_locator(x_minor)

axs_ss_eq.set_axis_off()
axs_ss_eq.text(0.5, 0.5, slope_equation, fontsize=13, transform=axs_ss_eq.transAxes, horizontalalignment='center')

axs_ss.scatter(xs, ys, c=lambdas, cmap=stim_cmap, norm=stim_norm)
axs_ss.set_ylabel(r'$\lambda$ Penalty')
axs_ss.set_ylim(-0.5, 1.5)
axs_ss.set_yticks([1, 0])
axs_ss.set_yticklabels(['0', r'$\lambda*$'])
axs_ss.set_xticks(x_spread)
axs_ss.set_xlabel(beta_k_title)

cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs_ss, aspect=30, location='top')# orientation='horizontal', pad=-0.175)
cb.set_label('Summed Slopes ($ss^K$)')
fig.align_ylabels(axs[:, 0])

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_a_bars_w_lower.png')
fig.savefig(fig_path, format='png', dpi=300)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot b:
bluered10 = ['#08509b', '#2070b4', '#4191c6', '#6aaed6', '#9dcae1', 'k', '#fca082', '#fb694a', '#e32f27', '#b11218']

colors = bluered10
scales = filters_0.stim.columns
cb_aspect = 30

xt = np.arange(1, 11, 1)
yt = np.arange(0, 22, 1)

xst, yst = np.meshgrid(xt, yt)

gs_kw = dict(height_ratios=[0.5, 0.5, 1], width_ratios=[1, 1, 0.5])
fig, axs = plt.subplots(figsize=(7, 7), nrows=3, ncols=3, gridspec_kw=gs_kw, constrained_layout=True)

i = 0
for scale in scales:
    axs[0, 0].plot(filters_0.stim[scale], color=colors[i], linewidth=2, label=scale)
    i += 1
axs[0, 0].set_ylim(-1, 8)
axs[0, 0].set_title('MC $\mathregular{K_A}$ Stimulus')
axs[0, 0].set_ylabel('logit FR')

i = 0
for scale in scales:
    axs[0, 1].plot(filters_0.hist[scale], color=colors[i], linewidth=2, label=scale)
    i += 1
axs[0, 1].set_ylim(-15, 5)
axs[0, 1].set_title('MC $\mathregular{K_A}$ History')

i = 0
for scale in scales:
    axs[1, 0].plot(filters_opt.stim[scale], color=colors[i], linewidth=2, label=scale)
    i += 1
axs[1, 0].set_ylim(-1, 8)
axs[1, 0].set_ylabel('logit FR')

i = 0
for scale in scales:
    axs[1, 1].plot(filters_opt.hist[scale], color=colors[i], linewidth=2, label=scale)
    i += 1
axs[1, 1].set_ylim(-15, 5)

# axs[0, 0].set_xlabel('Time (ms)')
# axs[0, 1].set_xlabel('Time (ms)')
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 1].set_xlabel('Time (ms)')
axs[0, 0].set_xticklabels([])
axs[0, 1].set_xticklabels([])

# draw legends
handles, labels = axs[0, 0].get_legend_handles_labels()
handles_rev = handles[::-1]
labels_rev = labels[::-1]

axs[0, 2].set_axis_off()
# axs[0, 2].legend(handles_rev, labels_rev, fontsize=10, frameon=False, markerscale=0.5, ncol=2, title='Scaling Factor', handlelength=1.0, columnspacing=1.0)
# axs[0, 2].legend(handles_rev, labels_rev, fontsize=10, frameon=False, markerscale=0.5, title='Scaling Factor', handlelength=1.0, bbox_transform=axs[1, 2].transAxes)
fig.legend(handles_rev, labels_rev, loc=(0.77, 0.6), fontsize=10, frameon=False, markerscale=0.5, title='Conductance Scaling\n        Factor ($g_s$)', handlelength=1.0)

axs[1, 2].set_axis_off()
# axs[1, 2].legend(handles_rev, labels_rev, fontsize=10, frameon=False, markerscale=0.5, ncol=2, title='Scaling Factor', handlelength=1.0, columnspacing=1.0)

axs[2, 0].scatter(xst, yst, c=stim_data, cmap=stim_cmap, norm=stim_norm)
stim_cb = fig.colorbar(cm.ScalarMappable(stim_norm, stim_cmap), ax=axs[2, 0], location='top', aspect=cb_aspect)
stim_cb.set_label(r'MC $\mathregular{K_A}$ $ss^K$')

axs[2, 1].scatter(xst, yst, c=hist_data, cmap=hist_cmap, norm=hist_norm)
hist_cb = fig.colorbar(cm.ScalarMappable(hist_norm, hist_cmap), ax=axs[2, 1], location='top', aspect=cb_aspect)
hist_cb.set_label(r'MC $\mathregular{K_A}$ $ss^H$')

axs[2, 2].plot(train, index, c='k', label='train')
axs[2, 2].plot(test, index, c='grey', label='test')
axs[2, 2].legend(loc='upper left', fontsize=10, markerscale=0.5, frameon=False, handlelength=1.0)

for ax in axs[2, :]:
    ax.set_ylim(22, -1)
    ax.set_yticks([0, ka_slopes.lambda_id, 21])
    ax.set_yticklabels(['0', '$\lambda*$', 'max'])
    ax.axhline(y=ka_slopes.lambda_id, color='grey', alpha=0.5, linestyle='--', linewidth=1.0)

axs[2, 0].set_ylabel(r'$\lambda$ Penalty')
axs[2, 0].set_xlabel(r'Stimulus Coefficient Index ($\beta^K_i$)')
axs[2, 0].set_xticks(x_spread)

axs[2, 1].set_xlabel(r'History Coefficient Index ($\beta^H_i$)')
axs[2, 1].set_xticks(x_spread)

axs[2, 2].set_xticks([llmin, llmax])
axs[2, 2].set_xticklabels(['min', 'max'])
axs[2, 2].set_xlabel('Log Likelihood')

axs[0, 0].text(0.7, 0.8, r'$\lambda = 0$', transform=axs[0, 0].transAxes)
axs[0, 1].text(0.1, 0.8, r'$\lambda = 0$', transform=axs[0, 1].transAxes)

axs[1, 0].text(0.7, 0.8, r'$\lambda = \lambda*$', transform=axs[1, 0].transAxes)
axs[1, 1].text(0.1, 0.8, r'$\lambda = \lambda*$', transform=axs[1, 1].transAxes)

fig.align_ylabels(axs[:, 0])

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_b_filters_top_smaller.png')
fig.savefig(fig_path, format='png', dpi=300)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot c: 
# TODO
# need to add in for all figures a way to say max and min slopes in labels automatically
cb_aspect = 5

fig, axs = plt.subplots(figsize=(13, 2.5), nrows=1, ncols=4, constrained_layout=True)

stim_plot_b = bhalla_slopes.plot_slopes(axs[0], 'stim', cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax)
stim_cb_b = fig.colorbar(cm.ScalarMappable(norm=stim_norm, cmap=stim_cmap), ax=stim_plot_b, location='top', aspect=cb_aspect)

hist_plot_b = bhalla_slopes.plot_slopes(axs[1], 'hist', cmap=hist_cmap, vmin=hist_vmin, vmax=hist_vmax)
hist_cb_b = fig.colorbar(cm.ScalarMappable(norm=hist_norm, cmap=hist_cmap), ax=hist_plot_b, location='top', aspect=cb_aspect)

stim_cb_b.set_label('MC $ss^K$')
hist_cb_b.set_label('MC $ss^H$')

stim_plot_a = alon_slopes.plot_slopes(axs[2], 'stim', cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax)
stim_cb_a = fig.colorbar(cm.ScalarMappable(norm=stim_norm, cmap=stim_cmap), ax=stim_plot_a, location='top', aspect=cb_aspect)

hist_plot_a = alon_slopes.plot_slopes(axs[3], 'hist', cmap=hist_cmap, vmin=hist_vmin, vmax=hist_vmax)
hist_cb_a = fig.colorbar(cm.ScalarMappable(norm=hist_norm, cmap=hist_cmap), ax=hist_plot_a, location='top', aspect=cb_aspect)

stim_cb_a.set_label('PC $ss^K$')
hist_cb_a.set_label('PC $ss^H$')

fig_path = os.path.join(bhalla_paths.figures, 'summed_slopes_c_shorttitle_small.png')
fig.savefig(fig_path, format='png', dpi=300)
