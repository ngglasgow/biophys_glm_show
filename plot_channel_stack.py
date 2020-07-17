import os
import pandas as pd
import set_paths
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import spiketrains
import conversions
import scipy
import biophys_output
import filters
import numpy as np
import reconstructions

# set paths
home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')

bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')

# plot parameters stolen from scaled_wns_analysis_bluered.py
trial = [15]
start = 2700
stop = 2900
xlim = (start, stop)
recon_xlim = (start - 650, stop - 650)
channel = 'kA'
scales = ['1.5', '1', '0.5']

# open data
bhalla_biophys = spiketrains.SpikeTrains(bhalla_paths, 'biophys')
bhalla_biophys.open_psth()

data = biophys_output.BiophysOutput(bhalla_paths, channel, scales, trial)
bhalla_recon = reconstructions.Reconstructions(bhalla_paths)

# make the plot
gs_kw = dict(height_ratios=[0.5, 0.7, 0.7, 0.7, 4, 1])
fig, axs = plt.subplots(6, 1, figsize=(2.75, 5.5), gridspec_kw=gs_kw, constrained_layout=True)

stimulus = data.plot_im(axs[0], trial, xlim)
for ax, scale in zip(axs[1:4], scales):
    scale = str(float(scale))
    data.plot_vm(ax, scale, trial, xlim)

raster = bhalla_biophys.plot_all_rasters(axs[4], channel, 'none', xlim)

for scale in scales:
    scale = str(float(scale))
    bhalla_biophys.plot_psth(axs[5], channel, scale, 'none', xlim)

stim_xlim = stimulus.get_xlim()
for ax in axs.flat:
    ax.set_axis_off()
    ax.set_xlim(stim_xlim)

fig_path = os.path.join(bhalla_paths.figures, 'kA_stack.png')
fig.savefig(fig_path, format='png', dpi=300)

'''
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#plot stim and history filters for kA
'''
filters_0 = conversions.filters(bhalla_paths, 'kA', '0')

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(2.5, 6), constrained_layout=True)

stim = filters.plot_filters(axs[0], filters_0, 'stim', 'all')
hist = filters.plot_filters(axs[1], filters_0, 'hist', 'all')
scales_index = np.array(filters_0.bias.columns).astype(float)
bias = axs[2].plot(scales_index, filters_0.bias.T, color='k', marker='.')

stim.set_ylim(-1, 8)
stim.set_ylabel('logit FR (spikes/ms)')
stim.set_title('Stimulus Filters')
stim.set_xlabel('Time (ms)')

hist.set_ylabel('logit FR (spikes/ms)')
hist.set_ylim(-15, 5)
hist.set_title('History Filters')
hist.set_xlabel('Time (ms)')

axs[2].set_xlabel('Conductance Scaling Factor')
axs[2].set_ylabel('logit FR (spikes/ms)')
axs[2].set_title('Bias')

for ax in axs.flat:
    ax.spines["right"].set_visible(False)   
    ax.spines["top"].set_visible(False)

fig_path = os.path.join(bhalla_paths.figures, 'stim_hist_filters.png')
fig.savefig(fig_path, format='png', dpi=300)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# we had talked about adding reconstructions to this figure, but I didn't do it
# plot reconstuctions
fig, axs = plt.subplots()
recon = bhalla_recon.plot_channel_reconstructions(axs, channel, scales=['0.5', '1.0', '1.5'], xlim=recon_xlim)
