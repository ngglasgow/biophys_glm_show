import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import set_paths
import conversions
import seaborn as sns
import coherence
from matplotlib.ticker import MultipleLocator
import reconstructions

# set paths
home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')

bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')

# open all data
bhalla_recon = reconstructions.Reconstructions(bhalla_paths)
bhalla_cohr = coherence.Coherence(bhalla_paths)

# overall plot params
channel = 'kA'
scales = ['0.5', '1.0', '1.5']
label_size = 14

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot coherences
# gs_kw = dict(height_ratios=[0.4, 1, 1])
pwidth = 2.75
cheight = 3
x_minor = MultipleLocator(25)
panel_labels = ['B', 'C']

fig_cohr, axs_c = plt.subplots(2, 1, figsize=(pwidth, cheight), constrained_layout=True, sharex=True)

cohr = bhalla_cohr.plot_channel_coherence(axs_c[0], channel, scales)
cohr_sub = bhalla_cohr.plot_coherence_subtraction(axs_c[1], channel, scales)

cohr.set_xlabel('')
cohr_sub.xaxis.set_minor_locator(x_minor)
cohr.set_ylim(0, 1)
lim = np.round(np.abs(cohr_sub.get_ylim()).max(), 1)
cohr_sub.set_ylim(-lim, lim)
fig_cohr.align_ylabels()

cohr.annotate(panel_labels[0], xy=(-0.275, 1.2), xycoords= "axes fraction",
                fontsize=label_size, fontweight='bold', va='top', ha='right')

cohr_sub.annotate(panel_labels[1], xy=(-0.275, 1.2), xycoords= "axes fraction",
                fontsize=label_size, fontweight='bold', va='top', ha='right')

fig_cohr_path = os.path.join(bhalla_paths.figures, 'fig5_bc.png')
fig_cohr.savefig(fig_cohr_path, dpi=300, format='png')
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot reconstructions
recon_start, recon_stop = 1350, 1550
rheight = 1
xbar = 50
ybar = 100
panel_label = 'A'

fig_recon, axs_r = plt.subplots(figsize=(pwidth, rheight), constrained_layout=True)
recon = bhalla_recon.plot_channel_reconstructions(axs_r, channel, scales, xlim=(recon_start, recon_stop))
recon.set_axis_off()
ymin = min(recon.get_ylim())
xmin = min(recon.get_xlim())
recon.hlines(y=ymin, xmin=xmin, xmax=xmin + xbar)
recon.vlines(x=xmin, ymin=ymin, ymax=ymin + ybar)
recon.text(xmin + xbar / 2, ymin + ybar * 0.1, f'{ybar} pA', horizontalalignment='center', verticalalignment='bottom')
recon.text(xmin + xbar / 2, ymin - ybar * 0.1, f'{xbar} ms', horizontalalignment='center', verticalalignment='top')

recon.annotate(panel_label, xy=(0.05, 1.1), xycoords= "axes fraction",
                fontsize=label_size, fontweight='bold', va='top', ha='right')

fig_recon_path = os.path.join(bhalla_paths.figures, 'fig5_a.png')
fig_recon.savefig(fig_recon_path, dpi=300, format='png')

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot coherence bands
bheight = cheight + rheight
bwidth = 2.5
x_minor = MultipleLocator(0.5)
y_minor = MultipleLocator(0.25)
panel_labels = ['D', 'E', 'F']

fig_bands, axs_b = plt.subplots(3, 1, figsize=(bwidth, bheight), constrained_layout=True, sharex=True, sharey=True)
bands = bhalla_cohr.plot_coherence_bands(axs_b, channels=[channel], legend=False)

bands[0].xaxis.set_minor_locator(x_minor)
bands[0].yaxis.set_minor_locator(y_minor)

for i, ax in enumerate(bands):
    ax.annotate(panel_labels[i], xy=(-0.275, 1.2), xycoords= "axes fraction",
                fontsize=label_size, fontweight='bold', va='top', ha='right')

fig_bands_path = os.path.join(bhalla_paths.figures, 'fig5_def.png')
fig_bands.savefig(fig_bands_path, dpi=300, format='png')


# get legends for a plot
handles, temp_labels = recon.get_legend_handles_labels()
labels = scales[::-1] + [temp_labels[0]]

recon.legend(handles[::-1], labels, ncol=4, bbox_to_anchor=(0, 1.05), columnspacing=1, loc='lower left', borderaxespad=0, handlelength=1)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plots for single panels of each of the above
# plot coherence
fig, axs = plt.subplots()
axs = bhalla_cohr.plot_channel_coherence(axs, 'kA', ['0.5', '1.0', '1.5'])
fig_path = os.path.join(bhalla_paths.figures, 'b_cohere_new.png')
fig.savefig(fig_path, format='png', dpi=300)

# plot coherence subtraction
fig, axs = plt.subplots()
axs = bhalla_cohr.plot_coherence_subtraction(axs, 'kA', ['0.5', '1.0', '1.5'])
fig_path = os.path.join(bhalla_paths.figures, 'c_cohere_subtraction_new.png')
fig.savefig(fig_path, format='png', dpi=300)

# plot reconstruction
fig, ax = plt.subplots()
recon = bhalla_recon.plot_channel_reconstructions(ax=ax, channel='kA', scales=['0.5', '1.0', '1.5'])
ax.set_axis_off()
fig_path = os.path.join(bhalla_paths.figures, 'a_recon_new.png')
fig.savefig(fig_path, format='png', dpi=300)