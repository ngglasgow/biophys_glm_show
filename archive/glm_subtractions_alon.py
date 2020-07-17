# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.io as sio

sns.set()
%matplotlib
%matplotlib inline
'''################### Set direcotories and open files #####################'''
# set system type as macOS or linux; set project directory; set downstream dirs
# set all these directory variables first

home_dir = os.path.expanduser("~")
project_dir = home_dir + '/ngglasgow@gmail.com/Data_Urban/NEURON/AlmogAndKorngreen2014/par_ModCell_5_thrdsafe/'
glm_dir = project_dir + 'scaled_wns_glm_survival/'
figure_dir = project_dir + 'analysis/figures/'
table_dir = project_dir + 'analysis/tables/'

''' lambda = 1, no penalty GLMs and subtractions '''
# grab no penalty glm filters
stim_filter = pd.read_csv(table_dir + 'stim_filters.csv', index_col=[0, 1])
hist_filter = pd.read_csv(table_dir + 'history_filters.csv', index_col=[0, 1])
bias = pd.read_csv(table_dir + 'bias_terms.csv', index_col=[0, 1])

# pull out ka stimulus, history, and bias
stim_cah = stim_filter.loc['cah']
hist_cah = hist_filter.loc['cah']
bias_cah = bias.loc['cah']

# reverse time for stimulus filter
len_stim = len(stim_cah.T)
time = np.arange(0, -len_stim, -1)
stim_cah.columns = time

# pull out control for each
control_stim_cah = stim_cah.loc[1.0]
control_hist_cah = hist_cah.loc[1.0]
control_bias_cah = bias_cah.loc[1.0]

# make subtraction dfs for each
stim_cah_sub = stim_cah - control_stim_cah
hist_cah_sub = hist_cah - control_hist_cah
bias_cah_sub = bias_cah - control_bias_cah


''' PLOTS FOR lambda = 1, no penalty GLMs '''
# set up plot properties
bluered10 = ['#08509b', '#2070b4', '#4191c6', '#6aaed6', '#9dcae1', 'k', '#fca082', '#fb694a', '#e32f27', '#b11218']
colors = bluered10
sns.set_style('ticks')
hist_time = np.arange(0, len(hist_cah.loc[1.0]), 1)

scales = stim_cah.index.tolist()

# set up figure
gs_kw = dict(width_ratios=[1, 1, 0.6])
fig, axs = plt.subplots(2, 3, figsize=(9, 5), gridspec_kw=gs_kw, constrained_layout=True)

# stimulus filter
i = 0
for scale in scales:
    axs[0, 0].plot(stim_cah.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[0, 0].set_xlabel('Time (ms)')
axs[0, 0].set_ylabel('logit FR (spikes/ms)')
axs[0, 0].set_xlim(-32, 2)
axs[0, 0].set_ylim(-20, 60)
axs[0, 0].set_title('Stimulus Filters')

# history filter
i = 0
for scale in scales:
    axs[0, 1].plot(hist_time, hist_cah.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[0, 1].set_xlabel('Time (ms)')
# axs[0, 1].set_ylabel('logit FR (spikes/ms)')
axs[0, 1].set_xlim(-5, 155)
axs[0, 1].set_ylim(-12, 10)
axs[0, 1].set_title('History Filters')

# bias
axs[0, 2].plot(bias_cah.index, bias_cah, marker='.', markersize=10, color='k')
axs[0, 2].set_xlabel('Scaling Factor')
axs[0, 2].set_title('Bias')
axs[0, 2].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[0, 2].set_xticks([0, 1, 2, 3])

# stimulus filter subtraction
i = 0
for scale in scales:
    axs[1, 0].plot(stim_cah_sub.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_ylabel(r'$\Delta$ logit FR (scaled-control)')
axs[1, 0].set_xlim(-32, 2)
axs[1, 0].set_ylim(-10, 10)

# history filter subtraction
i = 0
for scale in scales:
    axs[1, 1].plot(hist_time, hist_cah_sub.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[1, 1].set_xlabel('Time (ms)')
# axs[1, 1].set_ylabel(r'$\Delta$ logit FR (scaled-control)')
axs[1, 1].set_xlim(-5, 155)
axs[1, 1].set_ylim(-10, 10)

# bias subtraction
axs[1, 2].plot(bias_cah.index, bias_cah_sub, marker='.', markersize=10, color='k')
axs[1, 2].set_xlabel('Scaling Factor')
axs[1, 2].set_xticks([0, 1, 2, 3])
axs[1, 2].xaxis.set_minor_locator(MultipleLocator(0.5))

for ax in axs.flat:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig.savefig(figure_dir + 'alon_GLMs_lambda1.png', dpi=300, format='png')


''' lambda = *, best penalty GLMs and subtractions '''
# grab lamba* glm filters
glm_l = pd.read_csv(table_dir + 'cah_glm_lambda*.csv', index_col=[0, 1])

# pull out stimulus, history, and bias
stim_l = glm_l.loc['k']
hist_l = glm_l.loc['h']
bias_l = glm_l.loc['dc']['0']

# reverse time for stimulus filter
len_stim = len(stim_l.T)
time = np.arange(0, -len_stim, -1)
stim_l.columns = time

# pull out control for each
control_stim_l = stim_l.loc[1.0]
control_hist_l = hist_l.loc[1.0]
control_bias_l = bias_l.loc[1.0]

# make subtraction dfs for each
stim_l_sub = stim_l - control_stim_l
hist_l_sub = hist_l - control_hist_l
bias_l_sub = bias_l - control_bias_l


''' PLOTS FOR lambda = 1, no penalty GLMs '''
# set up plot properties
bluered10 = ['#08509b', '#2070b4', '#4191c6', '#6aaed6', '#9dcae1', 'k', '#fca082', '#fb694a', '#e32f27', '#b11218']
colors = bluered10
sns.set_style('ticks')
hist_time = np.arange(0, len(hist_l.loc[1.0]), 1)

scales = stim_l.index.tolist()

# set up figure
gs_kw = dict(width_ratios=[1, 1, 0.6])
fig, axs = plt.subplots(2, 3, figsize=(9, 5), gridspec_kw=gs_kw, constrained_layout=True)

# stimulus filter
i = 0
for scale in scales:
    axs[0, 0].plot(stim_l.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[0, 0].set_xlabel('Time (ms)')
axs[0, 0].set_ylabel('logit FR (spikes/ms)')
axs[0, 0].set_xlim(-32, 2)
axs[0, 0].set_ylim(-20, 60)
axs[0, 0].set_title('Stimulus Filters')

# history filter
i = 0
for scale in scales:
    axs[0, 1].plot(hist_time, hist_l.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[0, 1].set_xlabel('Time (ms)')
# axs[0, 1].set_ylabel('logit FR (spikes/ms)')
axs[0, 1].set_xlim(-5, 155)
axs[0, 1].set_ylim(-12, 10)
axs[0, 1].set_title('History Filters')

# bias
axs[0, 2].plot(bias_l.index, bias_l, marker='.', markersize=10, color='k')
axs[0, 2].set_xlabel('Scaling Factor')
axs[0, 2].set_title('Bias')
axs[0, 2].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[0, 2].set_xticks([0, 1, 2, 3])

# stimulus filter subtraction
i = 0
for scale in scales:
    axs[1, 0].plot(stim_l_sub.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_ylabel(r'$\Delta$ logit FR (scaled-control)')
axs[1, 0].set_xlim(-32, 2)
axs[1, 0].set_ylim(-10, 10)

# history filter subtraction
i = 0
for scale in scales:
    axs[1, 1].plot(hist_time, hist_l_sub.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[1, 1].set_xlabel('Time (ms)')
# axs[1, 1].set_ylabel(r'$\Delta$ logit FR (scaled-control)')
axs[1, 1].set_xlim(-5, 155)
axs[1, 1].set_ylim(-10, 10)

# bias subtraction
axs[1, 2].plot(bias_l.index, bias_l_sub, marker='.', markersize=10, color='k')
axs[1, 2].set_xlabel('Scaling Factor')
axs[1, 2].set_xticks([0, 1, 2, 3])
axs[1, 2].xaxis.set_minor_locator(MultipleLocator(0.5))

for ax in axs.flat:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig.savefig(figure_dir + 'alon_GLMs_lambda*.png', dpi=300, format='png')
