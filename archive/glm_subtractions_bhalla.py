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
# project_dir = home_dir + '/ngglasgow@gmail.com/Data_Urban/NEURON/Bhalla_par_scaling/'
# glm_dir = project_dir + 'scaled_wns_glm_survival/'
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')
glm_dir = os.path.join(project_dir, 'data/glm_survival')
figure_dir = project_dir + 'analysis/figures/'
table_dir = project_dir + 'analysis/tables/'
bhalla_glm = os.path.join(glm_dir, 'bhalla')

test_file = os.path.join(bhalla_glm, 'kA.mat')

test_matlab = sio.loadmat(test_file)
test_matlab['model_list']

# open stimulus filter
structure = test_matlab['model_list'][0]
n_scales = len(structure)

stim_time_index = pd.DataFrame({'parameter': 'kt', 'scale': 1.0}, index=range(1))
stim_time_index = pd.MultiIndex.from_frame(stim_time_index)
stim_time = pd.DataFrame(matlab['model_list'][0, 0]['kt'][0][0], columns=stim_time_index)

hist_time_index = pd.DataFrame({'parameter': 'ht', 'scale': 1.0}, index=range(1))
hist_time_index = pd.MultiIndex.from_frame(hist_time_index)
hist_time = pd.DataFrame(matlab['model_list'][0, 0]['ht'][0][0], columns=hist_time_index)

data_list = ['k', 'h', 'dc']
glm_df = pd.DataFrame()
for data in data_list:
    data_df = pd.DataFrame()
    labels_df = pd.DataFrame()
    for i in range(n_scales):
        parameter = matlab['model_list'][0, i][data][0][0]
        parameter = pd.DataFrame(parameter)
        scale = matlab['model_list'][0, i]['channel_scalar'][0][0][0][0]
        label = pd.DataFrame({'parameter': data, 'scale': scale}, index=range(1))
        labels_df = pd.concat([labels_df, label])
        data_df = pd.concat([data_df, parameter], axis=1)
    data_index = pd.MultiIndex.from_frame(labels_df)
    data_df.columns = data_index
    glm_df = pd.concat([glm_df, data_df], axis=1)
glm_df = pd.concat([stim_time, hist_time, glm_df], axis=1)

# save transposed table so that it is easier to open and index with
glm_df.T.to_csv(table_dir + 'kA_glm_lambda*.csv', float_format='%8.4f')

''' lambda = 1, no penalty GLMs and subtractions '''
# grab no penalty glm filters
stim_filter = pd.read_csv(table_dir + 'stim_filters.csv', index_col=[0, 1])
hist_filter = pd.read_csv(table_dir + 'history_filters.csv', index_col=[0, 1])
bias = pd.read_csv(table_dir + 'bias_terms.csv', index_col=[0, 1])

# pull out ka stimulus, history, and bias
stim_ka = stim_filter.loc['kA']
hist_ka = hist_filter.loc['kA']
bias_ka = bias.loc['kA']

# reverse time for stimulus filter
len_stim = len(stim_ka.T)
time = np.arange(0, -len_stim, -1)
stim_ka.columns = time

# pull out control for each
control_stim_ka = stim_ka.loc[1.0]
control_hist_ka = hist_ka.loc[1.0]
control_bias_ka = bias_ka.loc[1.0]rame({'parameter': data, 'scale': scale}, index=range(1))
        labels_df = pd.concat([labels_df, label])
        data_df = pd.concat([data_df, parameter], axis=1)
    data_index = pd.MultiIndex.from_frame(labels_df)
    data_df.columns = data_index
    glm_df = pd.concat([glm_df, data_df], axis=1)
glm_df = pd.concat([stim_time, hist_time, glm_df], axis=1)


# make subtraction dfs for each
stim_ka_sub = stim_ka - control_stim_ka
hist_ka_sub = hist_ka - control_hist_ka
bias_ka_sub = bias_ka - control_bias_ka


''' PLOTS FOR lambda = 1, no penalty GLMs '''
# set up plot properties
bluered10 = ['#08509b', '#2070b4', '#4191c6', '#6aaed6', '#9dcae1', 'k', '#fca082', '#fb694a', '#e32f27', '#b11218']
colors = bluered10
sns.set_style('ticks')
hist_time = np.arange(0, len(hist_ka.loc[1.0]), 1)

scales = stim_ka.index.tolist()

# set up figure
gs_kw = dict(width_ratios=[1, 1, 0.6])
fig, axs = plt.subplots(2, 3, figsize=(9, 5), gridspec_kw=gs_kw, constrained_layout=True)

# stimulus filter
i = 0
for scale in scales:
    axs[0, 0].plot(stim_ka.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[0, 0].set_xlabel('Time (ms)')
axs[0, 0].set_ylabel('logit FR (spikes/ms)')
axs[0, 0].set_xlim(-47, 3)
axs[0, 0].set_ylim(0, 10)
axs[0, 0].set_title('Stimulus Filters')

# history filter
i = 0
for scale in scales:
    axs[0, 1].plot(hist_time, hist_ka.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[0, 1].set_xlabel('Time (ms)')
# axs[0, 1].set_ylabel('logit FR (spikes/ms)')
axs[0, 1].set_xlim(-5, 155)
axs[0, 1].set_ylim(-12, 2)
axs[0, 1].set_title('History Filters')

# bias
axs[0, 2].plot(bias_ka.index, bias_ka, marker='.', markersize=10, color='k')
axs[0, 2].set_xlabel('Scaling Factor')
axs[0, 2].set_title('Bias')
axs[0, 2].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[0, 2].set_xticks([0, 1, 2, 3])

# stimulus filter subtraction
i = 0
for scale in scales:
    axs[1, 0].plot(stim_ka_sub.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_ylabel(r'$\Delta$ logit FR (scaled-control)')
axs[1, 0].set_xlim(-47, 3)
axs[1, 0].set_ylim(-3, 3)

# history filter subtraction
i = 0
for scale in scales:
    axs[1, 1].plot(hist_time, hist_ka_sub.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[1, 1].set_xlabel('Time (ms)')
# axs[1, 1].set_ylabel(r'$\Delta$ logit FR (scaled-control)')
axs[1, 1].set_xlim(-5, 155)
axs[1, 1].set_ylim(-10, 5)

# bias subtraction
axs[1, 2].plot(bias_ka.index, bias_ka_sub, marker='.', markersize=10, color='k')
axs[1, 2].set_xlabel('Scaling Factor')
axs[1, 2].set_xticks([0, 1, 2, 3])
axs[1, 2].xaxis.set_minor_locator(MultipleLocator(0.5))

for ax in axs.flat:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig.savefig(figure_dir + 'bhalla_GLMs_lambda1.png', dpi=300, format='png')


''' lambda = *, best penalty GLMs and subtractions '''
# grab lamba* glm filters
glm_l = pd.read_csv(table_dir + 'kA_glm_lambda*.csv', index_col=[0, 1])

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
axs[0, 0].set_xlim(-47, 3)
axs[0, 0].set_ylim(0, 10)
axs[0, 0].set_title('Stimulus Filters')

# history filter
i = 0
for scale in scales:
    axs[0, 1].plot(hist_time, hist_l.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[0, 1].set_xlabel('Time (ms)')
# axs[0, 1].set_ylabel('logit FR (spikes/ms)')
axs[0, 1].set_xlim(-5, 155)
axs[0, 1].set_ylim(-12, 2)
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
axs[1, 0].set_xlim(-47, 3)
axs[1, 0].set_ylim(-3, 3)

# history filter subtraction
i = 0
for scale in scales:
    axs[1, 1].plot(hist_time, hist_l_sub.loc[scale], color=colors[i], linewidth=2)
    i += 1
axs[1, 1].set_xlabel('Time (ms)')
# axs[1, 1].set_ylabel(r'$\Delta$ logit FR (scaled-control)')
axs[1, 1].set_xlim(-5, 155)
axs[1, 1].set_ylim(-10, 5)

# bias subtraction
axs[1, 2].plot(bias_l.index, bias_l_sub, marker='.', markersize=10, color='k')
axs[1, 2].set_xlabel('Scaling Factor')
axs[1, 2].set_xticks([0, 1, 2, 3])
axs[1, 2].xaxis.set_minor_locator(MultipleLocator(0.5))

for ax in axs.flat:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig.savefig(figure_dir + 'bhalla_GLMs_lambda*.png', dpi=300, format='png')
