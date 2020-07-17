# -*- coding: utf-8 -*-
import os
import numpy as np
from neo.core import SpikeTrain, AnalogSignal
import quantities as pq
import pandas as pd
import elephant
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sklearn

sns.set()
%matplotlib
# set system type as macOS or linux; set project directory; set downstream dirs
# set all these directory variables first

home_dir = os.path.expanduser("~")
project_dir = home_dir + '/ngglasgow@gmail.com/Data_Urban/NEURON/Bhalla_par_scaling/'
data_dir = project_dir + 'scaled_wns_500_9520_output/'
figure_dir = project_dir + 'analysis/figures/'
table_dir = project_dir + 'analysis/tables/'
noise_dir = project_dir + 'scaled_wns_500/'
noise_filename = noise_dir + 'wns_9520_forscaled.txt'
downsample_dir = project_dir + 'bhalla_scaled_9520_downsampled/'

# define channels in channel_list, establish dictionary
file_path = data_dir + 'wns_kA_1_v.dat'
spikes_path = data_dir + 'spiketimes_kA_1.csv'

vm = pd.read_csv(file_path, sep='\s+', header=None)
spikes = pd.read_csv(spikes_path, sep=',')
wns_file = pd.read_csv(noise_filename, sep='\t', header=None)
psth_df = pd.read_csv(table_dir + 'scaled_psth.csv', index_col=[0, 1])
psth = psth_df.loc['kA', 1]
psth.index = range(3110)
%matplotlib inline
# create fig and axes for subplot of one channel type
gs_kw = dict(height_ratios=[0.5, 0.7, 1, 1])
fig, axs = plt.subplots(4, 1, figsize=(2, 3.5), gridspec_kw=gs_kw, constrained_layout=True)

for ax in axs[:].flatten():
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

wns_trial_15 = wns_file.iloc[27000:29000, 15]
axs[0].plot(wns_trial_15, color='k')
axs[0].set_xlim(27000, 29000)

vm_trial_15 = vm.iloc[27000:29000, 15]
axs[1].plot(vm_trial_15, color='k')
axs[1].set_ylim(-71, 41)
axs[1].set_xlim(27000, 29000)

for i in range(100):
    xi = spikes.iloc[:, i]
    y = (i + 1) * np.ones_like(xi)
    axs[2].scatter(xi, y, s=1, marker='.', color='k')
axs[2].set_ylim(0, 101)
axs[2].set_xlim(2700, 2900)

psth_slice = psth.iloc[2705:2905]
axs[3].plot(psth_slice, color='k')
axs[3].set_ylim(-8, 168)
axs[3].set_xlim(2705, 2905)

fig
fig.savefig(figure_dir + 'sand_schematic.png', dpi=300, format='png')

''' glms '''
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
control_bias_ka = bias_ka.loc[1.0]
hist_time = np.arange(0, len(hist_ka.loc[1.0]), 1)

sns.set_style('ticks')

gs_kw = dict(width_ratios=[1, 1, 0.5])
fig, axs = plt.subplots(1, 3, figsize=(4, 1.6), gridspec_kw=gs_kw, constrained_layout=True)

axs[0].plot(control_stim_ka, linewidth=2, color='b')
axs[0].set_xlim(-47, 3)
axs[0].set_ylim(0, 10)
axs[0].set_title('Stimulus Filter')


axs[1].plot(hist_time, control_hist_ka, linewidth=2, color='r')
axs[1].set_xlim(-5, 155)
axs[1].set_ylim(-12, 2)
axs[1].set_title('History Filter')

axs[2].plot(control_bias_ka, color='k', marker='.')
axs[2].set_title('Bias')

for ax in axs.flat:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
fig.savefig(figure_dir + 'sand_schematic_glm.png', dpi=300, format='png')
