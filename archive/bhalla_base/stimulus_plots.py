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
% matplotlib inline
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

# grab file names from my output directory and sort by name
file_list = os.listdir(data_dir)
file_list.sort()

# define channels in channel_list, establish dictionary
channel_list = ['kA', 'kca3', 'kfasttab', 'kslowtab', 'lcafixed', 'nafast']
channels = {item:{} for item in channel_list}
ordered_channels = ['kA', 'kca3', 'lcafixed', 'kslowtab', 'kfasttab', 'nafast']


# open Vm files and save dataframes to dictionary
for file in file_list:
    if file.endswith('.dat'):
        if 'v' in file:
            vm = pd.read_csv(data_dir + file, sep='\s+', header=None)
            channel = file.split('_')[1]
            scale = file.split('_')[2]
            if scale not in channels[channel]:
                channels[channel][scale] = {}
            channels[channel][scale]['filename'] = file
            channels[channel][scale]['Vm'] = vm
            vm = None

'''
 Open saved spiketimes file and save to dictionary, convert to other types
 -The spike times contain all spikes, even those before a certain time,
    If I want to use a spike time with time after x, like 500 ms, use the
    operation: spiketimes.iloc[:, i][spiketimes.iloc[:, i] > 550]
 -The neo spike times are used for making binary/binned spiketrains for use
    with PSTH and with correlations
 -The binned_spiketrains have binary arrays with bin size = bins, and can be
   used for correlations and for calculating the PSTH
'''
for file in file_list:
    if 'spike' in file:
        spiketimes = pd.read_csv(data_dir + file, sep=',')
        channel = file.split('_')[1]
        scale = file.split('_')[2].split('.csv')[0]
        if scale not in channels[channel]:
            channels[channel][scale] = {}
        channels[channel][scale]['spiketimes'] = spiketimes

        # make a neo spiketrain for each column of spiketimes which happens
        # after 550 ms
        neo_spiketimes = []
        for i in range(len(spiketimes.columns)):
            spiketrain = spiketimes.iloc[:, i][spiketimes.iloc[:, i] > 550]
            neo_spiketrain = SpikeTrain(times=spiketrain.values*pq.ms, t_start=550, t_stop=3100)
            neo_spiketimes.append(neo_spiketrain)
        channels[channel][scale]['neo_spiketimes'] = neo_spiketimes

        # convert neo spiketimes to binned spiketrains with x ms bins
        bins = 1.0
        binned_spiketrains = elephant.conversion.BinnedSpikeTrain(neo_spiketimes, binsize=bins*pq.ms, t_start=550.*pq.ms, t_stop= 3100.*pq.ms)
        binned_spiketrains = pd.DataFrame(binned_spiketrains.to_array()).T
        channels[channel][scale]['binned_spiketrains'] = binned_spiketrains


# open the noise file to a data frame
wns_file = pd.read_csv(noise_filename, sep='\t', header=None)
wns = wns_file*1000
vm = channels['kA']['1']['Vm']
spikes = channels['kA']['1']['spiketimes']
wns_mean = wns.mean(axis=1)

''' make plots of 1 stimulus  and 1 full vm record'''
wns_1 = wns.iloc[:31000, 1]
vm_1 = vm.iloc[:, 1]
time = np.linspace(0, 3099.9, 31000)

sns.set_style('ticks')

fig, axs = plt.subplots(2, 1, figsize=(5, 5), constrained_layout=True)

for ax in axs.flat:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

axs[0].plot(time, wns_1, color='k', linewidth=0.75)
axs[0].spines["bottom"].set_visible(False)
axs[0].set_xticks([])
axs[0].set_ylabel('Current (pA)')

axs[1].plot(time, vm_1, color='k', linewidth=0.5)
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Voltage (mV)')

fig.align_ylabels(axs[:])
fig
fig.savefig(figure_dir + 'sand_1stimulus_vm.png', dpi=300, format='png')

''' plot multiple stimulus from 200 ms chunk '''
wns_10 = wns.iloc[27000:29000, :10]
vm_10 = vm.iloc[27000:29000, :10]
wns_mean_slice = wns_mean.iloc[27000:29000]

fig, axs = plt.subplots(3, 1, figsize=(3, 5), constrained_layout=True)

for ax in axs.flat:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    #ax.set_yticks([])
    # ax.set_xlim()

axs[0].plot(wns_10, color='grey', linewidth=0.7)
axs[0].plot(wns_mean_slice, color='k', linewidth=2)

axs[1].plot(vm_10, color='k', linewidth=0.5)

for i in range(len(spikes.columns)):
    xi = spikes.iloc[:, i]
    y = (i + 1) * np.ones_like(xi)
    axs[2].scatter(xi, y, s=1, marker='.', color='k')
    axs[2].set_xlim(2690, 2908)


# save fig to file
fig.savefig(figure_dir + 'sand_stim2.png', dpi=300, format='png')
fig.savefig(figure_dir + 'sand_stim2_xlabels.png', dpi=300, format='png')
fig.savefig(figure_dir + 'sand_stim2_ylabels.png', dpi=300, format='png')

''' correlation plot for kA '''
stats = pd.read_csv(table_dir + 'scaled_spiketrain_stats.csv', index_col=[0, 1])
ka_stats = stats.loc['kA']
correlation = ka_stats['Correlation']
firing_rate = ka_stats['Firing Rate (Hz)']

fig, axs = plt.subplots(2, 1, figsize=(3, 5), constrained_layout=True)

axs[0].plot(firing_rate, color='k', marker='.', markersize=10)
axs[0].set_ylim(0, 45)
# axs[0].set_xlabel('Scaling Factor')
axs[0].set_ylabel('Firing Rate')
axs[0].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[0].spines["right"].set_visible(False)
axs[0].spines["top"].set_visible(False)

axs[1].plot(correlation, color='k', marker='.', markersize=10)
axs[1].set_ylim(0, 0.35)
axs[1].set_xlabel('Scaling Factor')
axs[1].set_ylabel('Correlation')
axs[1].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[1].spines["right"].set_visible(False)
axs[1].spines["top"].set_visible(False)

fig.align_ylabels(axs[:])
fig
fig.savefig(figure_dir + 'sand_correlation.png', dpi=300, format='png')

''' correlation plot for cah from alon '''
table_dir_alon = home_dir + '/ngglasgow@gmail.com/Data_Urban/NEURON/AlmogAndKorngreen2014/par_ModCell_5_thrdsafe/analysis/tables/'

stats = pd.read_csv(table_dir_alon + 'scaled_spiketrain_stats.csv', index_col=[0, 1])
cah_stats = stats.loc['cah']
correlation = cah_stats['Correlation']
firing_rate = cah_stats['Firing Rate (Hz)']

fig, axs = plt.subplots(2, 1, figsize=(3, 5), constrained_layout=True)

axs[0].plot(firing_rate, color='k', marker='.', markersize=10)
axs[0].set_ylim(-4, 86)
# axs[0].set_xlabel('Scaling Factor')
axs[0].set_ylabel('Firing Rate')
axs[0].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[0].spines["right"].set_visible(False)
axs[0].spines["top"].set_visible(False)
axs[0].set_xticks([0, 1, 2, 3])
axs[0].get_ylim()

axs[1].plot(correlation, color='k', marker='.', markersize=10)
axs[1].set_ylim(0, 0.35)
axs[1].set_xlabel('Scaling Factor')
axs[1].set_ylabel('Correlation')
axs[1].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[1].spines["right"].set_visible(False)
axs[1].spines["top"].set_visible(False)
axs[1].set_xticks([0, 1, 2, 3])

fig.align_ylabels(axs[:])
fig
fig.savefig(figure_dir + 'sand_alon_correlation.png', dpi=300, format='png')
