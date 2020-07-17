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
wns_file



'''################### Make Plot of Vm, Raster, and PSTH ###################'''
'''################### for all channels in one plot ########################'''
'''################### for 3 Vm plots non overlay ##########################'''

# open any files I need
psth_df = pd.read_csv(table_dir + 'scaled_psth.csv')

# make scale list for reduced plots for ease to see
scale_list = ['1.5', '1', '0.5']
ordered_channels = ['kA', 'kca3', 'lcafixed', 'kslowtab', 'kfasttab', 'nafast']

# set plot style and colormaps
sns.set_style('ticks')
bluered10 = ['#08509b', '#2070b4', '#4191c6', '#6aaed6', '#9dcae1', 'k', '#fca082', '#fb694a', '#e32f27', '#b11218']
bluered3 = [u'#fb694a', 'k', u'#6aaed6']    # for just 3 scales

# create fig and axes for subplot of one channel type
gs_kw = dict(height_ratios=[0.5, 0.7, 0.7, 0.7, 4, 1])
fig, axs = plt.subplots(6, 6, figsize=(16.5, 5.5), gridspec_kw=gs_kw, constrained_layout=True)

for col, channel in zip(range(6), ordered_channels):

    for ax in axs[:, col].flatten():
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    ''' plot first stimulus trace '''
    wns_trial_15 = wns_file.iloc[27000:29000, 15]
    axs[0, col].plot(wns_trial_15, color='k')
    axs[0, col].set_xlim(27000, 29000)
    axs[0, col].set_title(channel)
    # axs[0].clear()

    ''' plot Vm for three scales, 0.5, 1, and 1.5 '''
    for i in range(len(scale_list)):
        vm = channels[channel][scale_list[i]]['Vm']
        vm_trial_15 = vm.iloc[27000:29000, 15]
        axs[i+1, col].plot(vm_trial_15, label=scale_list[i], color=bluered3[i])
        axs[i+1, col].set_ylim(-71, 41)
        axs[i+1, col].set_xlim(27000, 29000)
    # axs[1].clear()

    ''' plot rasters for all scales of 1 channel '''
    sorted_scales = channels[channel].keys()
    sorted_scales.sort()
    # axs[1].set_title('')
    axs[4, col].set_ylim(0, 1011)
    axs[4, col].set_xlim(2700, 2900)
    # ax.set_ylabel('Trial Number', fontsize=16)

    for k, scale in zip(range(len(sorted_scales)), sorted_scales):
        x = channels[channel][scale]['spiketimes']
        for i in range(100):
            xi = x.iloc[:, i]
            y = (k * 101) + (i + 1) * np.ones_like(xi)
            # axs[1].plot(xi, y, marker='_', markersize=1, linestyle='', color=bluered10[k])
            axs[4, col].scatter(xi, y, s=1, marker='.', color=bluered10[k])

        axs[4, col].axhline(y=(k*101), color='gray', linewidth=0.5)
    # axs[2].clear()

    ''' plot psth for 3 scales '''
    channel_psth = psth_df[psth_df['Channel'] == channel]
    for i in range(len(scale_list)):
        scale_psth = channel_psth[channel_psth['Scale'] == float(scale_list[i])]
        psth = scale_psth.iloc[:, 2:].T
        psth.index = np.arange(0, 3110, 1)
        psth_slice = psth.iloc[2705:2905]
        axs[5, col].plot(psth_slice, label=scale_list[i], color=bluered3[i])
    axs[5, col].set_ylim(-8, 168)
    axs[5, col].set_xlim(2705, 2905)

fig
fig.savefig(figure_dir + 'sand_bhalla_wns_vm_raster_psth_3nonoverlay_scaled.png', dpi=300, format='png')


'''################# STA Trace Plots on Single Figure ######################'''
# open STA files if not open already
sta_mean = pd.read_csv(table_dir + 'sta_mean.csv')
sta_std = pd.read_csv(table_dir + 'sta_std.csv')

# settings for x and y axis minor  ticks, the  number stands for delta tick
x_minor = MultipleLocator(5)
y_minor = MultipleLocator(10)

# set time and number of scales and color palette
time = np.linspace(-35, 0.0, num=351)
n_scales = 10
bluered10 = ['#08509b', '#2070b4', '#4191c6', '#6aaed6', '#9dcae1', 'k', '#fca082', '#fb694a', '#e32f27', '#b11218']
ordered_channels = ['kA', 'kca3', 'lcafixed', 'kslowtab', 'kfasttab', 'nafast']

# set the seaborn background style and generate plots
sns.set_style('ticks')

# make a subplot fig with 3 rows 2 cols 8"x8" with autosizing
fig, axs = plt.subplots(1, 6, figsize=(16.5, 2.25), constrained_layout=True)

for ax, channel in zip(axs.flat, ordered_channels):
    # pull out the STAs for each channel type individually
    channel_set = sta_mean[sta_mean['Channel'] == channel]

    # loop through all scales one by one
    for i in range(n_scales):
        scale_slice = channel_set.iloc[i]       # pull out scale row
        scale = scale_slice['Scale']            # define scale
        sta = scale_slice.iloc[3:-50] * 1000    # pull out STA convert to pA
        ax.plot(time, sta, label=scale, color=bluered10[i], linewidth=2)
        # ax.set_title(channel, pad=-14)          # the pad moves the title down
        ax.set_ylim(-25, 55)                    # make all y axes equal
        ax.xaxis.set_minor_locator(x_minor)     # add x axis minor ticks
        ax.yaxis.set_minor_locator(y_minor)     # add y axis minor ticks
        ax.spines["right"].set_visible(False)   # if I want cleaner look w/out
        ax.spines["top"].set_visible(False)     # top and right axes lines
        ax.set_ylabel('Current (pA)')
        ax.set_xlabel('Time (ms)')
        ax.set_xticks([-30, -20, -10, 0])
fig
axs[0].set_ylabel('Current (pA)')
# add legend
axs[5].legend(loc='center left', bbox_to_anchor=(1.1, 0.45))

# save fig to file
fig.savefig(figure_dir + 'sand_STAs_columns_ordered_ylabel_shorter.png', dpi=300, format='png')
==fig.savefig(figure_dir + 'STAs_columns_ordered_labeled.png', dpi=300, format='png')



''' palplot '''
sns.palplot(bluered10)
% matplotlib
colorcode

fig = colorcode.get_figure()

sns.palplot(bluered10)
