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

'''########################## Make Vm Plots ################################'''
# plot the same vm trace between 2700 and 3000 for 3 scales
# do the same time window for rasters
# do the same time window and scales for psth

sns.set_style('ticks')
scale_list = ['0.5', '1', '1.5']
for channel in channels:
    plt.figure()
    plt.title(channel)
    for scale in scale_list:
        vm = channels[channel][scale]['Vm']
        vm_trial_00 = vm.iloc[27000:30000, 0]
        plt.plot(vm_trial_00, label=scale)


'''################## Make Spike Rasters on one plot #######################'''
sns.set_style('ticks')
redmin1_bluemin1 = [u'#aa1016', u'#d52221', u'#f44f39', u'#fc8161', u'#fcaf93', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']
test = ['lcafixed']
for channel in channels:
    sorted_scales = channels[channel].keys()
    sorted_scales.sort()
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title('Spike Times for ' + channel)
    # ax.set_xlabel('Time (ms)', fontsize=16)
    ax.set_ylim(0, 511)
    ax.set_xlim(0, 3100)
    # ax.set_ylabel('Trial Number', fontsize=16)
    # ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)

    for k, scale in zip(range(len(sorted_scales)), sorted_scales):
        x = channels[channel][scale]['spiketimes']
        for i in range(50):
            xi = x.iloc[:, i]
            y = (k * 50) + (i + 1) * np.ones_like(xi)
            #ax.plot(xi, y, marker='_', markersize=1.5, linestyle='', color=redmin1_bluemin1[k])
            ax.scatter(xi, y, marker='_', s=1, color=redmin1_bluemin1[k])

        ax.axhline(y=(k*50), color='gray', linewidth=0.5)
            # plt.legend()
            # plt.savefig(figure_dir + 'raster_2500_3000_' + channel + '.png', dpi=300, format='png')

# get rid of x and y ticks and ticklabels entirely
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])


'''################### Calculate PSTH with full trains #####################'''
# create a gaussian kernel from elephant with sigma of 2 ms
gaussian_kernel = elephant.statistics.make_kernel('GAU', 2*pq.ms, 1*pq.ms)

# create empty psth df to store channels scales and psths
psth_df = pd.DataFrame()

# loop through all channels and all sorted scales
for channel in channels:
    sorted_scales = channels[channel].keys()
    sorted_scales.sort()
    for scale in sorted_scales:
        # select the full spiketime trace, convert to neo object
        spiketimes = channels[channel][scale]['spiketimes']
        neo_spiketimes = []
        for i in range(len(spiketimes.columns)):
            spiketrain = spiketimes.iloc[:, i]
            neo_spiketrain = SpikeTrain(times=spiketrain.values*pq.ms, t_start=0., t_stop=3100.)
            neo_spiketimes.append(neo_spiketrain)

        # convert neo spiketimes to binned spiketrains with x ms bins
        bins = 1.0      # 1 ms bins for 1 kHz final
        binned_spiketrains = elephant.conversion.BinnedSpikeTrain(neo_spiketimes, binsize=bins*pq.ms, t_start=0.*pq.ms, t_stop= 3100.*pq.ms)
        binned_spiketrains = pd.DataFrame(binned_spiketrains.to_array()).T

        # take the mean of all 100 trials, could add and divide by 100 or just mean
        mean_spiketrains = binned_spiketrains.apply(np.mean, axis=1)

        # convolve gaussian with mean and scale by 1000 for inst. FR for 1 kHz
        psth = gaussian_kernel[1] * scipy.signal.fftconvolve(mean_spiketrains, gaussian_kernel[0])
        psth_trial = pd.DataFrame(psth).T

        # create a single trial psth df with channel and scale labels
        labels = pd.DataFrame({'Channel': channel, 'Scale': scale}, index=range(1))
        label_psth = pd.concat([labels, psth_trial], axis=1)

        # create df for all channels and scales
        psth_df = pd.concat([psth_df, label_psth], ignore_index=True)
test = pd.concat([spiketimes, spiketimes], axis=2)
# save to file
psth_df.to_csv(table_dir + 'scaled_psth.csv', float_format='%8.4f', index=False, header=True)

'''###################### Make PSTH Plots from file ########################'''
\ need to work on this once I decide what section Im actually plotting
psth_df = pd.read_csv(table_dir + 'scaled_psth.csv')
psth_df?
scale_list = ['0.50', '1.00', '1.50']
for channel in channels:
    channel_psth = psth_df[psth_df['Channel'] == channel]
    plt.figure()
    for scale in scale_list:
        scale_psth = channel_psth[channel_psth['Scale'] == float(scale)]
        plt.plot(scale_psth.iloc[:, 2702:3002].T)
psth_slice = psth_df.iloc[:, 2702:3002]
plt.figure()
channel_psth = psth_slice[psth_df['Channel'] == 'kA']
channel_psth
plt.plot(channel_psth.T)

channel_psth = psth_df[psth_df['Channel'] == 'kA']
plt.figure()
for scale in scale_list:
    scale_psth = channel_psth[channel_psth['Scale'] == float(scale)]
    plt.plot(scale_psth.iloc[:, 2702:3002].T)

channel_psth[channel_psth['Scale'] == float('0.50')]
scale_psth.iloc[:, 2:].T


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
redblue10 = [u'#aa1016', u'#d52221', u'#f44f39', u'#fc8161', u'#fcaf93', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']
redblue3 = [ u'#6aaed6', 'k', u'#fc8161']    # for just 3 scales

# create fig and axes for subplot of one channel type
gs_kw = dict(height_ratios=[0.5, 0.7, 0.7, 0.7, 4, 1])
fig, axs = plt.subplots(6, 6, figsize=(16.5, 5.5), gridspec_kw=gs_kw, constrained_layout=True)

for col, channel in zip(range(6), ordered_channels):

    for ax in axs[:, col].flatten():
        ax.set_frame_on(False)
        ax.set_xticks([])
        # ax.set_yticks([])

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
        axs[i+1, col].plot(vm_trial_15, label=scale_list[i], color=redblue3[i])
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
            # axs[1].plot(xi, y, marker='_', markersize=1, linestyle='', color=redblue10[k])
            axs[4, col].scatter(xi, y, s=1, marker='.', color=redblue10[k])

        axs[4, col].axhline(y=(k*101), color='gray', linewidth=0.5)
    # axs[2].clear()

    ''' plot psth for 3 scales '''
    channel_psth = psth_df[psth_df['Channel'] == channel]
    for i in range(len(scale_list)):
        scale_psth = channel_psth[channel_psth['Scale'] == float(scale_list[i])]
        psth = scale_psth.iloc[:, 2:].T
        psth.index = np.arange(0, 3110, 1)
        psth_slice = psth.iloc[2705:2905]
        axs[5, col].plot(psth_slice, label=scale_list[i], color=redblue3[i])
    axs[5, col].set_ylim(-8, 168)
    axs[5, col].set_xlim(2705, 2905)

fig
fig.savefig(figure_dir + 'bhalla_wns_vm_raster_psth_3nonoverlay_scaled.png', dpi=300, format='png')

vm_lims = []
psth_lims = []
for col, channel in zip(range(6), channels):
    vm_lims.append(axs[1, col].get_ylim())
    vm_lims.append(axs[2, col].get_ylim())
    vm_lims.append(axs[3, col].get_ylim())

    psth_lims.append(axs[5, col].get_ylim())
vm_lims
psth_lims

'''################### Make Plot of Vm, Raster, and PSTH ###################'''
'''################### for all channels in one plot ########################'''
'''################### for 3 Vm plots overlay ##########################'''

# open any files I need
psth_df = pd.read_csv(table_dir + 'scaled_psth.csv')

# make scale list for reduced plots for ease to see
scale_list = ['1.5', '1', '0.5']

# set plot style and colormaps
sns.set_style('ticks')
redblue10 = [u'#aa1016', u'#d52221', u'#f44f39', u'#fc8161', u'#fcaf93', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']
redblue3 = [ u'#6aaed6', 'k', u'#fc8161']    # for just 3 scales

# create fig and axes for subplot of one channel type
gs_kw = dict(height_ratios=[0.5, 1, 4, 1])
fig, axs = plt.subplots(4, 6, figsize=(18, 4.5), gridspec_kw=gs_kw, constrained_layout=True)

for col, channel in zip(range(6), channels):

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
        axs[1, col].plot(vm_trial_15, label=scale_list[i], color=redblue3[i])
    axs[1, col].set_ylim(-71, 41)
    axs[1, col].set_xlim(27000, 29000)
    # axs[1].clear()

    ''' plot rasters for all scales of 1 channel '''
    sorted_scales = channels[channel].keys()
    sorted_scales.sort()
    # axs[1].set_title('')
    axs[2, col].set_ylim(0, 1011)
    axs[2, col].set_xlim(2700, 2900)
    # ax.set_ylabel('Trial Number', fontsize=16)

    for k, scale in zip(range(len(sorted_scales)), sorted_scales):
        x = channels[channel][scale]['spiketimes']
        for i in range(100):
            xi = x.iloc[:, i]
            y = (k * 101) + (i + 1) * np.ones_like(xi)
            # axs[1].plot(xi, y, marker='_', markersize=1, linestyle='', color=redblue10[k])
            axs[2, col].scatter(xi, y, s=1, marker='.', color=redblue10[k])

        axs[2, col].axhline(y=(k*101), color='gray', linewidth=0.5)
    # axs[2].clear()

    ''' plot psth for 3 scales '''
    channel_psth = psth_df[psth_df['Channel'] == channel]
    for i in range(len(scale_list)):
        scale_psth = channel_psth[channel_psth['Scale'] == float(scale_list[i])]
        psth = scale_psth.iloc[:, 2:].T
        psth.index = np.arange(0, 3110, 1)
        psth_slice = psth.iloc[2705:2905]
        axs[3, col].plot(psth_slice, label=scale_list[i], color=redblue3[i])
    axs[3, col].set_ylim(-8, 168)
    axs[3, col].set_xlim(2705, 2905)

fig.savefig(figure_dir + 'bhalla_wns_vm_raster_psth_yscales.png', dpi=300, format='png')


'''############## Calculate spike frequency after 500 ms of trial ##########'''
\needs annotations
freq_summary = pd.DataFrame()
channel_df = pd.Series()
scale_df = pd.Series()
firingrate_df = pd.Series()

for channel in channels:
    sorted_scales = channels[channel].keys()
    sorted_scales.sort()
    for scale in sorted_scales:
        spiketimes = channels[channel][scale]['binned_spiketrains']
        spikecount = spiketimes.apply(np.sum, axis=0)
        meancount = np.mean(spikecount)/2.5
        channels[channel][scale]['frequency'] = meancount

        firingrate_df = pd.concat([firingrate_df, pd.Series(meancount)], ignore_index=True)
        channel_df = pd.concat([channel_df, pd.Series(channel)], ignore_index=True)
        scale_df = pd.concat([scale_df, pd.Series(scale)], ignore_index=True)

freq_summary['Channel'] = channel_df
freq_summary['Scale'] = scale_df
freq_summary['Firing Rate (Hz)'] = firingrate_df
freq_summary


'''####################### Calculate the fano factor #######################'''
\needs annotations
fano_summary = pd.DataFrame()
channel_df = pd.Series()
scale_df = pd.Series()
fano_df = pd.Series()

for channel in channels:
    sorted_scales = channels[channel].keys()
    sorted_scales.sort()
    for scale in sorted_scales:
        fanofactor = pd.Series()
        for i in range(100):
            spiketrain = channels[channel][scale]['binned_spiketrains'].iloc[:, i]
            if np.sum(spiketrain) == 0:
                continue
            else:
                window_spikes = spiketrain.rolling(100).sum()
                tempfano = window_spikes.var()/window_spikes.mean()
                fanofactor = pd.concat([fanofactor, pd.Series(tempfano)], ignore_index=True)

        fano_df = pd.concat([fano_df, pd.Series(fanofactor.mean())], ignore_index=True)
        channel_df = pd.concat([channel_df, pd.Series(channel)], ignore_index=True)
        scale_df = pd.concat([scale_df, pd.Series(scale)], ignore_index=True)

fano_summary['Channel'] = channel_df
fano_summary['Scale'] = scale_df
fano_summary['Fano Factor'] = fano_df
fano_summary


'''########################### Calculate the CV ISI ########################'''
\needs annotations
cvisi_summary = pd.DataFrame()
channel_df = pd.Series()
scale_df = pd.Series()
cv_df = pd.Series()

for channel in channels:
    sorted_scales = channels[channel].keys()
    sorted_scales.sort()
    for scale in sorted_scales:
        neo_spiketrains = channels[channel][scale]['neo_spiketimes']
        isi_list = [elephant.statistics.isi(spiketrain) for spiketrain in neo_spiketrains]
        cv_list = []
        for i in range(len(isi_list)):
            if isi_list[i].sum() == 0:
                cv_list.append(np.nan)
            else:
                cv_list.append(scipy.stats.variation(isi_list[i]))
        channels[channel][scale]['cvisi'] = np.mean(cv_list)
        cv_mean = pd.Series(np.mean(cv_list))
        cv_df = pd.concat([cv_df, cv_mean], ignore_index=True)
        channel_df = pd.concat([channel_df, pd.Series(channel)], ignore_index=True)
        scale_df = pd.concat([scale_df, pd.Series(scale)], ignore_index=True)

cvisi_summary['Channel'] = channel_df
cvisi_summary['Scale'] = scale_df
cvisi_summary['CV ISI'] = cv_df
cvisi_summary


'''######  Calculate the trial-to-trial pairwise spike time correlation ####'''
# GALAN NON-MEAN-SUBTRACTED RELIABILIY/CORRELATION COEFFICIENT CALCULATION
# box kernel with width 8 ms (2*jitter) and amplitude 1
# NOTE not centered on spike time for convolution

box_kernel = scipy.signal.boxcar(2)
mean_corr_list = []
chan_list = []
scale_list = []
corr_summary = pd.DataFrame()

for channel in channels:
    sorted_scales = channels[channel].keys()
    sorted_scales.sort()
    for scale in sorted_scales:
        # the binary spike time data frame
        spiketrains = channels[channel][scale]['binned_spiketrains']

        # set number of neurons for trial to trial correlations/reliability
        num_spiketrains = len(spiketrains.columns)

        # pre-allocate correlaion matrix
        corr_mat = np.zeros((num_spiketrains, num_spiketrains))

        # set up empty data frame for convolved spiketrain data frame
        box_spiketrains = pd.DataFrame()

        # convolve box kernel with binary spiketrain data frame to make
        # convolved signal data frame
        for i in range(num_spiketrains):
            box_spiketrain = scipy.ndimage.convolve1d(spiketrains.iloc[:, i], box_kernel)
            box_spiketrains = pd.concat([box_spiketrains, pd.DataFrame(box_spiketrain)], axis=1)

        for i in range(num_spiketrains):
            for j in range(i, num_spiketrains):
                # numerator = dot product of (cst_i, cst_j),
                # where cst_i is the ith binary spike train convolved with a box kernel
                # and cst_j is the jth binary spike train convolved with a box kernel
                box_spiketrain_i = box_spiketrains.iloc[:, i]
                box_spiketrain_j = box_spiketrains.iloc[:, j]
                numerator = np.dot(box_spiketrain_i, box_spiketrain_j)

                # denominator = product of norm(box_spiketrain_i, box_spiketrain_j),
                # where norm = the sqrt(sum(x**2)) for box_spiketrain_i and box_spiketrain_j
                box_spiketrain_i_norm = np.linalg.norm(box_spiketrain_i)
                box_spiketrain_j_norm = np.linalg.norm(box_spiketrain_j)
                denominator = box_spiketrain_i_norm * box_spiketrain_j_norm

                # calculate the correlation coefficient and build the matrix
                corr_mat[i, j] = corr_mat[j, i] = numerator / denominator

        corr_mat_df = pd.DataFrame(corr_mat)
        #corr_mat_list.append(corr_mat_df)
        #corr_mat_df_df = pd.concat([corr_mat_df_df, corr_mat_df], axis=2)

        # take out 1's diaganol for mean calculation and maybe plotting
        corr_mat_df_no_ones = corr_mat_df[corr_mat_df < 1]
        mean_corr = np.mean(corr_mat_df_no_ones.mean().values)
        mean_corr_list.append(mean_corr)
        chan_list.append(channel)
        scale_list.append(scale)

mean_corr_df = pd.Series(np.asarray(mean_corr_list))
channel_df = pd.Series(np.asarray(chan_list))
scale_df = pd.Series(np.asarray(scale_list))
corr_summary['Channel'] = channel_df
corr_summary['Scale'] = scale_df
corr_summary['Correlation'] = mean_corr_df
corr_summary


'''############ Make Summary Tables of FR, CVisi, Corr, FF #################'''
# first make channel and scale dfs from FR, FF, corr, and CVisi summaries to test
# skip if already know all good
channel_test_df = pd.DataFrame()
channel_test_df = pd.concat([freq_summary['Channel'], fano_summary['Channel'], corr_summary['Channel'], cvisi_summary['Channel']], axis=1, ignore_index=True)
channel_test_df

scale_test_df = pd.DataFrame()
scale_test_df = pd.concat([freq_summary['Scale'], fano_summary['Scale'], corr_summary['Scale'], cvisi_summary['Scale']], axis=1, ignore_index=True)
scale_test_df

# once channels and scales line up appropriately take FR, FF, corr, and CVisi
spiketrain_statistics = pd.DataFrame()
spiketrain_statistics['Channel'] = freq_summary['Channel']
spiketrain_statistics['Scale'] = freq_summary['Scale']
spiketrain_statistics['Firing Rate (Hz)'] = freq_summary['Firing Rate (Hz)']
spiketrain_statistics['Fano Factor'] = fano_summary['Fano Factor']
spiketrain_statistics['CV ISI'] = cvisi_summary['CV ISI']
spiketrain_statistics['Correlation'] = corr_summary['Correlation']
spiketrain_statistics

# save to file
spiketrain_statistics.to_csv(table_dir + 'scaled_spiketrain_stats.csv', float_format='%8.4f', index=False, header=True)


'''################## Make Linear Regressions of Stats #####################'''
'''
seems like the best way for me to do this right now quickly is to do two different
fits for each one.
1. do   lm = sklearn.linear_model.LinearRegression()
        model = lm.fit(x, y)
        pred = model.predict(x)
        plt.scatter(x, y)
        plt.plot(x, pred)
2. do   slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    then have all the values I need and add to some sort of table along with
    making plots with plot(x, pred) from sklearn.

'''
# open spike statistics to do linear regressions
stats_df = pd.read_csv(table_dir + 'scaled_spiketrain_stats.csv')

channel_stats = stats_df[stats_df['Channel'] == 'kA']

# using statsmodels
x = pd.DataFrame(channel_stats['Scale'])
y = pd.DataFrame(channel_stats['Firing Rate (Hz)'])
x3 = channel_stats['Scale']
# using sklearn instead
# for some reason I needed to take the actual DF of each to make it the right
# shape to not throw an error, not sure why
lm = sklearn.linear_model.LinearRegression()
model = lm.fit(x, y)
pred = model.predict(x)
lm.intercept_
lm.score(x, y)
lm.get_params()
# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y, pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % r2_score(y, pred))

plt.figure()
plt.scatter(x, y)
plt.plot(x, pred)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x.values.flatten(), y.values.flatten())


'''################## Make Plots of FR, FF, CVisi, Corr ####################'''
% matplotlib inline
# open the  spiketrain statistics file if not already open
stats_df = pd.read_csv(table_dir + 'scaled_spiketrain_stats.csv')

# quick fix for making 0 Hz a nan
list0 = stats_df.index[stats_df['Firing Rate (Hz)']  == 0.0].tolist()
for item in list0:
    stats_df['Firing Rate (Hz)'].iloc[item] = np.nan
stats_df

# settings for x and y axis minor  ticks, the  number stands for delta tick
x_minor = MultipleLocator(0.5)
y_minor = MultipleLocator(10)

# set the seaborn background style and generate plots
sns.set_style('ticks')
dark2 = [u'#1b9e77', u'#d95f02', u'#7570b3', u'#e7298a', u'#66a61e', u'#e6ab02', u'#a6761d', u'#666666']

# make a subplot fig with 3 rows 2 cols 8"x8" with autosizing
stat_list = stats_df.columns[2:]
stat_list_less = ['Firing Rate (Hz)', 'CV ISI', 'Correlation']

important_channels = ['kA', 'kca3', 'lcafixed']
na_k_channels = ['kslowtab', 'kfasttab',  'nafast']

'''all channels'''
fig, axs = plt.subplots(1, 4, figsize=(12, 2.25), constrained_layout=True)

for ax, stat in zip(axs.flat, stat_list):
    stat_set = stats_df.loc[:, ['Channel', 'Scale', stat]]
    for channel, i in zip(ordered_channels, range(6)):
        channel_stats = stat_set[stat_set['Channel'] == channel]
        scale = channel_stats['Scale']
        ax.plot(scale, channel_stats[stat], label=channel, marker='o', color=dark2[i])
    ax.set_title(stat)

fig

axs[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

'''important channels'''
fig, axs = plt.subplots(1, 3, figsize=(8, 2), constrained_layout=True)

for ax, stat in zip(axs.flat, stat_list_less):
    stat_set = stats_df.loc[:, ['Channel', 'Scale', stat]]
    for channel, i in zip(important_channels, range(3)):
        channel_stats = stat_set[stat_set['Channel'] == channel]
        scale = channel_stats['Scale']
        ax.plot(scale, channel_stats[stat], label=channel, marker='o', color=dark2[i])
        ax.xaxis.set_minor_locator(x_minor)
        ax.set_xlim(-0.2, 3.2)
    ax.set_title(stat)
fig
axs[2].legend(loc='upper right')

'''na_k channels'''
fig, axs = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

for ax, stat in zip(axs.flat, stat_list):
    stat_set = stats_df.loc[:, ['Channel', 'Scale', stat]]
    for channel in na_k_channels:
        channel_stats = stat_set[stat_set['Channel'] == channel]
        scale = channel_stats['Scale']
        ax.plot(scale, channel_stats[stat], label=channel, marker='o')
    ax.set_title(stat)
axs[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

axs[0, 0].set_ylabel('Current (pA)')
axs[0, 1].set_ylabel('Scaling Factor (fold-change)')
axs[1, 0].set_ylabel('Current (pA)')
axs[1, 1].set_ylabel('Scaling Factor (fold-change)'y)


# save fig to file
fig.savefig(figure_dir + 'STAs_notopright.png', dpi=300, format='png')


'''############################ STA Creation ###############################'''
# convert wns_file to neo.AnalogSignal
wns_list = []
for i in range(len(wns_file.columns)):
    wns = wns_file.iloc[:, i]
    wns_mean = np.mean(wns.iloc[500:30501])
    wns_mean_sub = wns - wns_mean
    wns_trial = AnalogSignal(signal=wns_mean_sub.values*pq.nA, sampling_rate=10*pq.kHz)
    wns_list.append(wns_trial)

# create empty dataframes for stas and run through loop
# loop caluclates mean and std of STA for each channel and scale, then saves to
# a df, which I have to save to file afterwards

sta_mean = pd.DataFrame()
sta_std = pd.DataFrame()

for channel in channels:
    sorted_scales = channels[channel].keys()
    sorted_scales.sort()
    for scale in sorted_scales:
        sta_df = pd.DataFrame()
        spikes = channels[channel][scale]['neo_spiketimes']
        for i in range(len(spikes)):
            wns_trial = wns_list[i]
            spikes_trial = spikes[i]
            sta_trial = elephant.sta.spike_triggered_average(wns_trial, spikes_trial, (-35*pq.ms, 5.1*pq.ms))
            sta_trial_df = pd.DataFrame(sta_trial.as_array().flatten())
            sta_df = pd.concat([sta_df, sta_trial_df], axis=1)
        sta_df_mean = sta_df.apply(np.mean, axis=1)
        # sta_mean_list.append(sta_df_mean)
        sta_df_std = sta_df.apply(np.std, axis=1)
        # sta_std_list.append(sta_df_std)
        # chan_list.append(channel)
        # scale_list.append(scale)
        labels = pd.DataFrame({'Channel': channel, 'Scale': scale}, index=range(1))
        mean_tag = pd.DataFrame({'Measure': 'mean (nA)'}, index=range(1))
        std_tag = pd.DataFrame({'Measure': 'std. (nA)'}, index=range(1))
        mean = pd.DataFrame(sta_df_mean).T
        std = pd.DataFrame(sta_df_std).T
        label_sta_mean = pd.concat([labels, mean_tag, mean], axis=1)
        label_sta_std = pd.concat([labels, std_tag, std], axis=1)
        sta_mean = pd.concat([sta_mean, label_sta_mean])
        sta_std = pd.concat([sta_std, label_sta_std])
sta_mean.to_csv(table_dir + 'sta_mean.csv', float_format='%10.6f', index=False, header=True)
sta_std.to_csv(table_dir + 'sta_std.csv', float_format='%10.6f', index=False, header=True)


'''############################ STA Features ###############################'''
# open STA files if not open already
sta_mean = pd.read_csv(table_dir + 'sta_mean.csv')
sta_std = pd.read_csv(table_dir + 'sta_std.csv')

# create a data frame for just the sta time series and its corresponding index
stas = sta_mean.iloc[:, 3:-50] * 1000
stas.columns = np.arange(0, 351, 1)
timeindex = np.linspace(-35, 0.0, num=351)

# create data frame with channel and scale labels and add peaks and times
# using df methods
sta_features = sta_mean.iloc[:, :2]
sta_features['Positive Peak (pA)'] = stas.max(axis=1)
sta_features['Pos. Peak Index'] = stas.idxmax(axis=1)
sta_features['Pos. Peak Time (ms)'] = 0.0
sta_features['Negative Peak (pA)'] = stas.min(axis=1)
sta_features['Neg. Peak Index'] = stas.idxmin(axis=1)
sta_features['Pos. Peak Time (ms)'] = 0.0

# pull out slope, slope time, and integration time with a loop for each sta
# use a windowed sta based on neg and pos peak times to calculate the max slope
# and zero crossing from neg peak to pos peak for integration time

# create empty series for storing values in loop for each sta
pos_time_df = pd.Series()           # df for storing actual time of pos peak
neg_time_df = pd.Series()           # df for storing actual time of neg peak
slope_df = pd.Series()              # df to store max slope
slope_index_df = pd.Series()        # df to store index of max slope
slope_time_df = pd.Series()         # df to store actual time of max slope
int_index_df = pd.Series()          # df to store index of integration start
int_index_time_df = pd.Series()     # df to store actual time of int start
int_value_df = pd.Series()          # store the value as a check of lowest amp
int_time_df = pd.Series()           # the value = x to 0 in time (ms)

for i in range(len(stas.index)):
    # pull out a single sta out of df
    sta_trial = stas.loc[i]

    # define the start and end of the window based on neg and pos peak times
    window_start = sta_features['Neg. Peak Index'].loc[i]
    window_end = sta_features['Pos. Peak Index'].loc[i] + 1

    # use index of peak times to convert to actual times for df
    if np.isnan(window_start) == True:
        pos_time_df = pd.concat([pos_time_df, pd.Series(np.nan)], ignore_index=True)
        neg_time_df = pd.concat([neg_time_df, pd.Series(np.nan)], ignore_index=True)

    else:
        pos_time_df = pd.concat([pos_time_df, pd.Series(timeindex[int(window_end)])], ignore_index=True)
        neg_time_df = pd.concat([neg_time_df, pd.Series(timeindex[int(window_start)])], ignore_index=True)

    # the following if else logic will exclude nan values and bad slopes
    if window_start < window_end:
        # set up window as row neg peak time to pos peak time
        window = sta_trial.loc[window_start:window_end]
        slope = pd.Series(np.append([0], np.diff(window)))   # 0 pad beginning of slope, and make a series so I can index
        slope.index = window.index
        max_slope = slope.max() * 10            # multiply by 10 so value is nA/ms
        slope_index = slope.idxmax()            # find index where slope is max in terms of real index time
        slope_time = timeindex[slope_index]     # pull out the actual time

        # save the slope and the slope time to the series
        slope_df = pd.concat([slope_df, pd.Series(max_slope)], ignore_index=True)
        slope_index_df = pd.concat([slope_index_df, pd.Series(float(slope_index))], ignore_index=True)
        slope_time_df = pd.concat([slope_time_df, pd.Series(slope_time)], ignore_index=True)

        # slice window to choose only positive going values, then find minimum index
        window_positive = window[window >= 0]
        crossover_index = pd.Series(window_positive.idxmin())
        crossover_time = pd.Series(timeindex[crossover_index.values])

        # calculate index size for crossover index to 0 and convert to time
        int_delta =float(350 - crossover_index.values)
        int_time = pd.Series(int_delta / 10)       # convert index to ms time

        # save the actual time, absolute value of time, and value at that time as checks
        int_index_df = pd.concat([int_index_df, crossover_index], ignore_index=True)
        int_index_time_df = pd.concat([int_index_time_df, crossover_time], ignore_index=True)
        int_time_df = pd.concat([int_time_df, int_time], ignore_index=True)
        int_value_df = pd.concat([int_value_df, pd.Series(window_positive.min())], ignore_index=True)

    # this else statement is supposed to exclude any nan or anomalous stas
    else:
        # save all remaining features as nan
        slope_df = pd.concat([slope_df, pd.Series(np.nan)], ignore_index=True)
        slope_index_df = pd.concat([slope_index_df, pd.Series(np.nan)], ignore_index=True)
        slope_time_df = pd.concat([slope_time_df, pd.Series(np.nan)], ignore_index=True)
        int_index_df = pd.concat([int_index_df, pd.Series(np.nan)], ignore_index=True)
        int_index_time_df = pd.concat([int_index_time_df, pd.Series(np.nan)], ignore_index=True)
        int_time_df = pd.concat([int_time_df, pd.Series(np.nan)], ignore_index=True)
        int_value_df = pd.concat([int_value_df, pd.Series(np.nan)], ignore_index=True)

# add all features to the dataframe
sta_features['Pos. Peak Time (ms)'] = pos_time_df
sta_features['Neg. Peak Time (ms)'] = neg_time_df
sta_features['Max Slope (pA/ms)'] = slope_df
sta_features['Slope Index'] = slope_index_df
sta_features['Slope Time (ms)'] = slope_time_df
sta_features['Crosstime Index'] = int_index_df
sta_features['Crosstime Time (ms)'] = int_index_time_df
sta_features['Integration Time (ms)'] = int_time_df
sta_features['Cross value (pA)'] = int_value_df

# save the feature data frame to file
sta_features.to_csv(table_dir + 'sta_features_index.csv', float_format='%8.3f', index=False, header=True)
sta_features

'''########################## STA Trace Plots ##############################'''
# open STA files if not open already
sta_mean = pd.read_csv(table_dir + 'sta_mean.csv')
sta_std = pd.read_csv(table_dir + 'sta_std.csv')

# set time and number of scales and color palette
time = np.linspace(-35, 0.0, num=351)
n_scales = 10
redmin1_bluemin1 = [u'#aa1016', u'#d52221', u'#f44f39', u'#fc8161', u'#fcaf93', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']

# set the seaborn background style and generate plots
sns.set_style('ticks')
for channel in channels:
    plt.figure()
    channel_set = sta_mean[sta_mean['Channel'] == channel]
    for i in range(n_scales):
        scale_slice = channel_set.iloc[i]
        scale = scale_slice['Scale']
        sta = scale_slice.iloc[3:-50] * 1000
        plt.plot(time, sta, label=scale, color=redblue10[i], linewidth=2)
    plt.legend()
    plt.title('STA for ' + channel)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    #plt.savefig(figure_dir + 'sta_test2_' + channel + '.png', dpi=300, format='png')
% matplotlib inline
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
redmin1_bluemin1 = [u'#aa1016', u'#d52221', u'#f44f39', u'#fc8161', u'#fcaf93', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']
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
        ax.plot(time, sta, label=scale, color=redmin1_bluemin1[i], linewidth=2)
        # ax.set_title(channel, pad=-14)          # the pad moves the title down
        ax.set_ylim(-25, 55)                    # make all y axes equal
        ax.xaxis.set_minor_locator(x_minor)     # add x axis minor ticks
        ax.yaxis.set_minor_locator(y_minor)     # add y axis minor ticks
        ax.spines["right"].set_visible(False)   # if I want cleaner look w/out
        ax.spines["top"].set_visible(False)     # top and right axes lines
        ax.set_ylabel('Current (pA)')
        ax.set_xlabel('Time (ms)')
fig
axs[0].set_ylabel('Current (pA)')
# add legend
axs[5].legend(loc='center left', bbox_to_anchor=(1.1, 0.45))

# save fig to file
fig.savefig(figure_dir + 'STAs_columns_ordered_ylabel_shorter.png', dpi=300, format='png')
fig.savefig(figure_dir + 'STAs_columns_ordered_labeled.png', dpi=300, format='png')


'''####################### Plots of STAs with features #####################'''
# open STA files if not open already
sta_mean = pd.read_csv(table_dir + 'sta_mean.csv')
sta_std = pd.read_csv(table_dir + 'sta_std.csv')
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')

n_scales = 10
time = np.linspace(-35, 0.0, num=351)
stas = sta_mean.iloc[:, 3:-50] * 1000
stas.columns = np.arange(0, 351, 1)


# settings for x and y axis minor  ticks, the  number stands for delta tick
x_minor = MultipleLocator(5)
y_minor = MultipleLocator(10)

sns.set_style('ticks')

fig, axs = plt.subplots(1, 6, figsize=(18, 3), constrained_layout=True)

for ax, channel in zip(axs.flat, channels):
    # pull out the STAs for each channel type individually
    channel_set = sta_mean[sta_mean['Channel'] == channel]
    feat_set = sta_features[sta_features['Channel'] == channel]

    # loop through all scales one by one
    for i in range(n_scales):
        scale_slice = channel_set.iloc[i]       # pull out scale row
        scale = scale_slice['Scale']            # define scale
        sta = scale_slice.iloc[3:-50] * 1000    # pull out STA convert to pA
        sta.index = np.arange(0, 351, 1)
        feat = feat_set.iloc[i]
        feat = feat.fillna(0)
        pos_peak = feat['Positive Peak (pA)']
        pos_index = feat['Pos. Peak Time (ms)']
        neg_peak = feat['Negative Peak (pA)']
        neg_index = feat['Neg. Peak Time (ms)']
        slope = sta.loc[feat['Slope Index']]
        slope_index = feat['Slope Time (ms)']
        cross_value = feat['Cross value (pA)']
        cross_index = feat['Crosstime Time (ms)']
        ax.plot(time, sta, label=scale, color=redmin1_bluemin1[i], linewidth=2)
        ax.plot(pos_index, pos_peak, marker='X', linestyle='', color=redmin1_bluemin1[i])
        ax.plot(neg_index, neg_peak, marker='o', linestyle='', color=redmin1_bluemin1[i])
        ax.plot(cross_index, cross_value, marker='D', linestyle='', color=redmin1_bluemin1[i])
        ax.set_title(channel, pad=-14)          # the pad moves the title down
        ax.set_ylim(-25, 55)                    # make all y axes equal
        ax.xaxis.set_minor_locator(x_minor)     # add x axis minor ticks
        ax.yaxis.set_minor_locator(y_minor)     # add y axis minor ticks
        ax.spines["right"].set_visible(False)   # if I want cleaner look w/out
        ax.spines["top"].set_visible(False)     # top and right axes lines

# add label to y-axes
axs[0].set_ylabel('Current (pA)')

# add label to x-axes
for n in range(2):
    axs[2, n].set_xlabel('Time (ms)')

# add legendf\
axs[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

'''############### STA Features vs. Channel Conductance ####################'''
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')
n_scales = 10


feature_list = ['Positive Peak (pA)', 'Pos. Peak Time (ms)', 'Negative Peak (pA)',\
                'Neg. Peak Time (ms)', 'Max Slope (pA/ms)', 'Integration Time (ms)']

important_channels = ['kA', 'kca3', 'lcafixed']
na_k_channels = ['nafast', 'kslowtab', 'kfasttab']

# put into a single figure
x_minor = MultipleLocator(5)
y_minor = MultipleLocator(10)

fig, axs = plt.subplots(3, 2, figsize=(8, 8), constrained_layout=True)

for ax, feature in zip(axs.flat, feature_list):
    feature_set = sta_features.loc[:, ['Channel', 'Scale', feature]]
    for channel in ordered_channels:
        channel_feature = feature_set[feature_set['Channel'] == channel]
        scale = channel_feature['Scale']
        ax.plot(scale, channel_feature[feature], label=channel, marker='o')
    ax.set_title(feature)

    # ax.set_ylim(-25, 55)
    # ax.xaxis.set_minor_locator(x_minor)
    # ax.yaxis.set_minor_locator(y_minor)
    # ax.spines["right"].set_visible(False)   # if I want cleaner look w/out
    # ax.spines["top"].set_visible(False)     # top and right axes lines
# set all y labels
fig
axs[0, 0].set_ylabel('Current (pA)')
axs[0, 1].set_ylabel('Time (ms)')
axs[1, 0].set_ylabel('Current (pA)')
axs[1, 1].set_ylabel('Time (ms)')
axs[2, 0].set_ylabel('Slope (pA/ms)')
axs[2, 1].set_ylabel('Time (ms)')

# set x labels
for n in range(2):
    axs[2, n].set_xlabel('Scaling Factor (fold-change)')

axs[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

''' just the 3 modulatory channels '''
fig, axs = plt.subplots(3, 2, figsize=(8, 8), constrained_layout=True)

for ax, feature in zip(axs.flat, feature_list):
    feature_set = sta_features.loc[:, ['Channel', 'Scale', feature]]
    for channel in important_channels:
        channel_feature = feature_set[feature_set['Channel'] == channel]
        scale = channel_feature['Scale']
        ax.plot(scale, channel_feature[feature], label=channel, marker='o')
    ax.set_title(feature)

# set all y labels
axs[0, 0].set_ylabel('Current (pA)')
axs[0, 1].set_ylabel('Time (ms)')
axs[1, 0].set_ylabel('Current (pA)')
axs[1, 1].set_ylabel('Time (ms)')
axs[2, 0].set_ylabel('Slope (pA/ms)')
axs[2, 1].set_ylabel('Time (ms)')

# set x labels
for n in range(2):
    axs[2, n].set_xlabel('Scaling Factor (fold-change)')

axs[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

''' just the Na and K dr channels '''
fig, axs = plt.subplots(3, 2, figsize=(8, 8), constrained_layout=True)

for ax, feature in zip(axs.flat, feature_list):
    feature_set = sta_features.loc[:, ['Channel', 'Scale', feature]]
    for channel in na_k_channels:
        channel_feature = feature_set[feature_set['Channel'] == channel]
        scale = channel_feature['Scale']
        ax.plot(scale, channel_feature[feature], label=channel, marker='o')
    ax.set_title(feature)

# set all y labels
axs[0, 0].set_ylabel('Current (pA)')
axs[0, 1].set_ylabel('Time (ms)')
axs[1, 0].set_ylabel('Current (pA)')
axs[1, 1].set_ylabel('Time (ms)')
axs[2, 0].set_ylabel('Slope (pA/ms)')
axs[2, 1].set_ylabel('Time (ms)')

# set x labels
for n in range(2):
    axs[2, n].set_xlabel('Scaling Factor (fold-change)')

axs[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

#fig.savefig(figure_dir + '', dpi=300, format='png')


'''################ STA Features and Spike Statistics Plots ################'''
# open data from files
stats_df = pd.read_csv(table_dir + 'scaled_spiketrain_stats.csv')
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')

# merge into a  single df to allow easy calling later on
stats_features = pd.merge(stats_df, sta_features, on=['Channel', 'Scale'])
stats_features.columns

# set up lists of subsets of channels for testing
important_channels = ['kA', 'kca3', 'lcafixed']
na_k_channels = ['nafast', 'kslowtab', 'kfasttab']
test = ['kA']

''' plot 2d  images with labels at 3 an 0.01'''

plt.figure()
for channel in important_channels:
    channel_set = stats_features[stats_features['Channel'] == channel]
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    scale = channel_set['Scale']
    plt.plot(slope, int_time, label=channel, marker='o')
    for label, x, y in zip(scale, slope, int_time):
        if label == 3.0:
            plt.annotate('(%s)' % label, xy=(x, y), textcoords='data')
        elif label == 0.01:
            plt.annotate('(%s)' % label, xy=(x, y), textcoords='data')
plt.legend()

scale
for label, x, y in zip (scale, slope, int_time):
    if label == 3.0:
        print label, x, y
    elif label == 0.01:
        print label, x, y
zip(scale, slope, int_time)

''' plot 2d  images with different markers at 3, 1, 0.01'''
'''
USe a ax.get_line[-1].get_markerfacecolor() to set the color of all markers for
a given line. NOt fully tested yet but should work.
'''
fig = plt.figure()
ax = fig.add_subplot(111)
for channel in important_channels:
    channel_set = stats_features[stats_features['Channel'] == channel]
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    scale = channel_set['Scale']
    ax.plot(slope, int_time, label=channel)
    # for label, x, y in zip(scale, slope, int_time):
    #    if label == 3.0:
    #        ax.scatter(x, y, marker='^')
    #    elif label == 1:
    #        ax.scatter(x, y, marker='o')
    #    elif label == 0.01:
    #        ax.scatter(x, y, marker='v')
plt.legend()
ax.get_facecolor()
ax.get_lines()[2].get_markerfacecolor()
sns.palplot([(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)])

for feature in stats_features_list:
    plt.figure()
    for channel in important_channels:
        channel_set = stats_features[stats_features['Channel'] == channel]
        slope = channel_set['Max Slope (pA/ms)']
        int_time = channel_set['Integration Time (ms)']
        scale = channel_set[feature]
        ax.plot(slope, int_time, feat, label=channel, marker='o')
    ax.legend()
    ax.set_title(feature)
    ax.set_xlabel('max slope (pA/ms)')
    ax.set_ylabel('integration time (ms)')
    ax.set_zlabel(feature)

''' plot  3d images '''

from mpl_toolkits.mplot3d import Axes3D

stats_features_list = ['Correlation', 'Firing Rate (Hz)', 'Fano Factor', 'CV ISI']
for feature in stats_features_list:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for channel in important_channels:
        channel_set = stats_features[stats_features['Channel'] == channel]
        slope = channel_set['Max Slope (pA/ms)']
        int_time = channel_set['Integration Time (ms)']
        feat = channel_set[feature]
        ax.plot(slope, int_time, feat, label=channel, marker='o')
    ax.legend()
    ax.set_title(feature)
    ax.set_xlabel('max slope (pA/ms)')
    ax.set_ylabel('integration time (ms)')
    ax.set_zlabel(feature)

'''########################### Color Palettes ##############################'''
# below are tests for colors and plots, with ones that I sort of like saved
# these two lines are to generate hex codes and plot them as colors

sns.color_palette('Blues', 5)
sns.palplot(sns.color_palette('BrBG', 11))
sns.color_palette('BrBG', 11).as_hex()
BrBG11_k = [u'#824b09', u'#ad7021', u'#cfa256', u'#e7cf94', u'#f6eacb', 'k', u'#98d7cd', u'#58b0a7', u'#23867e', u'#015f56']
PuOr11_k = [u'#4d207d', u'#70589f', u'#9990bf', u'#bfbbda', u'#dddfed', 'k',  u'#fdc57f', u'#ee9b39', u'#d0730f', u'#aa5306']
magma10_g = [u'#fed395', u'#fea973', u'#fa7d5e', u'#e95462', u'#c83e73', 'gray', u'#7e2482', u'#59157e', u'#331067', u'#120d31']
inferno10_g = [u'#fed395', u'#fea973', u'#fa7d5e', u'#e95462', u'#c83e73', 'gray', u'#7e2482', u'#59157e', u'#331067', u'#120d31']
redblue = [u'#b11218', u'#e32f27', u'#fb6b4b', u'#fca082', u'#fdd4c2', 'k', u'#d0e1f2', u'#94c4df', u'#4a98c9', u'#1764ab']
darkredblue = [u'#6d322f', u'#a8312b', u'#e33027', u'#eb5545', u'#f47b64', 'k', u'#d0e1f2', u'#94c4df', u'#4a98c9', u'#1764ab']
darkpurpleblue = [u'#662e51', u'#9a2870', u'#ce248f', u'#dc519d', u'#ec7fab', 'k', u'#d0e1f2', u'#94c4df', u'#4a98c9', u'#1764ab']
purpleblue = [u'#8b0179', u'#cd238f', u'#f769a1', u'#fbacb9', u'#fdd7d4', 'k', u'#d0e1f2', u'#94c4df', u'#4a98c9', u'#1764ab']
darkreddarkblue = [u'#6d322f', u'#a8312b', u'#e33027', u'#eb5545', u'#f47b64', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']
redmin1_bluemin1 = [u'#aa1016', u'#d52221', u'#f44f39', u'#fc8161', u'#fcaf93', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']
seismic11_k = [(0.6647058823529411, 0.0, 0.0), (0.8294117647058823, 0.0, 0.0), (1.0, 0.00392156862745098, 0.00392156862745098), (1.0, 0.3333333333333333, 0.3333333333333333), (1.0, 0.6627450980392154, 0.6627450980392154), 'k', (0.6627450980392157, 0.6627450980392157, 1.0), (0.33333333333333337, 0.33333333333333337, 1.0), (0.0, 0.0, 0.991764705882353), (0.0, 0.0, 0.7611764705882353), (0.0, 0.0, 0.5305882352941177)]
PRGn11_k = [(0.4253748558246828, 0.1356401384083045, 0.4749711649365628),(0.5515570934256055, 0.3423298731257209, 0.6152249134948097),(0.6819684736639753, 0.5451749327181853, 0.7425605536332179),(0.8091503267973855, 0.7084967320261437, 0.8444444444444444),(0.9157247212610534, 0.8529027297193387, 0.9190311418685121),'k',(0.8694348327566321, 0.9454825067281815, 0.8495963091118801),(0.7176470588235296, 0.8862745098039216, 0.6941176470588237),(0.4931949250288352, 0.7653979238754326, 0.4966551326412919),(0.2657439446366782, 0.6076124567474048, 0.3222606689734717),(0.08719723183391004, 0.43460207612456747, 0.19630911188004616)]
RdBu11_k =  [(0.6461361014994232, 0.07750865051903114, 0.16032295271049596),(0.7893886966551327, 0.2768166089965398, 0.2549019607843137),(0.8991926182237601, 0.5144175317185697, 0.4079200307574009),(0.9686274509803922, 0.7176470588235293, 0.5999999999999999),(0.9884659746251442, 0.8760476739715493, 0.809919261822376),'k',(0.654901960784314, 0.8143790849673205, 0.8941176470588236),(0.4085351787773935, 0.6687427912341408, 0.8145328719723184),(0.21568627450980393, 0.5141868512110727, 0.7328719723183391),(0.11003460207612457, 0.36262975778546713, 0.6226066897347174)]
dark2 = [u'#1b9e77', u'#d95f02', u'#7570b3', u'#e7298a', u'#66a61e', u'#e6ab02', u'#a6761d', u'#666666']
redblue10 = [u'#aa1016', u'#d52221', u'#f44f39', u'#fc8161', u'#fcaf93', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']

sns.palplot(redblue10)
% matplotlib
colorcode

fig = colorcode.get_figure()

sns.palplot(redmin1_bluemin1)
