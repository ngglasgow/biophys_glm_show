# -*- coding: utf-8 -*-
'''
10 Oct 2018
Clean up analysis routines from before for analyzing colored noise statistics
that will be appropriate for choosing a set of noise for actual scaling sims.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import elephant
import quantities as pq
import scipy
import os
from neo.core import SpikeTrain
from neuron import h

# define directory structure specific to this file and make a data file list
home_dir = '/Users/nate/ngglasgow@gmail.com/Data_Urban/NEURON/Bhalla_par_scaling'
data_dir = 'test_wns_500_output'
figure_dir = 'analysis/figures'
table_dir = 'analysis/tables'

os.chdir(home_dir)
data_files = os.listdir(data_dir)
data_files.sort()
print data_files

# read data into dataframes and lists
i_list = []
vm_list = []
corr_list = []
sd_list = []
binary_spikes = []
spike_times = []
neo_spikes = []
file_list = []

time = pd.DataFrame(np.arange(0, 3100., 0.1))

for file in data_files:
    if 'i' in file:
        i = pd.read_csv(data_dir + '/' + file, sep='\s+', header=None)
        i_list.append(i)
        corr = file.split('_')[1][:2]
        corr_list.append(corr)
        sd = file.split('_')[1][2:]
        sd_list.append(sd)
        file_list.append(file[:8])

    elif 'v' in file:
        vm = pd.read_csv(data_dir + '/' + file, sep='\s+', header=None)
        vm_list.append(vm)
        binary = pd.DataFrame()
        for i in range(50):
            h_vm = h.Vector(vm.iloc[:, i].values)
            h_st = h.Vector()
            h_st.spikebin(h_vm, 0)
            h_st_df = pd.DataFrame(h_st.to_python())
            binary = pd.concat([binary, h_st_df], axis=1)
        binary_spikes.append(binary)
        binary_time = pd.concat([time, binary], axis=1)
        tst_list = []
        tneost_list = []
        for i in range(1, 51):
            tst = binary_time.iloc[5500:, [0, i]]
            tst = tst.iloc[:, 0][tst.iloc[:, 1] > 0]
            tst_list.append(tst)
            tneost = SpikeTrain(times=tst.values*pq.ms, t_start=550., t_stop=3100)
            tneost_list.append(tneost)
        spike_times.append(tst_list)
        neo_spikes.append(tneost_list)

# determine spike frequency and put into df, cut off first 500 ms

mean_list = []
for i in range(len(spike_times)):
    freq_list = []
    for trial in spike_times[i]:
        freq_list.append(len(trial)/2.5)
        mean = np.mean(np.asarray(freq_list))
    mean_list.append(mean)

mean_freq = pd.DataFrame(np.asarray(mean_list))
correlation = pd.DataFrame(np.asarray(corr_list))
SD = pd.DataFrame(np.asarray(sd_list))
file_bases = pd.DataFrame(np.asarray(file_list))
frequency = pd.concat([file_bases, correlation, SD, mean_freq], axis=1, ignore_index=True)

# bin into 1 ms bins
binnedst_list = []
bst_df_list = []
bst_sum_list = []
for item in neo_spikes:
    bst_list = elephant.conversion.BinnedSpikeTrain(item,
                                                    binsize=1.0*pq.ms,
                                                    t_start=550.0*pq.ms,
                                                    t_stop=3100*pq.ms)


    bst_arr = bst_list.to_array()           #export binned spike times to an array
    bst_df = pd.DataFrame(bst_arr).T        #turn into a df and transpose (.T)
    bst_sum = bst_df.apply(np.sum,axis=1)   #sum by row across columns
    bst_df_list.append(bst_df)
    bstc_df = bst_df.iloc[550:3050]
    binnedst_list.append(bst_list)
    bst_sum_list.append(bst_sum)

bst_df_list     # has len 16 and are all df with 50 columns
bst_df_list[0]


# correlation calculation

spktrain_corr = None
mean_corr_df = pd.DataFrame()
corr_mat_df_df = pd.DataFrame()
tmpbox = scipy.signal.boxcar(2)
corr_mat_list = []
mean_corr_list = []
for item in bst_df_list:
        # the binary spike time data frame
        tmpbstdf = item

        # set number of neurons for trial to trial correlations/reliability
        num_spiketrains = len(tmpbstdf.columns)

        # pre-allocate correlaion matrix
        corr_mat = np.zeros((num_spiketrains, num_spiketrains))

        # set up empty data frame for convolved spike train data frame
        tmpcstdf = pd.DataFrame()

        # convolve box kernel with binary spike train data frame to make convolved
        # signal data frame
        for i in range(num_spiketrains):
            tmpkern = scipy.ndimage.convolve1d(tmpbstdf.iloc[:, i], tmpbox)
            tmpkerndf = pd.DataFrame(tmpkern)
            tmpcstdf = pd.concat([tmpcstdf, tmpkerndf], axis=1)

        for i in range(num_spiketrains):
            for j in range(i, num_spiketrains):
                # numerator = dot product of (cst_i, cst_j),
                # where cst_i is the ith binary spike train convolved with a box kernel
                # and cst_j is the jth binary spike train convolved with a box kernel
                cst_i = tmpcstdf.iloc[:, i]
                cst_j = tmpcstdf.iloc[:, j]
                numerator = np.dot(cst_i, cst_j)

                # denominator = product of norm(cst_i, cst_j),
                # where norm = the sqrt(sum(x**2)) for cst_i and cst_j
                cst_i_norm = np.linalg.norm(cst_i)
                cst_j_norm = np.linalg.norm(cst_j)
                denominator = cst_i_norm * cst_j_norm

                # calculate the correlation coefficient and build the matrix
                corr_mat[i, j] = corr_mat[j, i] = numerator / denominator

        corr_mat_df = pd.DataFrame(corr_mat)
        #corr_mat_list.append(corr_mat_df)
        #corr_mat_df_df = pd.concat([corr_mat_df_df, corr_mat_df], axis=2)

        # take out 1's diaganol for mean calculation and maybe plotting
        corr_mat_df_no_ones = corr_mat_df[corr_mat_df < 1]
        mean_corr = np.mean(corr_mat_df_no_ones.mean().values)
        mean_corr_list.append(mean_corr)

spiketime_corr = pd.DataFrame(np.asarray(mean_corr_list))

corr_freq = pd.concat([frequency, spiketime_corr], axis=1, ignore_index=True)
corr_freq.columns = ['Filename', 'WNS Corr.', 'SD (nA)', 'Spike Freq.', 'Correlation']
corr_freq

# save to file
corr_freq.to_csv(home_dir + '/' + table_dir + '/spike_freq_corr.csv',
                 sep=',', float_format='%10.3f', index=False, header=True)

# make a plot of spike time correlations as function of SD and wns correlation
plt.figure()
corrvalue_list = ['80', '90', '95', '98']
for corr in corrvalue_list:
    noisecorr = corr_freq[corr_freq['WNS Corr.'] == corr]
    x = noisecorr.iloc[:, 2]
    y = noisecorr.iloc[:, 4]
    plt.plot(x, y, label=corr + ' corr. WNS', marker='.')
    plt.xlabel('WNS %SD')
    plt.ylabel('Cross-trial Spike Train Correlation Coefficient')
    plt.legend()
    plt.title('Reliability Correlation with 1 ms bins')
    plt.savefig(home_dir + '/' + figure_dir + '/spike_correlation.png',
                dpi=300, format='png')


# make raster plots to inspect
raster_list = []
for k in range(16):
    plt.figure()
    for i, spiketrain in enumerate(neo_spikes[k]):
        t = spiketrain.rescale(pq.ms)
        raster_list.append(t)
        plt.plot(t,(i+1)*np.ones_like(t),'k.',markersize=2)
        plt.axis('tight')
        plt.xlim(0,3100)
        plt.xlabel('Time (ms)',fontsize=16)
        plt.ylim(0,51)
        plt.ylabel('Trial Number', fontsize=16)
        plt.gca().tick_params(axis='both',which='major',labelsize=14)
        plt.title('WNS Corr. ' + corr_list[k] + ' with %SD of ' + sd_list[k])
        plt.show()

test = vm_list[9].iloc[:, :5]
test2 = vm_list[10].iloc[:, :5]
plt.figure()
plt.plot(test)
plt.figure()
plt.plot(test2)
