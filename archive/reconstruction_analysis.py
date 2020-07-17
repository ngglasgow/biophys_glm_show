# -*- coding: utf-8 -*-
import os
import numpy as np
from neo.core import SpikeTrain, AnalogSignal
import quantities as pq
import platform
import pandas as pd
import elephant
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sklearn
from scipy.signal.windows import dpss


sns.set()
#% matplotlib inline
%matplotlib
'''################### Set direcotories and open files #####################'''
# set system type as macOS or linux; set project directory; set downstream dirs
# set all these directory variables first

home_dir = os.path.expanduser("~")
project_path = 'ngglasgow@gmail.com/Data_Urban/NEURON/biophys_glm'
project_path = 'projects/biophys_glm_show'
project_dir = os.path.join(home_dir, project_path)
recon_dir = os.path.join(project_dir, 'data', 'reconstructed_stimulus')
alon_recon = os.path.join(recon_dir, 'alon')

bhalla_recon = os.path.join(recon_dir, 'bhalla')
figure_dir = os.path.join(project_dir, 'analysis/figures/')
table_dir = os.path.join(project_dir, 'analysis/tables/')
# grab file names from my output directory and sort by name
alon_file_list = os.listdir(alon_recon)
alon_file_list.sort()

alon_recons = []
alon_stim = []
alon_models = []

for file in alon_file_list:
    if 'reconstructed' in file:
        alon_recons.append(file)
    elif 'stims' in file:
        alon_stim.append(file)
    elif 'model' in file:
        alon_models.append(file)


bhalla_file_list = os.listdir(bhalla_recon)
bhalla_file_list.sort()

bhalla_recons = []
bhalla_stim = []
bhalla_models = []

for file in bhalla_file_list:
    if 'reconstructed' in file:
        bhalla_recons.append(file)
    elif 'stims' in file:
        bhalla_stim.append(file)
    elif 'model' in file:
        bhalla_models.append(file)

def extract_files_to_list(dir_path, file_list):
    output_list = []
    for file in file_list:
        path = os.path.join(dir_path, file)
        data = pd.read_csv(path, header=None)
        output_list.append(data)
    return output_list

alon_stimreals = extract_files_to_list(alon_recon, alon_stim)
alon_reconstructed_stimuli = extract_files_to_list(alon_recon, alon_recons)

alon_recon = alon_reconstructed_stimuli[0].T.values
alon_recon.resize(2400)
alon_real_stim = alon_stimreals[0].T
alon_stim_mean = alon_real_stim.mean(axis=1).values - 0.4
alon_stim_mean.resize(2400)
# set up coherence parameters and multitaper windows
window_len = 256
NW = 2.5
n_tapers = 4
overlap = 0.5

windows, eigvals = dpss(window_len, NW, n_tapers, return_ratios=True)

# do coherence between mean stimulus and reconstruction
mean_multitaper_df = pd.DataFrame()

for window, eigval in zip(windows, eigvals):
    freq, cohere = scipy.signal.coherence(alon_stim_mean, alon_recon, fs=1000, window=window)
    cohere = pd.DataFrame(cohere)
    cohere_scale = cohere * eigval
    mean_multitaper_df = pd.concat([mean_multitaper_df, cohere_scale],
                                      axis=1,
                                      ignore_index=True)
    
coherence_from_mean = mean_multitaper_df.mean(axis=1)

# do coherence between each stimulus and reconstruction and then take mean
individual_coherences_df = pd.DataFrame()
for stim in alon_real_stim:
    real_stim = alon_real_stim.loc[:, stim].values - 0.4
    individual_multitaper_df = pd.DataFrame()
    
    for window, eigval in zip(windows, eigvals):
        freq, cohere = scipy.signal.coherence(real_stim, alon_recon, fs=1000, window=window)
        cohere = pd.DataFrame(cohere)
        cohere_scale = cohere * eigval
        individual_multitaper_df = pd.concat([individual_multitaper_df, cohere_scale],
                                      axis=1,
                                      ignore_index=True)
    individual_coherence = individual_multitaper_df.mean(axis=1)
    individual_coherences_df = pd.concat([individual_coherences_df, individual_coherence], axis=1, ignore_index=True)
coherence_from_individual = individual_coherences_df.mean(axis=1)
coherence_from_individual_median = individual_coherences_df.median(axis=1)
# compare individual_to_mean coherence to mean_stimulus_coherence
plt.figure()
plt.plot(freq, coherence_from_individual, label='individual')
plt.plot(freq, coherence_from_mean, label='mean_stimulus')
plt.xlabel('frequency (Hz)')
plt.ylabel('coherence')
plt.legend()
plt.savefig(os.path.join(figure_dir, 'mean_individual_coherence_compare.png'), dpi=300, format='png')

# median doesn't change it at all
plt.figure()
plt.plot(freq, coherence_from_individual_median, label='individual median')
plt.plot(freq, coherence_from_mean, label='mean_stimulus')
plt.xlabel('frequency (Hz)')
plt.ylabel('coherence')
plt.legend()
plt.savefig(os.path.join(figure_dir, 'mean_individual_coherence_compare.png'), dpi=300, format='png')

# plot all the coherences from individuals
plt.figure()
plt.plot(freq, individual_coherences_df)
plt.xlabel('frequency (Hz)')
plt.ylabel('coherence')
plt.savefig(os.path.join(figure_dir, 'individual_coherences.png'), dpi=300, format='png')

# define channels in channel_list, establish dictionary
bhalla_channel_list = ['kA', 'kca3', 'kfasttab', 'kslowtab', 'lcafixed', 'nafast']
bhalla_channels = {item:{} for item in channel_list}
ordered_channels = ['kA', 'kca3', 'lcafixed', 'kslowtab', 'kfasttab', 'nafast']

'''############### Open files, fill out data sets and save files ###########'''
# open glm files
reconstruction_df = pd.DataFrame()
stimreal_df = pd.DataFrame()
timeline_df = pd.DataFrame()

for file in alon_file_list:
    if file.endswith('.csv'):
        channel = file.split('_')[1]
        scale = file.split('_')[2].split('.csv')[0]
        labels = pd.DataFrame({'Channel': channel, 'Scale': scale}, index=range(1))

        if 'reconstructed' in file:
            reconstruction = pd.read_csv(os.path.join(alon_recon_dir, file), header=None).T
            labels_reconstruction = pd.concat([labels, reconstruction], axis=1)
            reconstruction_df = pd.concat([reconstruction_df, labels_reconstruction], ignore_index=True)

        elif 'stims' in file:
            stimreal = pd.read_csv(os.path.join(alon_recon_dir, file), header=None).T
            mean_stimreal = pd.DataFrame(stimreal.mean()).T
            labels_stimreal = pd.concat([labels, mean_stimreal], axis=1)
            stimreal_df = pd.concat([stimreal_df, labels_stimreal], ignore_index=True)

# since some values are missing from the GLM fits due to no GLM, make a full df
scale_list = ['0.01', '0.05', '0.2', '0.5', '0.8', '1', '1.2', '1.5', '2.0', '3']
full_labels_df = pd.DataFrame()

# the full_lables_df has all channels at all scales
for channel in channels:
    for scale in scale_list:
        label = pd.DataFrame({'Channel': channel, 'Scale': scale}, index=range(1))
        full_labels_df = pd.concat([full_labels_df, label], ignore_index=True)

# merge the full_lables_df with each df to make full stim, history, and bias
# full_coherence = pd.merge(full_labels_df, coherence_df, how='left', on=['Channel', 'Scale'])
full_frequency = pd.merge(full_labels_df, frequency_df, how='left', on=['Channel', 'Scale'])
full_reconstruction = pd.merge(full_labels_df, reconstruction_df, how='left', on=['Channel', 'Scale'])
full_stimreal = pd.merge(full_labels_df, stimreal_df, how='left', on=['Channel', 'Scale'])
full_timeline = pd.merge(full_labels_df, timeline_df, how='left', on=['Channel', 'Scale'])
# save all the full_df to file for use later if need be
full_coherence.to_csv(table_dir + 'coherence.csv', float_format='%10.6f', index=False)
full_frequency.to_csv(table_dir + 'frequency_filters.csv', float_format='%10.6f', index=False)
full_reconstruction.to_csv(table_dir + 'reconstruction.csv', float_format='%10.6f', index=False)
full_stimreal.to_csv(table_dir + 'stimreal.csv', float_format='%10.6f', index=False)
full_timeline.to_csv(table_dir + 'timeline.csv', float_format='%10.6f', index=False)
mean_stimreal.T.to_csv(table_dir + 'mean_stimreal.csv', float_format='%10.6f', index=False)
frequency.T.to_csv(table_dir + 'frequency_range.csv', float_format='%10.6f', index=False)

'''################## Make plots of Coherence ########################'''
# open files if not open already
mean_coherence_df = pd.read_csv(table_dir + 'coherence.csv')

frequency = pd.read_csv(table_dir + 'frequency_range.csv')

# settings for x and y axis minor  ticks, the  number stands for delta tick
x_minor = MultipleLocator(25)
y_minor = MultipleLocator(0.25)

# set time and number of scales and color palette
n_scales = 10
redblue10 = [u'#aa1016', u'#d52221', u'#f44f39', u'#fc8161', u'#fcaf93', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']
scale_list = ['1.50', '1.00', '0.50']
redblue3 = [ u'#6aaed6', 'k', u'#fc8161']    # for just 3 scales

# set the seaborn background style and generate plots
sns.set_style('ticks')

# make a subplot fig with 3 rows 2 cols 8"x8" with autosizing
fig, axs = plt.subplots(1, 6, figsize=(16.5, 1.75), constrained_layout=True)

for ax, channel in zip(axs.flat, ordered_channels):
    # pull out the STAs for each channel type individually
    channel_set = mean_coherence_df[mean_coherence_df['Channel'] == channel]

    # loop through all scales one by one
    for i in range(len(scale_list)):
        scale_slice = channel_set[channel_set['Scale'] == float(scale_list[i])]      # pull out scale row
        scale = scale_slice['Scale'].values          # define scale
        coherence = scale_slice.iloc[0, 2:]           # pull out STA convert to pA
        ax.plot(frequency, coherence, label=scale, color=redblue3[i], linewidth=2)
        # ax.set_title(channel, pad=-8)          # the pad moves the title down
        # ax.set_ylim(-25, 75)                    # make all y axes equal
        ax.xaxis.set_minor_locator(x_minor)     # add x axis minor ticks
        ax.yaxis.set_minor_locator(y_minor)     # add y axis minor ticks
        ax.set_yticks([0, 0.50, 1.00])
        ax.spines["right"].set_visible(False)   # if I want cleaner look w/out
        ax.spines["top"].set_visible(False)     # top and right axes lines
        ax.set_ylabel('Coherence Mag.')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_xlim(-5, 105)
# axs[1].set_ylim(-45, 225)
fig

# axs[0].set_ylabel('log firing rate')
# add legend
axs[7].legend(loc='center left', bbox_to_anchor=(1.1, 0.45))
fig
# save fig to file
fig.savefig(figure_dir + 'coherence.png', dpi=300, format='png')
fig.savefig(figure_dir + 'stim_filters_scaled_labels.png', dpi=300, format='png')

'''################## Make plots mean coherence in bands ###################'''
# open files if not open already
mean_coherence_df = pd.read_csv(table_dir + 'coherence.csv')
frequency = pd.read_csv(table_dir + 'frequency_range.csv')

# pull out indecies for frequency ranges
theta_min = frequency.index[frequency['0'] < 10].min()
theta_max = frequency.index[frequency['0'] < 10].max()
beta_min = frequency.index[(frequency['0'] > 10) & (frequency['0'] < 30)].min()
beta_max = frequency.index[(frequency['0'] > 10) & (frequency['0'] < 30)].max()
gamma_min = frequency.index[(frequency['0'] > 30) & (frequency['0'] < 100)].min()
gamma_max = frequency.index[(frequency['0'] > 30) & (frequency['0'] < 100)].max()

# convert to mean coherence in given bands
coherence = mean_coherence_df.iloc[:, 2:]

theta_slice = coherence.iloc[:, theta_min:theta_max+1]
theta_mean = theta_slice.mean(axis=1)

beta_slice = coherence.iloc[:, beta_min:beta_max+1]
beta_mean = beta_slice.mean(axis=1)

gamma_slice = coherence.iloc[:, gamma_min:gamma_max+1]
gamma_mean = gamma_slice.mean(axis=1)

coherence_bands = mean_coherence_df.iloc[:, :2]

coherence_bands['theta'] = theta_mean
coherence_bands['beta'] = beta_mean
coherence_bands['gamma'] = gamma_mean

# make plots for mean coherence over bands and scales
two_channels = ['kA', 'lcafixed']
range_list = ['theta', 'beta', 'gamma']
x_minor = MultipleLocator(0.5)

# set time and number of scales and color palette
dark2_2channel = [u'#1b9e77', u'#7570b3']

sns.set_style('ticks')

fig, axs = plt.subplots(3, 1, figsize=(3, 6), constrained_layout=True)
channel_ka = coherence_bands[coherence_bands['Channel'] == 'kA']
channel_cal = coherence_bands[coherence_bands['Channel'] == 'lcafixed']

scale = channel_ka['Scale']

theta_ka = channel_ka.loc[:, 'theta']
beta_ka = channel_ka.loc[:, 'beta']
gamma_ka = channel_ka.loc[:, 'gamma']

theta_cal = channel_cal.loc[:, 'theta']
beta_cal = channel_cal.loc[:, 'beta']
gamma_cal = channel_cal.loc[:, 'gamma']

axs[0].plot(scale, theta_ka, color=dark2_2channel[0], marker='o', label='kA')
axs[1].plot(scale, beta_ka, color=dark2_2channel[0], marker='o')
axs[2].plot(scale, gamma_ka, color=dark2_2channel[0], marker='o')

axs[0].plot(scale, theta_cal, color=dark2_2channel[1], marker='o', label='cal')
axs[1].plot(scale, beta_cal, color=dark2_2channel[1], marker='o')
axs[2].plot(scale, gamma_cal, color=dark2_2channel[1], marker='o')

axs[0].set_ylim(0.25, 0.8)
axs[1].set_ylim(0.1, 0.85)
axs[2].set_ylim(0.0, 0.4)

axs[0].set_title('0-10 Hz')
axs[1].set_title('10-30 Hz')
axs[2].set_title('30-100 Hz')

axs[2].set_xlabel('Scaling Factor')
for i in range(3):
    axs[i].spines["right"].set_visible(False)
    axs[i].spines["top"].set_visible(False)
    axs[i].set_ylabel('Coherence')
    axs[i].xaxis.set_minor_locator(x_minor)

axs[0].legend(loc='lower center', bbox_to_anchor=(0.4, 1.25), ncol=2)
fig

# save fig to file
fig.savefig(figure_dir + 'mean_coherence_bands.png', dpi=300, format='png')


'''################## Make plots of Stim real vs. reconstruction ########################'''
# open mean real stimulus and mean subtract
mean_real_stim = pd.read_csv(table_dir + 'mean_stimreal.csv')
mean_real_stim = mean_real_stim - 0.5

# open reconstructed stimuli df
recon_stim_df = pd.read_csv(table_dir + 'reconstruction.csv')
recon_stim_df

# settings for x and y axis minor  ticks, the  number stands for delta tick
x_minor = MultipleLocator(25)
y_minor = MultipleLocator(10)

# set time and number of scales and color palette
time = np.linspace(801, 3000, num=2200)
scale_list = ['1.50', '1.00', '0.50']
redblue10 = [u'#aa1016', u'#d52221', u'#f44f39', u'#fc8161', u'#fcaf93', 'k', u'#abd0e6', u'#6aaed6', u'#3787c0', u'#105ba4']
redblue3 = [ u'#6aaed6', 'k', u'#fc8161']    # for just 3 scales

# set the seaborn background style and generate plots
sns.set_style('ticks')

# make a subplot fig with 3 rows 2 cols 8"x8" with autosizing
fig, axs = plt.subplots(1, 6, figsize=(16.5, 1.1), constrained_layout=True)

for ax, channel in zip(axs.flat, ordered_channels):
    # pull out the STAs for each channel type individually
    channel_set = recon_stim_df[recon_stim_df['Channel'] == channel]
    ax.plot(time, mean_real_stim, label='real avg', color='lightgray', linewidth=4)
    # loop through all scales one by one
    for i in range(len(scale_list)):
        scale_slice = channel_set[channel_set['Scale'] == float(scale_list[i])]       # pull out scale row
        scale = scale_slice['Scale'].values            # define scale
        recon_stim = scale_slice.iloc[0, 2:]           # pull out STA convert to pA
        ax.plot(time, recon_stim, label=scale, color=redblue3[i], linewidth=2)
        ax.set_xlim(2700, 2900)
        # ax.set_title(channel, pad=-8)          # the pad moves the title down
        ax.set_ylim(-0.37, 0.275)                    # make all y axes equal
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.xaxis.set_minor_locator(x_minor)     # add x axis minor ticks
        # ax.yaxis.set_minor_locator(y_minor)     # add y axis minor ticks
        ax.spines["right"].set_visible(False)   # if I want cleaner look w/out
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)     # top and right axes lines
        ax.spines["left"].set_visible(False)
        # ax.set_ylabel('logit FR (spikes/ms)')
        # ax.set_xlabel('Time (ms)')
axs[5].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
fig


# save fig to file
fig.savefig(figure_dir + 'reconstruction.png', dpi=300, format='png')
fig.savefig(figure_dir + 'reconstruction_labels.png', dpi=300, format='png')
