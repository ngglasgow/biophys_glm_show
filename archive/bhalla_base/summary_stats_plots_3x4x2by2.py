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

ordered_channels = ['kA', 'kca3', 'lcafixed', 'kslowtab', 'kfasttab', 'nafast']


# open the  spiketrain statistics file if not already open
stats_df = pd.read_csv(table_dir + 'scaled_spiketrain_stats.csv')
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')

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
stat_list = ['Firing Rate (Hz)', 'CV ISI', 'Fano Factor', 'Correlation']
feature_list = ['Positive Peak (pA)', 'Pos. Peak Time (ms)', 'Negative Peak (pA)', 'Neg. Peak Time (ms)']

important_channels = ['kA', 'kca3', 'lcafixed']
na_k_channels = ['kslowtab', 'kfasttab',  'nafast']

# create figure for all four stats, two channel types, then sta features
fig, axs = plt.subplots(3, 4, figsize=(12.5, 6), constrained_layout=True)
# add modulatory channel row for spiketrain stats
for col, stat in zip(range(4), stat_list):
    stat_set = stats_df.loc[:, ['Channel', 'Scale', stat]]
    for channel, i in zip(important_channels, range(3)):
        channel_stats = stat_set[stat_set['Channel'] == channel]
        scale = channel_stats['Scale']
        axs[0, col].plot(scale, channel_stats[stat], label=channel, marker='o', color=dark2[i])
        axs[0, col].xaxis.set_minor_locator(x_minor)
        axs[0, col].set_xlim(-0.2, 3.2)
    axs[0, col].set_title(stat)
fig
# axs[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

# add non modulatory channel row for spiketrain stats
for col, stat in zip(range(4), stat_list):
    stat_set = stats_df.loc[:, ['Channel', 'Scale', stat]]
    for channel, i in zip(na_k_channels, range(3)):
        channel_stats = stat_set[stat_set['Channel'] == channel]
        scale = channel_stats['Scale']
        axs[1, col].plot(scale, channel_stats[stat], label=channel, marker='o', color=dark2[i+3])
        axs[1, col].xaxis.set_minor_locator(x_minor)
        axs[1, col].set_xlim(-0.2, 3.2)
    # axs[1, col].set_title(stat)
fig

# add non modulatory channel row for spiketrain stats
for col, feature in zip(range(4), feature_list):
    feature_set = sta_features.loc[:, ['Channel', 'Scale', feature]]
    for channel, i in zip(na_k_channels, range(3)):
        channel_features = feature_set[feature_set['Channel'] == channel]
        scale = channel_features['Scale']
        axs[2, col].plot(scale, channel_features[feature], label=channel, marker='o', color=dark2[i])
        axs[2, col].xaxis.set_minor_locator(x_minor)
        axs[2, col].set_xlim(-0.2, 3.2)
    axs[2, col].set_title(feature)
fig

axs[0, 0].set_ylabel('Current (pA)')
axs[0, 1].set_ylabel('Scaling Factor (fold-change)')
axs[1, 0].set_ylabel('Current (pA)')
axs[1, 1].set_ylabel('Scaling Factor (fold-change)'y)


# save fig to file
fig.savefig(figure_dir + 'STAs_notopright.png', dpi=300, format='png')


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
