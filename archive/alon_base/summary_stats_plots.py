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
project_dir = home_dir + '/ngglasgow@gmail.com/Data_Urban/NEURON/AlmogAndKorngreen2014/par_ModCell_5_thrdsafe/'
data_dir = project_dir + 'scaled_wns_400_8015_output/'
figure_dir = project_dir + 'analysis/figures/'
table_dir = project_dir + 'analysis/tables/'
noise_dir = project_dir + 'scaled_wns_400/'
noise_filename = noise_dir + 'wns_8015_forscaled.txt'
downsample_dir = project_dir + 'alon_scaled_8015_downsampled/'

ordered_channels = ['iA', 'iH', 'sk', 'cah', 'car', 'bk', 'kslow', 'na']

'''################ Plot Spiketrain Stats on One plot with #################'''
# open the  spiketrain statistics file if not already open
stats_df = pd.read_csv(table_dir + 'scaled_spiketrain_stats.csv')
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')

# quick fix for making 0 Hz a nan
list0 = stats_df.index[stats_df['Firing Rate (Hz)'] == 0.0].tolist()
for item in list0:
    stats_df['Firing Rate (Hz)'].iloc[item] = np.nan
stats_df

# settings for x and y axis minor  ticks, the  number stands for delta tick
x_minor = MultipleLocator(0.5)

# set the seaborn background style and generate plots
sns.set_style('ticks')
dark2 = [u'#1b9e77', u'#d95f02', u'#7570b3', u'#e7298a', u'#66a61e', u'#e6ab02', u'#a6761d', u'#666666']

# make a subplot fig with 3 rows 2 cols 8"x8" with autosizing
stat_list = ['Firing Rate (Hz)', 'CV ISI', 'Correlation']
feature_list = ['Positive Peak (pA)', 'Pos. Peak Time (ms)', 'Negative Peak (pA)', 'Neg. Peak Time (ms)']

important_channels = ['iA', 'iH', 'cah', 'sk']
na_k_channels = ['na', 'kslow', 'bk', 'car']

# create figure for all four stats, two channel types, then sta features
fig, axs = plt.subplots(3, 2, figsize=(5, 5.5), constrained_layout=True)
# add modulatory channel row for spiketrain stats
for row, stat in zip(range(3), stat_list):
    stat_set = stats_df.loc[:, ['Channel', 'Scale', stat]]
    for channel, i in zip(important_channels, range(4)):
        channel_stats = stat_set[stat_set['Channel'] == channel]
        scale = channel_stats['Scale']
        axs[row, 0].plot(scale, channel_stats[stat], label=channel, marker='o', color=dark2[i])
        axs[row, 0].set_xticks([0, 1, 2])
        axs[row, 0].set_xlim(-0.2, 2.2)
        axs[row, 0].xaxis.set_minor_locator(x_minor)
        axs[row, 0].spines['right'].set_visible(False)
        axs[row, 0].spines['top'].set_visible(False)
    axs[row, 0].set_ylabel(stat)

# add non mod channels
for row, stat in zip(range(3), stat_list):
    stat_set = stats_df.loc[:, ['Channel', 'Scale', stat]]
    for channel, i in zip(na_k_channels, range(4)):
        channel_stats = stat_set[stat_set['Channel'] == channel]
        scale = channel_stats['Scale']
        axs[row, 1].plot(scale, channel_stats[stat], label=channel, marker='o', color=dark2[i+3])
        axs[row, 1].set_xticks([0, 1, 2])
        axs[row, 1].set_xlim(-0.2, 2.2)
        axs[row, 1].xaxis.set_minor_locator(x_minor)
        axs[row, 1].spines['right'].set_visible(False)
        axs[row, 1].spines['top'].set_visible(False)

# set x-labels
axs[2, 1].set_xlabel('Scaling Factor')
axs[2, 0].set_xlabel('Scaling Factor')
fig
# set yscales and minor ticks for each row and col
fr_yscale = (-5, 100)
cv_yscale = (-0.05, 1.4)
corr_yscale = (0.08, 0.4)
y_scale_list = [fr_yscale, cv_yscale, corr_yscale]

fr_yminor = MultipleLocator(25)
cv_yminor = MultipleLocator()
corr_yminor = MultipleLocator(0.05)
yminor_list = [fr_yminor, cv_yminor, corr_yminor]

for row in range(3):
    for col in range(2):
        axs[row, col].set_ylim(y_scale_list[row])
        axs[row, col].yaxis.set_minor_locator(yminor_list[row])

fig

# save fig to file
fig.savefig(figure_dir + 'stats_summary_vertical55.png', dpi=300, format='png')

'''################ Plot Spiketrain Stats on One plot horizontally #################'''
# open the  spiketrain statistics file if not already open
stats_df = pd.read_csv(table_dir + 'scaled_spiketrain_stats.csv')
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')

# quick fix for making 0 Hz a nan
list0 = stats_df.index[stats_df['Firing Rate (Hz)'] == 0.0].tolist()
for item in list0:
    stats_df['Firing Rate (Hz)'].iloc[item] = np.nan
stats_df

# settings for x and y axis minor  ticks, the  number stands for delta tick
x_minor = MultipleLocator(0.5)

# set the seaborn background style and generate plots
sns.set_style('ticks')
dark2 = [u'#1b9e77', u'#d95f02', u'#7570b3', u'#e7298a', u'#66a61e', u'#e6ab02', u'#a6761d', u'#666666']

# make a subplot fig with 3 rows 2 cols 8"x8" with autosizing
stat_list = ['Firing Rate (Hz)', 'CV ISI', 'Correlation']
feature_list = ['Positive Peak (pA)', 'Pos. Peak Time (ms)', 'Negative Peak (pA)', 'Neg. Peak Time (ms)']

important_channels = ['iA', 'iH', 'cah', 'sk']
na_k_channels = ['na', 'kslow', 'bk', 'car']

# create figure for all four stats, two channel types, then sta features
fig, axs = plt.subplots(1, 6, figsize=(16.5, 1.75), constrained_layout=True)
# add modulatory channel row for spiketrain stats
for col, stat in zip(np.arange(0, 6, 2), stat_list):
    stat_set = stats_df.loc[:, ['Channel', 'Scale', stat]]
    for channel, i in zip(important_channels, range(3)):
        channel_stats = stat_set[stat_set['Channel'] == channel]
        scale = channel_stats['Scale']
        axs[col].plot(scale, channel_stats[stat], label=channel, marker='o', color=dark2[i])
    axs[col].set_xticks([0, 1, 2, 3])
    axs[col].set_xlim(-0.2, 3.2)
    axs[col].xaxis.set_minor_locator(x_minor)
    axs[col].spines['right'].set_visible(False)
    axs[col].spines['top'].set_visible(False)
    axs[col].set_xlabel('Scaling Factor')
    axs[col].set_ylabel(stat)

# add non mod channels
for col, stat in zip(np.arange(1, 6, 2), stat_list):
    stat_set = stats_df.loc[:, ['Channel', 'Scale', stat]]
    for channel, i in zip(na_k_channels, range(3)):
        channel_stats = stat_set[stat_set['Channel'] == channel]
        scale = channel_stats['Scale']
        axs[col].plot(scale, channel_stats[stat], label=channel, marker='o', color=dark2[i+3])
    axs[col].set_xticks([0, 1, 2, 3])
    axs[col].set_xlim(-0.2, 3.2)
    axs[col].xaxis.set_minor_locator(x_minor)
    axs[col].spines['right'].set_visible(False)
    axs[col].spines['top'].set_visible(False)
    axs[col].set_xlabel('Scaling Factor')
    # axs[col].set_ylabel(stat)

fig
# set yscales and minor ticks for each row and col
fr_yscale = (20, 70)
cv_yscale = (0.10, 0.4)
corr_yscale = (0.08, 0.26)
y_scale_list = [fr_yscale, cv_yscale, corr_yscale]

fr_yminor = MultipleLocator(10)
cv_yminor = MultipleLocator(0.1)
corr_yminor = MultipleLocator(0.05)
yminor_list = [fr_yminor, cv_yminor, corr_yminor]

col_list = [[0, 1], [2, 3], [4, 5]]
for cols, i in zip(col_list, range(3)):
    for col in cols:
        axs[col].set_ylim(y_scale_list[i])
        axs[col].yaxis.set_minor_locator(yminor_list[i])

fig

# save fig to file
fig.savefig(figure_dir + 'stats_summary_labels_horiz.png', dpi=300, format='png')

'''######################## STA correlations ###############################'''
sta_mean = pd.read_csv(table_dir + 'sta_mean.csv').T
sta_mean.T

stas = sta_mean.iloc[3:-50] * 1000
stas = stas.astype(float)
sta_mean
sta_corr = stas.corr()

sns_plot = sns.heatmap(sta_corr, vmin=0.5, cmap='inferno', xticklabels=10, yticklabels=10, square=True, robust=True, linecolor='w')

fig = sns_plot.get_figure()
fig.savefig(figure_dir + 'sta_correlations.png', dpi=300, format='png')

'''############### STA Features vs. Channel Conductance ####################'''
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')
dark2 = [u'#1b9e77', u'#d95f02', u'#7570b3', u'#e7298a', u'#66a61e', u'#e6ab02', u'#a6761d', u'#666666']
sns.set_style('ticks')

feature_list = ['Positive Peak (pA)', 'Negative Peak (pA)', 'Max Slope (pA/ms)', 'Integration Time (ms)']

important_channels = ['iA', 'iH']
na_k_channels = ['na', 'kslow', 'bk', 'car']

# put into a single figure
x_minor = MultipleLocator(0.5)
y_minor = MultipleLocator(10)

gs_kw = dict(height_ratios=[1, 1, 1.5])
fig, axs = plt.subplots(3, 2, figsize=(6, 5.5), constrained_layout=True, gridspec_kw=gs_kw)

for ax, feature in zip(axs.flat, feature_list):
    feature_set = sta_features.loc[:, ['Channel', 'Scale', feature]]
    for channel, i in zip(important_channels, range(4)):
        channel_feature = feature_set[feature_set['Channel'] == channel]
        scale = channel_feature['Scale']
        ax.plot(scale, channel_feature[feature], label=channel, marker='o', color=dark2[i])
    ax.set_xlim(-0.2, 2.2)
    ax.xaxis.set_minor_locator(x_minor)
    # ax.set_ylabel(feature)
    # ax.set_ylim(-25, 55)
    # ax.yaxis.set_minor_locator(y_minor)
    ax.spines["right"].set_visible(False)   # if I want cleaner look w/out
    ax.spines["top"].set_visible(False)     # top and right axes lines

# set bottom x labels
axs[1, 0].set_xlabel('Scaling Factor')
axs[1, 1].set_xlabel('Scaling Factor')

axs[0, 0].set_ylabel('Pos. Peak (pA)')
axs[0, 1].set_ylabel('Neg. Peak (pA)')
axs[1, 0].set_ylabel('Slope (pA/ms)')
axs[1, 1].set_ylabel('Int. Time (ms)')
axs[1, 1].set_ylim(0, 15)
# fig
# set yscales and minor ticks for each row and col
# axs[0, 0].set_ylim(15, 55)
# axs[0, 0].yaxis.set_minor_locator(MultipleLocator(10))
# axs[0, 1].set_ylim(-25, -7.5)
# axs[0, 1].yaxis.set_minor_locator(MultipleLocator(5))
#axs[1, 0].set_ylim(2, 10)
# axs[1, 0].yaxis.set_minor_locator(MultipleLocator(10))
#axs[1, 1].set_ylim(0, 15)
# axs[1, 1].yaxis.set_minor_locator(MultipleLocator(10))
#fig
list0 = sta_features.index[sta_features['Integration Time (ms)'] > 20.0].tolist()
for item in list0:
    sta_features['Integration Time (ms)'].iloc[item] = np.nan
# add the bottom two plots
for channel, i in zip(important_channels, range(4)):
    channel_set = sta_features[feature_set['Channel'] == channel]

    scale01 = channel_set['Scale'][channel_set['Scale'] == 0.50].index
    scale1 = channel_set['Scale'][channel_set['Scale'] == 1.00].index
    scale3 = channel_set['Scale'][channel_set['Scale'] == 1.50].index
    pos_peak = channel_set['Positive Peak (pA)']
    neg_peak = channel_set['Negative Peak (pA)']
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    axs[2, 0].plot(pos_peak, slope, color=dark2[i], marker='.')
    axs[2, 1].plot(slope, int_time, color=dark2[i], marker='.')

axs[2, 0].plot(pos_peak.loc[scale1], slope.loc[scale1], linestyle='', marker='o', markersize=8, color='k')
axs[2, 1].plot(slope.loc[scale1], int_time.loc[scale1], linestyle='', marker='o', markersize=8, color='k')

for channel, i in zip(important_channels, range(4)):
    channel_set = sta_features[feature_set['Channel'] == channel]
    scale01 = channel_set['Scale'][channel_set['Scale'] == 0.50].index
    scale1 = channel_set['Scale'][channel_set['Scale'] == 1.00].index
    scale3 = channel_set['Scale'][channel_set['Scale'] == 1.50].index
    pos_peak = channel_set['Positive Peak (pA)']
    neg_peak = channel_set['Negative Peak (pA)']
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    axs[2, 0].plot(pos_peak.loc[scale01], slope.loc[scale01], linestyle='', marker=11, markersize=10, color=dark2[i], markeredgecolor='k')
    axs[2, 0].plot(pos_peak.loc[scale3], slope.loc[scale3], linestyle='', marker=10, markersize=10, color=dark2[i], markeredgecolor='k')
    axs[2, 1].plot(slope.loc[scale01], int_time.loc[scale01], linestyle='', marker=11, markersize=10, color=dark2[i], markeredgecolor='k')
    axs[2, 1].plot(slope.loc[scale3], int_time.loc[scale3], linestyle='', marker=10, markersize=10, color=dark2[i], markeredgecolor='k')

axs[2, 0].set_xlabel('Pos. Peak (pA)')
axs[2, 0].set_ylabel('Slope (pA/ms)')
axs[2, 0].spines["right"].set_visible(False)   # if I want cleaner look w/out
axs[2, 0].spines["top"].set_visible(False)     # top and right axes lines

axs[2, 1].set_ylim(0, 15)
axs[2, 1].set_xlabel('Slope (pA/ms)')
axs[2, 1].set_ylabel('Int. Time (ms)')
axs[2, 1].spines["right"].set_visible(False)   # if I want cleaner look w/out
axs[2, 1].spines["top"].set_visible(False)     # top and right axes lines

fig
# save to file
fig.savefig(figure_dir + 'sta_feature_summary_vertical55_2.png', dpi=300, format='png')


'''############### STA Features vs. Scales Horizontal  ####################'''
% matplotlib inline
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')

feature_list = ['Positive Peak (pA)', 'Negative Peak (pA)', 'Max Slope (pA/ms)', 'Integration Time (ms)']

important_channels = ['kA', 'kca3', 'lcafixed']
na_k_channels = ['nafast', 'kslowtab', 'kfasttab']

# put into a single figure
x_minor = MultipleLocator(0.5)

fig, axs = plt.subplots(1, 6, figsize=(16.5, 2.25), constrained_layout=True)

for ax, feature in zip(axs.flat, feature_list):
    feature_set = sta_features.loc[:, ['Channel', 'Scale', feature]]
    for channel, i in zip(important_channels, range(4)):
        channel_feature = feature_set[feature_set['Channel'] == channel]
        scale = channel_feature['Scale']
        ax.plot(scale, channel_feature[feature], label=channel, marker='o', color=dark2[i])
    ax.set_xlim(-0.2, 3.2)
    ax.xaxis.set_minor_locator(x_minor)
    ax.set_ylabel(feature)
    # ax.set_ylim(-25, 55)
    # ax.yaxis.set_minor_locator(y_minor)
    ax.spines["right"].set_visible(False)   # if I want cleaner look w/out
    ax.spines["top"].set_visible(False)     # top and right axes lines
    ax.set_xlabel('Scaling Factor')

# set y limits
axs[0].set_ylim(15, 55)
axs[1].set_ylim(-25, -7.5)
axs[2].set_ylim(2, 10)
axs[3].set_ylim(7, 15)

# add sta_feature vs. feature plots
for channel, i in zip(important_channels, range(4)):
    channel_set = sta_features[feature_set['Channel'] == channel]
    scale01 = channel_set['Scale'][channel_set['Scale'] == 0.01].index
    scale1 = channel_set['Scale'][channel_set['Scale'] == 1.00].index
    scale3 = channel_set['Scale'][channel_set['Scale'] == 3.00].index
    pos_peak = channel_set['Positive Peak (pA)']
    neg_peak = channel_set['Negative Peak (pA)']
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    axs[4].plot(pos_peak, slope, color=dark2[i], marker='.')
    axs[5].plot(slope, int_time, color=dark2[i], marker='.')

axs[4].plot(pos_peak.loc[scale1], slope.loc[scale1], linestyle='', marker='o', markersize=8, color='k')
axs[5].plot(slope.loc[scale1], int_time.loc[scale1], linestyle='', marker='o', markersize=8, color='k')

for channel, i in zip(important_channels, range(4)):
    channel_set = sta_features[feature_set['Channel'] == channel]
    scale01 = channel_set['Scale'][channel_set['Scale'] == 0.01].index
    scale1 = channel_set['Scale'][channel_set['Scale'] == 1.00].index
    scale3 = channel_set['Scale'][channel_set['Scale'] == 3.00].index
    pos_peak = channel_set['Positive Peak (pA)']
    neg_peak = channel_set['Negative Peak (pA)']
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    axs[4].plot(pos_peak.loc[scale01], slope.loc[scale01], linestyle='', marker=11, markersize=10, color=dark2[i], markeredgecolor='k')
    axs[4].plot(pos_peak.loc[scale3], slope.loc[scale3], linestyle='', marker=10, markersize=10, color=dark2[i], markeredgecolor='k')
    axs[5].plot(slope.loc[scale01], int_time.loc[scale01], linestyle='', marker=11, markersize=10, color=dark2[i], markeredgecolor='k')
    axs[5].plot(slope.loc[scale3], int_time.loc[scale3], linestyle='', marker=10, markersize=10, color=dark2[i], markeredgecolor='k')

axs[4].set_xlabel(pos_peak.name)
axs[4].set_ylabel(slope.name)
axs[4].spines["right"].set_visible(False)   # if I want cleaner look w/out
axs[4].spines["top"].set_visible(False)     # top and right axes lines

axs[5].set_xlabel(slope.name)
axs[5].set_ylabel(int_time.name)
axs[5].spines["right"].set_visible(False)   # if I want cleaner look w/out
axs[5].spines["top"].set_visible(False)     # top and right axes lines

fig

# save to file
fig.savefig(figure_dir + 'sta_feature_summary_horizontal2.png', dpi=300, format='png')

'''################ make channel legends for bhalla model ##################'''
fig, ax = plt.subplots(1, 2, constrained_layout=True)
for channel, i in zip(ordered_channels, range(8)):
    channel_set = sta_features['Scale'][sta_features['Channel'] == channel]
    ax[0].plot(channel_set, marker='o', color=dark2[i], label=channel)
ax[0].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
fig
fig.savefig(figure_dir + 'channel_legend.png', dpi=300, format='png')


'''#################### STA Features vs. STA Features ######################'''
% matplotlib
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')

important_channels = ['kA', 'kca3', 'lcafixed']

fig, axs = plt.subplots(2, 1, figsize=(3, 5.25), constrained_layout=True)

for channel, i in zip(important_channels, range(3)):
    channel_set = sta_features[feature_set['Channel'] == channel]
    scale01 = channel_set['Scale'][channel_set['Scale'] == 0.01].index
    scale1 = channel_set['Scale'][channel_set['Scale'] == 1.00].index
    scale3 = channel_set['Scale'][channel_set['Scale'] == 3.00].index
    pos_peak = channel_set['Positive Peak (pA)']
    neg_peak = channel_set['Negative Peak (pA)']
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    axs[0].plot(pos_peak, neg_peak, color=dark2[i])
    axs[1].plot(slope, int_time, color=dark2[i])

for channel, i in zip(important_channels, range(3)):
    channel_set = sta_features[feature_set['Channel'] == channel]
    scale01 = channel_set['Scale'][channel_set['Scale'] == 0.01].index
    scale1 = channel_set['Scale'][channel_set['Scale'] == 1.00].index
    scale3 = channel_set['Scale'][channel_set['Scale'] == 3.00].index
    pos_peak = channel_set['Positive Peak (pA)']
    neg_peak = channel_set['Negative Peak (pA)']
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    axs[0].plot(pos_peak.loc[scale01], neg_peak.loc[scale01], linestyle='', marker='^', markersize=10, color=dark2[i])
    axs[0].plot(pos_peak.loc[scale1], neg_peak.loc[scale1], linestyle='', marker='o', markersize=10, color='k')
    axs[0].plot(pos_peak.loc[scale3], neg_peak.loc[scale3], linestyle='', marker='v', markersize=10, color=dark2[i])
    axs[1].plot(slope.loc[scale01], int_time.loc[scale01], linestyle='', marker='^', markersize=10, color=dark2[i])
    axs[1].plot(slope.loc[scale1], int_time.loc[scale1], linestyle='', marker='o', markersize=10, color='k')
    axs[1].plot(slope.loc[scale3], int_time.loc[scale3], linestyle='', marker='v', markersize=10, color=dark2[i])

axs[0].set_xlabel('Pos. Peak (pA)')
axs[0].set_ylabel('Neg. Peak (pA)')
axs[1].set_xlabel('Max Slope (pA/ms)')
axs[1].set_ylabel('Integration Time (ms)')
fig

# save fig
fig.savefig(figure_dir + 'posvneg_slopevint.png', dpi=300, format='png')


'''#################### STA Features vs. Spike Stats ######################'''
% matplotlib
sta_features = pd.read_csv(table_dir + 'sta_features_index.csv')

important_channels = ['kA', 'kca3', 'lcafixed']

fig, axs = plt.subplots(2, 1, figsize=(3, 5.25), constrained_layout=True)

for channel, i in zip(important_channels, range(3)):
    channel_set = sta_features[feature_set['Channel'] == channel]
    scale01 = channel_set['Scale'][channel_set['Scale'] == 0.01].index
    scale1 = channel_set['Scale'][channel_set['Scale'] == 1.00].index
    scale3 = channel_set['Scale'][channel_set['Scale'] == 3.00].index
    pos_peak = channel_set['Positive Peak (pA)']
    neg_peak = channel_set['Negative Peak (pA)']
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    axs[0].plot(pos_peak, neg_peak, color=dark2[i])
    axs[1].plot(slope, int_time, color=dark2[i])

for channel, i in zip(important_channels, range(3)):
    channel_set = sta_features[feature_set['Channel'] == channel]
    scale01 = channel_set['Scale'][channel_set['Scale'] == 0.01].index
    scale1 = channel_set['Scale'][channel_set['Scale'] == 1.00].index
    scale3 = channel_set['Scale'][channel_set['Scale'] == 3.00].index
    pos_peak = channel_set['Positive Peak (pA)']
    neg_peak = channel_set['Negative Peak (pA)']
    slope = channel_set['Max Slope (pA/ms)']
    int_time = channel_set['Integration Time (ms)']
    axs[0].plot(pos_peak.loc[scale01], neg_peak.loc[scale01], linestyle='', marker='^', markersize=10, color=dark2[i])
    axs[0].plot(pos_peak.loc[scale1], neg_peak.loc[scale1], linestyle='', marker='o', markersize=10, color='k')
    axs[0].plot(pos_peak.loc[scale3], neg_peak.loc[scale3], linestyle='', marker='v', markersize=10, color=dark2[i])
    axs[1].plot(slope.loc[scale01], int_time.loc[scale01], linestyle='', marker='^', markersize=10, color=dark2[i])
    axs[1].plot(slope.loc[scale1], int_time.loc[scale1], linestyle='', marker='o', markersize=10, color='k')
    axs[1].plot(slope.loc[scale3], int_time.loc[scale3], linestyle='', marker='v', markersize=10, color=dark2[i])

axs[0].set_xlabel('Pos. Peak (pA)')
axs[0].set_ylabel('Neg. Peak (pA)')
axs[1].set_xlabel('Max Slope (pA/ms)')
axs[1].set_ylabel('Integration Time (ms)')
fig

# save fig
fig.savefig(figure_dir + 'posvneg_slopevint.png', dpi=300, format='png')


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
