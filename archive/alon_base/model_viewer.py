import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

home_dir = os.path.expanduser("~")
project_dir = home_dir + '/ngglasgow@gmail.com/Data_Urban/NEURON/AlmogAndKorngreen2014/par_ModCell_5_thrdsafe/'
data_dir = project_dir + 'scaled_wns_400_8015_output/'
figure_dir = project_dir + 'analysis/figures/'
table_dir = project_dir + 'analysis/tables/'
noise_dir = project_dir + 'scaled_wns_400/'
noise_filename = noise_dir + 'wns_8015_forscaled.txt'
downsample_dir = project_dir + 'alon_scaled_8015_downsampled/'

os.chdir(project_dir)

from neuron import h, gui

h.load_file('stdgui.hoc')
h.load_file('stdrun.hoc')

h.load_file('cell.hoc')

# create a Shape plot object
morph = h.Shape()

# 3d plot shape options
morph.show(0)   # show as filled volume
morph.show(1)   # show as lines through centroids
morph.show(2)   # show as linear schematic

# unsure how to programmatically change the size, but it is very to do in the
# gui, so just do that, and make it "whole scene" option

# translate 90 degrees over each axis
morph.rotate(0, 0, 0, 1.5708, 0, 0)     # over x axis
morph.rotate(0, 0, 0, 0, 1.5708, 0)     # over y axis
morph.rotate(0, 0, 0, 0, 0, 0.1)     # over z axis

# save current view to file as a post script image
morph.printfile(figure_dir + 'alon_morphology_section.ps')

# get all channel conductnaces as a function of distance from soma in apical dend
dist_list = []
bk_list = []
cah_list = []
car_list = []
iA_list = []
iH_list = []
kslow_list = []
na_list = []
sk_list = []

distance_g_df = pd.DataFrame()

for sec in h.allsec():
    for i in range(85):
        if sec == h.apic[i]:
            seg = sec(0.5)
            dist_list.append(h.distance(seg))
            bk_list.append(seg.bk.gbar)
            cah_list.append(seg.cah.pbar)
            car_list.append(seg.car.pbar)
            iA_list.append(seg.iA.gbar)
            iH_list.append(seg.iH.gbar)
            kslow_list.append(seg.kslow.gbar)
            na_list.append(seg.na.gbar)
            sk_list.append(seg.sk.gbar)

distance_g_df['distance'] = dist_list
distance_g_df['bk'] = bk_list
distance_g_df['cah'] = cah_list
distance_g_df['car'] = car_list
distance_g_df['iA'] = iA_list
distance_g_df['iH'] = iH_list
distance_g_df['kslow'] = kslow_list
distance_g_df['na'] = na_list
distance_g_df['sk'] = sk_list
distance_g_df

distance_g_sorted = distance_g_df.sort_values('distance')
distance_g_sorted

ordered_channels = ['iA', 'iH', 'sk', 'cah', 'car', 'bk', 'kslow', 'na']
dark2 = [u'#1b9e77', u'#d95f02', u'#7570b3', u'#e7298a', u'#66a61e', u'#e6ab02', u'#a6761d', u'#666666']

dist = distance_g_sorted['distance']

sns.set_style('ticks')
fig, axs = plt.subplots(4, 2, figsize=(5, 6), constrained_layout=True)

for ax, channel, i in zip(axs.flat, ordered_channels, range(len(ordered_channels))):
    g = distance_g_sorted[channel]
    ax.plot(dist, g, color=dark2[i])
    ax.set_title(channel)
    ax.set_xlim(-20, 1020)
    ax.set_xticks([0, 500, 1000])
    # ax.set_xticklabels(labels=[])
axs[3, 0].set_xlabel('Distance from Soma (uM)')
axs[3, 1].set_xlabel('Distance from Soma (uM)')
axs[3, 0].set_ylabel('G (pS/uM^2)')
fig
fig.savefig(figure_dir + 'conductance_density.png', dpi=300, format='png')
quit()
