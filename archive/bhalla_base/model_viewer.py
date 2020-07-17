import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
% matplotlib inline

home_dir = os.path.expanduser("~")
project_dir = home_dir + '/ngglasgow@gmail.com/Data_Urban/NEURON/Bhalla_par_scaling/'
data_dir = project_dir + 'scaled_wns_500_9520_output/'
figure_dir = project_dir + 'analysis/figures/'
table_dir = project_dir + 'analysis/tables/'
noise_dir = project_dir + 'scaled_wns_500/'
noise_filename = noise_dir + 'wns_9520_forscaled.txt'
downsample_dir = project_dir + 'bhalla_scaled_9520_downsampled/'

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
morph.rotate(0, 0, 0, 0, 0, 1.5708)     # over z axis

# save current view to file as a post script image
morph.printfile(figure_dir + 'bhalla_morphology_section.ps')

# pull out conductances as function of distance from soma or in different compartments
for sec in h.allsec():
    h.psection(sec)

for sec in h.allsec():

    if sec == h.soma:
        h.psection(sec)

    elif sec == h.axon[0]:
        h.psection(sec)

    elif sec == h.axon[5]:
        h.psection(sec)

    elif sec == h.prim_dend[4]:
        h.psection(sec)

    elif sec == h.glom[1]:
        h.psection(sec)

    elif sec == h.sec_dendd1[0][1]:
        h.psection(sec)

    elif sec == h.sec_dendd1[1][1]:
        h.psection(sec)

for sec in h.allsec():
    if 'glom' in str(sec):
        for seg in sec.allseg():
            if "kslowtab" in dir(seg):
                seg.kslowtab.gkbar = 0.014

quit()


# take values from Bhalla Paper
ordered_channels = ['kA', 'kca3', 'lcafixed', 'kslowtab', 'kfasttab', 'nafast']
dark2 = [u'#1b9e77', u'#d95f02', u'#7570b3', u'#e7298a', u'#66a61e', u'#e6ab02', u'#a6761d', u'#666666']

axon=[51.5, 88.7, 20.0, 15.5, 1541.0, 4681.0]
soma=[58.7, 142.0, 40.0, 28.0, 1956.0, 1532.0]
primdend=[0, 0, 22, 17.4, 12.3, 13.4]
glom=[0, 0, 95, 28, 0, 0]
secprox=[0, 0, 4, 8.5, 226, 330]
secdist=[0, 0, 0, 0, 128, 122]

channels_df = pd.DataFrame()

channels_df['Ax.'] = axon
channels_df['S'] = soma
channels_df['Ap.'] = primdend
channels_df['T'] = glom
channels_df['L'] = secprox
channels_df['L.Dist.'] = secdist
channels_df.index = ordered_channels

sns.set_style('ticks')
fig, axs = plt.subplots(2, 3, figsize=(6, 3), constrained_layout=True)
labels = ['Axon', 'Soma', 'Apic.', 'Tuft', 'Lat.', 'L.Dist.']
for ax, channel, i in zip(axs.flat, ordered_channels, range(len(ordered_channels))):
    ax.plot(channels_df.columns, channels_df.loc[channel], color=dark2[i], )
    ax.set_xticklabels(labels=[])
    ax.set_title(channel)

axs[1, 0].set_xticklabels(labels=labels, rotation=90)
axs[1, 1].set_xticklabels(labels=labels, rotation=90)
axs[1, 2].set_xticklabels(labels=labels, rotation=90)

axs[0, 0].set_ylabel('G (pS/uM^2)')
axs[1, 0].set_ylabel('G (pS/uM^2)')

fig
fig.savefig(figure_dir + 'conductance_gradients_horiz.png', dpi=300, format='png')
quit()
