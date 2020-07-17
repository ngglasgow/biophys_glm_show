# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.io as sio
from matplotlib import cm
from matplotlib.colors import Normalize

'''################### Set direcotories and open files #####################'''

'''###################### convert basis from matlab #########################'''
basis_file_list = [os.path.join(basis_dir, file) for file in os.listdir(basis_dir) if 'bhalla' in file and file.endswith('mat')]
basis_file_list.sort()

stim = sio.loadmat(basis_file_list[1])
stim_basis = stim['kbas']
stim_time = stim['kt'].reshape(-1)
stim_df = pd.DataFrame(stim_basis, index=stim_time)
stim_peaks = stim_df.idxmax()

hist = sio.loadmat(basis_file_list[0])
hist_basis = hist['hbas']
hist_time = hist['ht'].reshape(-1)
hist_df = pd.DataFrame(hist_basis, index=hist_time)
hist_peaks = hist_df.idxmax()

'''################### grab GLM filters #################'''



'''#################### plotting bases off of weights to make glm ############'''
fig, axs = plt.subplots(3, 2, sharex=True)
axs[0, 0].plot(stim_df)
axs[0, 0].set_ylabel('Basis')

axs[1, 0].scatter(stim_peaks, np.ones_like(stim_peaks))
axs[1, 0].set_ylabel('Weights')

axs[2, 0].plot(bhalla_filters.stim_control)
axs[2, 0].set_ylabel('Filter')

axs[0, 0].set_title('Stimulus Bhalla')


#fig, axs = plt.subplots(3, 1, sharex=True)
axs[0, 1].plot(hist_df)
#axs[0, 1].set_ylabel('Basis')

axs[1, 1].scatter(hist_peaks, np.ones_like(hist_peaks))
#axs[1, 1].set_ylabel('Weights')

axs[2, 1].plot(bhalla_filters.hist_control)
axs[2, 1].set_ylim(-15, 8)
#axs[2, 1].set_ylabel('Filter')

axs[0, 1].set_title('History Bhalla')

fig.savefig(os.path.join(figure_dir, 'basis_eights.png'), dpi=300, format='png')
