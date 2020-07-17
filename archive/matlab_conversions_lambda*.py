# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.io as sio

sns.set()
%matplotlib inline
%matplotlib
'''################### Set direcotories and open files #####################'''
# set system type as macOS or linux; set project directory; set downstream dirs
# set all these directory variables first

home_dir = os.path.expanduser("~")
project_dir = home_dir + '/ngglasgow@gmail.com/Data_Urban/NEURON/Bhalla_par_scaling/'
glm_dir = project_dir + 'scaled_wns_glm_survival/'
figure_dir = project_dir + 'analysis/figures/'
table_dir = project_dir + 'analysis/tables/'



# grab file names from my output directory and sort by name
file_list = os.listdir(glm_dir)
file_list.sort()
file_list

matlab = sio.loadmat(glm_dir + 'kA.mat')

# open stimulus filter
structure = matlab['model_list'][0]
n_scales = len(structure)

stim_time_index = pd.DataFrame({'parameter': 'kt', 'scale': 1.0}, index=range(1))
stim_time_index = pd.MultiIndex.from_frame(stim_time_index)
stim_time = pd.DataFrame(matlab['model_list'][0, 0]['kt'][0][0], columns=stim_time_index)

hist_time_index = pd.DataFrame({'parameter': 'ht', 'scale': 1.0}, index=range(1))
hist_time_index = pd.MultiIndex.from_frame(hist_time_index)
hist_time = pd.DataFrame(matlab['model_list'][0, 0]['ht'][0][0], columns=hist_time_index)

data_list = ['k', 'h', 'dc']
glm_df = pd.DataFrame()
for data in data_list:
    data_df = pd.DataFrame()
    labels_df = pd.DataFrame()
    for i in range(n_scales):
        parameter = matlab['model_list'][0, i][data][0][0]
        parameter = pd.DataFrame(parameter)
        scale = matlab['model_list'][0, i]['channel_scalar'][0][0][0][0]
        label = pd.DataFrame({'parameter': data, 'scale': scale}, index=range(1))
        labels_df = pd.concat([labels_df, label])
        data_df = pd.concat([data_df, parameter], axis=1)
    data_index = pd.MultiIndex.from_frame(labels_df)
    data_df.columns = data_index
    glm_df = pd.concat([glm_df, data_df], axis=1)
glm_df = pd.concat([stim_time, hist_time, glm_df], axis=1)

# save transposed table so that it is easier to open and index with
glm_df.T.to_csv(table_dir + 'kA_glm_lambda*.csv', float_format='%8.4f')
