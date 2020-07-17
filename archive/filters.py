# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matconv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

'''################### Set direcotories and open files #####################'''
bhalla_paths = matconv.set_paths.Paths('projects/biophys_glm_show', 'bhalla')
alon_paths = matconv.set_paths.Paths('projects/biophys_glm_show', 'alon')

bhalla_glm0 = sio.loadmat(os.path.join(bhalla_paths.survival, 'kA_1_CV.mat'))
bhalla_glm0['model_list'][0, 0]['channel_scalar']
bhalla_glm0['model_list'][0].shape
'''notes on how the matlab file is organized
matlab['model_list'][0, scale_index][data_type]
RIGHT HERE
'''

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
