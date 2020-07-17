import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.io as sio
from matplotlib import cm
from matplotlib.colors import Normalize
import beta_slopes


'''################### Set direcotories and open files #####################'''
# set system type as macOS or linux; set project directory; set downstream dirs
# set all these directory variables first

home_dir = os.path.expanduser("~")
project_path = 'projects/biophys_glm_show'
project_dir = os.path.join(home_dir, project_path)
bhalla_dir = os.path.join(project_dir, 'data/beta_slopes', 'bhalla')
alon_dir = os.path.join(project_dir, 'data/beta_slopes', 'alon')
figure_dir = os.path.join(project_dir, 'analysis/figures/')
table_dir = os.path.join(project_dir, 'analysis/tables/')


# run loop for pulling out lambda values and making a dataframe
def channels_lambda_to_pandas(data_dir):
    channel_file_list = os.listdir(data_dir)
    channel_file_list.sort()

    model_slopes = pd.DataFrame()

    for file in channel_file_list:
        channel_beta_slopes = beta_slopes.beta_slopes_ml_to_np(data_dir, file)
        channel_slopes_lambda_star = channel_beta_slopes.get_optimal_values()
        model_slopes = pd.concat([model_slopes, channel_slopes_lambda_star], ignore_index=True)

    model_slopes.set_index(['model', 'channel'], inplace=True)

    return model_slopes

def get_channel_list(model_slopes):
    model_name = model_slopes.index[0][0]
    channel_list = model_slopes.loc[model_name].index.unique().tolist()

    return channel_list


# basic test
alon = beta_slopes.optimal_slopes(alon_dir)


plt.figure()
for channel in alon.channels:
    channel_lambda_star = alon.df.loc[alon.model, channel]
    channel_stimulus = channel_lambda_star['stimulus'].values
    plt.plot(channel_stimulus)
    