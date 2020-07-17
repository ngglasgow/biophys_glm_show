import os
import pandas as pd
import scipy.io as sio
import scipy
import set_paths
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import conversions
from colors import bluered10

home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')

bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')

filters_0 = conversions.filters(bhalla_paths, 'kA', '0')

def plot_filters(ax, filter_object, type, scales):
    '''
    Plots the stimulus or history filters for a given filter object
    Parameters
    ----------
    ax: plt.axes
        ax on which to plot
    filter_object: conversions.filters() object
        the filter obj to plot
    type: str
        'stim', or 'hist'
    scales: str or list(str)
        'all' for all scales, or specific scales
    '''
    if type is 'stim':
        data = filter_object.stim
    
    elif type is 'hist':
        data = filter_object.hist
    
    if scales is 'all':
        scales = data.columns.tolist()
        scales.sort(reverse=True)

    for scale in scales:
        color_index = scales.index(scale)
        color = bluered10[color_index]
        ax.plot(data[scale], color=color, label=scale)
    
    return ax

