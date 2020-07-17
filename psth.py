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

# set paths
home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')

bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')


class PSTH:
    '''
    '''
    def __init__(self, paths_object, data_type, trial_length, sd, fs):
        

        self.channels = self.indices.columns.levels[0].tolist()
        self.scales = self.indices.columns.levels[1].tolist()
        self.lambdas = self.indices.columns.levels[2].tolist()