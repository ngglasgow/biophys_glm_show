import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import set_paths
import conversions
import seaborn as sns
import coherence
from matplotlib.ticker import MultipleLocator

# set paths
home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')

bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')

# open all data
bhalla_cohr = coherence.Coherence(bhalla_paths)

#fig, axs = bhalla_cohr.plot_coherence_bands()
#fig, axs = bhalla_cohr.plot_coherence_bands(channels=['kA', 'kca3'])
fig, axs = plt.subplots()
axs = bhalla_cohr.plot_channel_coherence(axs, 'kA', ['0.5', '1.0', '1.5'])
fig_path = os.path.join(bhalla_paths.figures, 'b_cohere_new.png')
fig.savefig(fig_path, format='png', dpi=300)

