import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import set_paths
import conversions
import seaborn as sns
import reconstructions
from matplotlib.ticker import MultipleLocator

# set paths
home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')

bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')

# open all data
bhalla_recon = reconstructions.Reconstructions(bhalla_paths)

#fig, axs = bhalla_cohr.plot_coherence_bands()
#fig, axs = bhalla_cohr.plot_coherence_bands(channels=['kA', 'kca3'])
fig, ax = plt.subplots()
recon = bhalla_recon.plot_channel_reconstructions(ax=ax, channel='kA', scales=['0.5', '1.0', '1.5'])

ax.set_axis_off()
