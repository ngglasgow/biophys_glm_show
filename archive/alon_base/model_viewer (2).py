import os

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
