import os
import pandas as pd
import scipy.io as sio
import set_paths
import conversions

home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, 'projects/biophys_glm_show')
bhalla_paths = set_paths.Paths(project_dir, 'bhalla')
alon_paths = set_paths.Paths(project_dir, 'alon')

# test conversions - these open and save things as csv for easy open
conversions.convert_bases(bhalla_paths)
conversions.convert_bases(alon_paths)

conversions.convert_filters(bhalla_paths, 'kA_1_CV.mat')
conversions.convert_filters(bhalla_paths, 'kA_13_CV.mat')

# exmaple slopes - this is a class and no saving involved
lambda_0 = conversions.example_weight_slopes(bhalla_paths 'kA_beta_curves_lambda_0.mat')
lambda_opt = conversions.example_weight_slopes(bhalla_slopes, 'kA_beta_curves_lambda_star.mat')
lambda_max = conversions.example_weight_slopes(bhalla_slopes, 'kA_beta_curves_lambda_largest.mat')

# slopes - this is a class
kA = conversions.beta_slopes(bhalla_paths, 'kA')
