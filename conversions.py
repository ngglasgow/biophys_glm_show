import os
import pandas as pd
import scipy.io as sio
from scipy.signal import gaussian, fftconvolve
import set_paths
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import spiketrains


def convert_bases(paths_object, file='kA.mat'):
    '''
    Opens data from matlab, converts to pandas ad saves
    Parameters
    ----------
    paths_object: set_paths.Paths object
        The paths object from set_paths.py
    file: str
        The file to open in matlab, default=kA.mat
    '''
    matlab = sio.loadmat(os.path.join(paths_object.survival, file))

    kbas = pd.DataFrame(matlab['model_list'][0, 0]['kbas'][0, 0])
    hbas = pd.DataFrame(matlab['model_list'][0, 0]['hbas'][0, 0])

    kbas.to_csv(os.path.join(paths_object.bases, 'stim_bases.csv'), float_format='%8.4f', index=False)
    hbas.to_csv(os.path.join(paths_object.bases, 'hist_bases.csv'), float_format='%8.4f', index=False)

    return 'Stimulus and history objects for {} model saved.'.format(paths_object.model)


class bases:
    def __init__(self, paths_object):
        '''
        Opens the bases functions as an object
        Parameters
        ----------
        paths_object: set_paths.Paths object
            the path object from set_paths.py
        '''
        self.stim = pd.read_csv(os.path.join(paths_object.bases, 'stim_bases.csv'))
        self.hist = pd.read_csv(os.path.join(paths_object.bases, 'hist_bases.csv'))
        self.model = paths_object.model

    def __repr__(self):
        return 'Bases functions for stimulus and history for {} model.'.format(self.model)


def convert_filters(paths_object, file):
    '''
    Converts matlab data to csv for stimulus, history, and bias
    Parameters
    ----------
    paths_object: set_paths.Paths object
        the path object from set_paths.py
    file: str
        The file to open in matlab

    Returns
    -------
        saves stim, hist, and bias dataframes as csv
    '''
    lambda_index = file.split('_')[1]
    if lambda_index == '1':
        lambda_value = '0'
    else:
        lambda_value = '*'

    channel = file.split('_')[0]

    matlab = sio.loadmat(os.path.join(paths_object.survival, file))

    stim_df = pd.DataFrame()
    stim_weights_df = pd.DataFrame()
    hist_df = pd.DataFrame()
    hist_weights_df = pd.DataFrame()
    bias_df = pd.DataFrame()

    for i in range(10):
        model = matlab['model_list'][0, i]
        scaler = str(float(model['channel_scalar'][0, 0][0, 0]))
        stim = pd.DataFrame(model['k'][0, 0], columns=[scaler])
        stim_weights = pd.DataFrame(model['kw'][0, 0], columns=[scaler])
        hist = pd.DataFrame(model['h'][0, 0], columns=[scaler])
        hist_weights = pd.DataFrame(model['hw'][0, 0], columns=[scaler])
        bias = pd.DataFrame(model['dc'][0, 0], columns=[scaler])

        stim_df = pd.concat([stim_df, stim], axis=1)
        stim_weights_df = pd.concat([stim_weights_df, stim_weights], axis=1)
        hist_df = pd.concat([hist_df, hist], axis=1)
        hist_weights_df = pd.concat([hist_weights_df, hist_weights], axis=1)
        bias_df = pd.concat([bias_df, bias], axis=1)

    stim_df.to_csv(os.path.join(paths_object.survival, 'stim_{}_{}.csv'.format(channel, lambda_value)), float_format='%8.4f', index=False)
    stim_weights_df.to_csv(os.path.join(paths_object.survival, 'stim_weights_{}_{}.csv'.format(channel, lambda_value)), float_format='%8.4f', index=False)
    hist_df.to_csv(os.path.join(paths_object.survival, 'hist_{}_{}.csv'.format(channel, lambda_value)), float_format='%8.4f', index=False)
    hist_weights_df.to_csv(os.path.join(paths_object.survival, 'hist_weights_{}_{}.csv'.format(channel, lambda_value)), float_format='%8.4f', index=False)
    bias_df.to_csv(os.path.join(paths_object.survival, 'bias_{}_{}.csv'.format(channel, lambda_value)), float_format='%8.4f', index=False)

    return 'Stimulus and history filters and bias terms saved for {} saved.'.format(file)


class filters:
    def __init__(self, paths_object, channel, lambda_value):
        '''
        Opens the glm filters of a given channel and lambda value and stores in object.
        Parameters
        ----------
        paths_object: set_paths.Paths object
            the path object from set_paths.py
        channel: str
            the channel code of interest
        lambda_value: str
            currently only '0' or '*'
        '''
        self.stim = pd.read_csv(os.path.join(paths_object.survival, 'stim_{}_{}.csv'.format(channel, lambda_value)))
        self.stim_weights = pd.read_csv(os.path.join(paths_object.survival, 'stim_weights_{}_{}.csv'.format(channel, lambda_value)))
        self.hist = pd.read_csv(os.path.join(paths_object.survival, 'hist_{}_{}.csv'.format(channel, lambda_value)))
        self.hist_weights = pd.read_csv(os.path.join(paths_object.survival, 'hist_weights_{}_{}.csv'.format(channel, lambda_value)))
        self.bias = pd.read_csv(os.path.join(paths_object.survival, 'bias_{}_{}.csv'.format(channel, lambda_value)))
        self.channel = channel
        self.lambda_value = lambda_value

    def __repr__(self):
        return 'DataFrames for glm filters of channel {} and lambda = {}.'.format(self.channel, self.lambda_value)


class example_weight_slopes:
    '''
    Class for opening the example slopes into dataframes for easy plotting
    '''
    def __init__(self, paths_object, file):
        weights = sio.loadmat(os.path.join(paths_object.example_slopes, file))
        self.conductances = weights['g'][0]
        self.lambda_value = weights['lambda'][0]
        self.stim_weights = pd.DataFrame(weights['KW'].T, index=self.conductances)
        self.hist_weights = pd.DataFrame(weights['HW'].T, index=self.conductances)
        self.bias_weights = weights['DC'][0]

 
    def __repr__(self):
        return 'weights for lambda={}'.format(self.lambda_value[0])


class beta_slopes:
    def __init__(self, paths_object, channel):
        file_list = os.listdir(paths_object.slopes)
        for name in file_list:
            if channel in name:
                file = name

        matlab = sio.loadmat(os.path.join(paths_object.slopes, file))
        beta_slopes = matlab['beta_slopes']
        self.stim_slopes = beta_slopes[:10]
        self.hist_slopes = beta_slopes[11:]
        self.bias_slopes = beta_slopes[10]
        self.lambda_plot = matlab['lambda_plot']
        self.lambda_id = matlab['lambd_id'][0][0] - 1
        self.channel = file.split('_')[0]
        self.ll_train = matlab['log_likelihood_train']
        self.ll_test = matlab['log_likelihood_test']
        self.model = paths_object.model

    
    def get_optimal_values(self):
        # return the optimal beta slope values
        lambda_stim = self.stim_slopes.T[self.lambda_id]
        lambda_hist = self.hist_slopes.T[self.lambda_id]
        lambda_bias = self.bias_slopes.T[self.lambda_id]
        lambda_value = self.lambda_plot[0][self.lambda_id]
        lambda_star = pd.DataFrame({'stimulus': lambda_stim,
                                    'history': lambda_hist,
                                    'bias': lambda_bias,
                                    'channel': self.channel,
                                    'model': self.model,
                                    'lambda index': self.lambda_id,
                                    'lambda value': lambda_value})
        return lambda_star


    def get_lambda_values(self, lambda_index):
        # return the beta slopes of a supplied lambda_index
        lambda_stim = self.stim_slopes.T[lambda_index]
        lambda_hist = self.hist_slopes.T[lambda_index]
        lambda_bias = self.bias_slopes.T[lambda_index]
        lambda_value = self.lambda_plot[0][lambda_index]
        lambda_values = pd.DataFrame({'stimulus': lambda_stim,
                                    'history': lambda_hist,
                                    'bias': lambda_bias,
                                    'channel': self.channel,
                                    'model': self.model,
                                    'lambda index': lambda_index,
                                    'lambda value': lambda_value})

        return lambda_values
        
    def __repr__(self):
        return 'Absolute summed change in slopes for {} '.format(self.channel)


class sim_slopes:
    def __init__(self, paths_object):
        file = 'sim1_beta_slopes.mat'
        matlab = sio.loadmat(os.path.join(paths_object.sim_slopes, file))
        beta_slopes = matlab['beta_slopes']
        self.stim_slopes = beta_slopes[:10]
        self.hist_slopes = beta_slopes[11:]
        self.bias_slopes = beta_slopes[10]
        self.lambda_plot = matlab['lambda_plot']
        self.lambda_id = matlab['lambd_id'][0][0] - 1
        self.ll_train = matlab['log_likelihood_train']
        self.ll_test = matlab['log_likelihood_test']
        self.model = paths_object.model

    
    def get_optimal_values(self):
        # return the optimal beta slope values
        lambda_stim = self.stim_slopes.T[self.lambda_id]
        lambda_hist = self.hist_slopes.T[self.lambda_id]
        lambda_bias = self.bias_slopes.T[self.lambda_id]
        lambda_value = self.lambda_plot[0][self.lambda_id]
        lambda_star = pd.DataFrame({'stimulus': lambda_stim,
                                    'history': lambda_hist,
                                    'bias': lambda_bias,
                                    'model': self.model,
                                    'lambda index': self.lambda_id,
                                    'lambda value': lambda_value})
        return lambda_star


    def get_lambda_values(self, lambda_index):
        # return the beta slopes of a supplied lambda_index
        lambda_stim = self.stim_slopes.T[lambda_index]
        lambda_hist = self.hist_slopes.T[lambda_index]
        lambda_bias = self.bias_slopes.T[lambda_index]
        lambda_value = self.lambda_plot[0][lambda_index]
        lambda_values = pd.DataFrame({'stimulus': lambda_stim,
                                    'history': lambda_hist,
                                    'bias': lambda_bias,
                                    'model': self.model,
                                    'lambda index': lambda_index,
                                    'lambda value': lambda_value})

        return lambda_values
        

    def __repr__(self):
        return 'Absolute summed change in slopes for simulated GLM model'


class optimal_slopes:
    def __init__(self, paths_object):
        # set channel list
        channel_df = pd.read_csv(os.path.join(paths_object.slopes, 'channels.txt'), index_col=[0], header=None, dtype=str)
        self.channels_files = channel_df.loc['filenames'].tolist()
        self.channels_proper = channel_df.loc['proper'].tolist()

        self.stim = pd.DataFrame()
        self.hist = pd.DataFrame()
        bias = []

        for channel in self.channels_files[::-1]:
            slopes = beta_slopes(paths_object, channel)
            opt = slopes.get_optimal_values()
            channel_stim = pd.DataFrame(opt['stimulus'])
            channel_stim.columns = [channel]
            self.stim = pd.concat([self.stim, channel_stim], axis=1)
            channel_hist = pd.DataFrame(opt['history'])
            channel_hist.columns = [channel]
            self.hist = pd.concat([self.hist, channel_hist], axis=1)
            bias.append(opt['bias'][0])

        self.bias = pd.DataFrame(bias, index=self.channels_files)

    def plot_slopes(self, ax, data, s=None, marker=None, cmap=None, vmin=None, vmax=None):
        '''
        Plots a scatter with the saturation of the markers equal the the value supplied.
        Parmeters
        ---------
        ax: plt.axes
            the axes on which to supply the scatter
        data: np.array
            an array of data corresponding to slopes with amount of saturation
        '''
        xt = np.arange(1, 11, 1)
        yt = np.arange(0, len(self.channels_proper), 1)
        xst, yst = np.meshgrid(xt, yt)

        norm = Normalize(vmin, vmax)

        if data == 'stim':
            c = self.stim.values.T
            x_title = r'Stimulus Coefficient Index ($\beta^K_i$)'
        elif data == 'hist':
            c = self.hist.values.T
            x_title = r'History Coefficient Index ($\beta^H_i$)'

        channel_names = ['$\\mathregular{{{}}}$'.format(channel) for channel in self.channels_proper]

        ax.scatter(xst, yst, s=s, marker=marker, c=c, cmap=cmap, norm=norm)
        ax.set_ylim(-1, len(channel_names))
        ax.set_yticks(np.arange(0, len(channel_names), 1))
        ax.set_yticklabels(channel_names[::-1])
        ax.set_xticks(xt)
        ax.set_xlabel(x_title)

        return ax

    
    def plot_colorbar(self, fig, axs, cmap=None, vmin=None, vmax=None, fraction=0.10):
        norm = Normalize(vmin, vmax)

        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, fraction=fraction)

        return cb


def convert_coherences(paths_object):
    '''
    Takes a directory of coherence measures from matlab and converts to a csv with every
    channel, scale, and lambda value (no lambda or opitmal lambda)
    
    Parameters:
    -----------
    paths_object: set_paths.Paths object
        the path object from set_paths.py

    Returns:
    none: saves coherence.csv in paths_object.coherence
    '''
    coherence_files = [file for file in os.listdir(paths_object.coherence) if file.endswith('mat')]
    coherence_files.sort()

    coherence_df = pd.DataFrame()

    for file in coherence_files:
        name_split = file.split('_')
        channel = name_split[1]
        lambda_id = name_split[5]
        scaler = str(float(name_split[2]))

        if lambda_id is '1':
            lambda_value = 'none'
        else:
            lambda_value = 'optimal'

        index_df = pd.DataFrame({'channel': channel, 'scale': scaler, 'lambda': lambda_value}, index=[1])

        file_path = os.path.join(paths_object.coherence, file)

        matlab = sio.loadmat(file_path)
        coherence = matlab['cohr'].reshape(-1)
        frequency = matlab['f'].reshape(-1)

        multiindex = pd.MultiIndex.from_frame(index_df)
        coherence = pd.DataFrame(coherence, index=frequency, columns=multiindex)
        coherence_df = pd.concat([coherence_df, coherence], axis=1)

    coherence_df.sort_index(axis=1, level='scale', inplace=True)
    coherence_df.to_csv(os.path.join(paths_object.coherence, 'coherence.csv'), float_format='%8.3f')
    channels = coherence_df.columns.levels[0].tolist()
    scales = coherence_df.columns.levels[1].tolist()
    lambdas = coherence_df.columns.levels[2].tolist()

    return(print(f'Cohernece DataFrame saved as coherence.csv saved in\n{paths_object.coherence}\nchannels: {channels}\nscales: {scales}\nlambdas: {lambdas}'))


def convert_reconstructions(paths_object):
    '''
    Takes a directory of reconstructed stimuli and converts to a csv with every
    channel, scale, and lambda value (no lambda or opitmal lambda)

    Assumes that the stimuli are the same for every reconstruciton
        33 stimuli taken at random from the 100 stimuli, 67 of which were used for training
        Stimuli were tested by comparing their means and sums, which are identical across all
    
    Parameters:
    -----------
    paths_object: set_paths.Paths object
        the path object from set_paths.py

    Returns:
    none: saves reconstructions.csv AND stimulus.csv in paths_object.reconstructions
    '''
    reconstruction_files = [file for file in os.listdir(paths_object.reconstructed) if 'reconstructed' in file]
    reconstruction_files.sort()

    reconstruction_df = pd.DataFrame()

    for file in reconstruction_files:
        name_split = file.split('_')
        channel = name_split[1]
        lambda_id = name_split[5]
        scaler = str(float(name_split[2]))

        if lambda_id is '1':
            lambda_value = 'none'
        else:
            lambda_value = 'optimal'

        index_df = pd.DataFrame({'channel': channel, 'scale': scaler, 'lambda': lambda_value}, index=[1])

        file_path = os.path.join(paths_object.reconstructed, file)

        reconstruction = pd.read_csv(os.path.join(paths_object.reconstructed, file_path), header=None).values[0]

        multiindex = pd.MultiIndex.from_frame(index_df)
        reconstruction = pd.DataFrame(reconstruction, columns=multiindex)
        reconstruction_df = pd.concat([reconstruction_df, reconstruction], axis=1)

    reconstruction_df.sort_index(axis=1, level='scale', inplace=True)
    reconstruction_df.to_csv(os.path.join(paths_object.reconstructed, 'reconstruction.csv'), float_format='%6.4f')

    channels = reconstruction_df.columns.levels[0].tolist()
    scales = reconstruction_df.columns.levels[1].tolist()
    lambdas = reconstruction_df.columns.levels[2].tolist()

    stim_path = os.path.join(paths_object.reconstructed, 'spiketimes_kA_1_lambda_index_1_stims.csv')
    test_stimuli = pd.read_csv(stim_path, header=None).T
    test_stimuli.to_csv(os.path.join(paths_object.reconstructed, 'test_stimuli.csv'), float_format='%6.4f')

    return(print(f'Reconstructed stimulus DataFrame saved as reconstruction.csv and actual stimuli as test_stimuli.csv in{paths_object.reconstructed}\nFor the following:\n\tchannels: {channels}\n\tscales: {scales}\n\tlambdas: {lambdas}'))


def convert_glm_spiketrains(paths_object):
    '''
    Takes a directory of glm simulated spiketrains and converts to a csv with every
    channel, scale, and lambda value (no lambda or opitmal lambda)

    The matlab files have a list of spike times and indices corresponding to the trial.
    This will convert to 2 csv files, one for trial indices and one for spike times.
    
    Parameters:
    -----------
    paths_object: set_paths.Paths object
        the path object from set_paths.py

    Returns:
    none: saves spikeindices.csv AND spiketimes.csv in paths_object.glm_sims
    '''
    spiketimes_files = [file for file in os.listdir(paths_object.glm_sims) if file.endswith('.mat')]
    spiketimes_files.sort()

    spiketimes_df = pd.DataFrame()
    indices_df = pd.DataFrame()

    for file in spiketimes_files:
        name_split = file.split('_')
        channel = name_split[2]
        lambda_id = name_split[6].split('.')[0]
        scaler = str(float(name_split[3]))

        if lambda_id is '1':
            lambda_value = 'none'
        else:
            lambda_value = 'optimal'

        index_df = pd.DataFrame({'channel': channel, 'scale': scaler, 'lambda': lambda_value}, index=[1])
        multiindex = pd.MultiIndex.from_frame(index_df)

        file_path = os.path.join(paths_object.glm_sims, file)

        matlab = sio.loadmat(file_path)
        data = matlab['MC2']

        spiketimes = data['spikeTimes'][0, 0].reshape(-1)
        indices = data['spikeIndices'][0, 0].reshape(-1)

        spiketimes = pd.DataFrame(spiketimes, columns=multiindex)
        indices = pd.DataFrame(indices, columns=multiindex)
        
        spiketimes_df = pd.concat([spiketimes_df, spiketimes], axis=1)
        indices_df = pd.concat([indices_df, indices], axis=1)
    
    spiketimes_df.sort_index(axis=1, level='scale', inplace=True)
    indices_df.sort_index(axis=1, level='scale', inplace=True)

    spiketimes_df.to_csv(os.path.join(paths_object.glm_sims, 'spiketimes.csv'), float_format='%6.0f')
    indices_df.to_csv(os.path.join(paths_object.glm_sims, 'spikeindices.csv'), float_format='%3.0f')

    channels = spiketimes_df.columns.levels[0].tolist()
    scales = spiketimes_df.columns.levels[1].tolist()
    lambdas = spiketimes_df.columns.levels[2].tolist()

    return(print(f'spiketimes and spike indicies DataFrames saved as spiketimes.csv and spikeindices.csv saved in\n{paths_object.glm_sims}\nchannels: {channels}\nscales: {scales}\nlambdas: {lambdas}'))


def convert_biophys_spiketrains(paths_object):
    '''
    Takes a directory of biophys simulated spiketrains and converts to a csv with every
    channel, scale, and lambda value (there is only no lambda for biophys)

    The csv files are a for a channel and scale with a sparse matrix of spiketimes
    where each column is a trial.
    This will convert the matrix into a list for that channel and scale, and make a new list
    with corresponding trial indices. Will return 2 csv files, one for trial indices and one
    for spike times.
    
    Parameters:
    -----------
    paths_object: set_paths.Paths object
        the path object from set_paths.py

    Returns:
    none: saves spikeindices.csv AND spiketimes.csv in paths_object.biophys_output
    '''
    data_path = paths_object.biophys_output
    spiketimes_files = [file for file in os.listdir(data_path) if file.startswith('spiketimes_')]
    spiketimes_files.sort()

    spiketimes_df = pd.DataFrame()
    indices_df = pd.DataFrame()

    for file in spiketimes_files:
        name_split = file.split('_')
        channel = name_split[1]
        scaler = str(float(name_split[2][:-4]))
        lambda_value = 'none'
        
        index_df = pd.DataFrame({'channel': channel, 'scale': scaler, 'lambda': lambda_value}, index=[1])
        multiindex = pd.MultiIndex.from_frame(index_df)

        file_path = os.path.join(data_path, file)

        spikes = pd.read_csv(file_path)
        spikes_flat = spikes.values.flatten('F')
        index_flat = np.repeat(range(len(spikes.columns)), len(spikes.index))

        spiketimes = pd.DataFrame(spikes_flat, columns=multiindex)
        indices = pd.DataFrame(index_flat, columns=multiindex)
        
        spiketimes_df = pd.concat([spiketimes_df, spiketimes], axis=1)
        indices_df = pd.concat([indices_df, indices], axis=1)
    
    spiketimes_df.sort_index(axis=1, level='scale', inplace=True)
    indices_df.sort_index(axis=1, level='scale', inplace=True)

    spiketimes_df.to_csv(os.path.join(data_path, 'spiketimes.csv'), float_format='%6.1f')
    indices_df.to_csv(os.path.join(data_path, 'spikeindices.csv'), float_format='%3.0f')

    channels = spiketimes_df.columns.levels[0].tolist()
    scales = spiketimes_df.columns.levels[1].tolist()
    lambdas = spiketimes_df.columns.levels[2].tolist()

    return(print(f'spiketimes and spike indicies DataFrames saved as spiketimes.csv and spikeindices.csv saved in\n{data_path}\nchannels: {channels}\nscales: {scales}\nlambdas: {lambdas}'))


def create_psths(paths_object, data_type, trial_length, sd, fs):
    '''
    Creates a PSTH df of all channels, scales, and lambda values for a given data type.
    Parameters
    ----------
    paths_object: set_paths.Paths object
        the path object from set_paths.py
    data_type: str
        'biophys' or 'glm'
        str describing what kind of data it is to determine the appropriate source
    trial_length: int
        the index of the length of the actual trial
        default = 3200 ms
    sd: int, float
        standard deviation to use for the gaussian window for the PSTH kernel
        default = 2 ms
    fs: int
        sampling frequency in Hz
        default = 1000 Hz
    Returns
    -------
    psths.csv
        the PSTH df in the dir where the data_type points
    '''
    data = spiketrains.SpikeTrains(paths_object, data_type)
    

    width = 3 * sd * 2 + 1
    cut = width // 2
    raw_kernel = gaussian(width, std=sd)
    norm_kernel = raw_kernel / raw_kernel.sum()

    psth_df = pd.DataFrame()

    for channel in data.channels:
        channel_data = data.spiketimes.loc[:, (channel, slice(None), slice(None))]
        scales_set = set(channel_data.columns.remove_unused_levels().levels[1].tolist())
        for scale in scales_set:
            for value in data.lambdas:
                index_df = pd.DataFrame({'channel': channel, 'scale': scale, 'lambda': value}, index=[1])
                multiindex = pd.MultiIndex.from_frame(index_df)

                spiketimes = data.spiketimes.loc[:, (channel, scale, value)].dropna()
                indices = data.indices.loc[:, (channel, scale, value)].dropna()
                if len(indices) is 0:
                    continue
                indices_spikes = np.array((indices, spiketimes))
                trial_counter = np.arange(1, max(indices) + 1, 1)
                spikes_list = [indices_spikes[1][indices_spikes[0] == trial] for trial in trial_counter]

                binned_spikes = pd.DataFrame()
                for spike_index in spikes_list:
                    bins = np.zeros(trial_length)
                    bins[spike_index.astype(int)] = 1
                    binned_spikes = pd.concat([binned_spikes, pd.DataFrame(bins)], axis=1)

                mean_binned_spikes = binned_spikes.mean(axis=1)

                psth = fftconvolve(mean_binned_spikes, norm_kernel) * fs
                psth = pd.DataFrame(psth, columns=multiindex)

                psth_df = pd.concat([psth_df, psth], axis=1)
    
    psth_df.sort_index(axis=1, level='scale', inplace=True)
    psth_cut = psth_df.loc[cut:]
    psth_cut.index = range(len(psth_cut))
    psth_cut.to_csv(os.path.join(data.data_path, 'psths.csv'), float_format='%8.3f')

    channels = psth_df.columns.levels[0].tolist()
    scales = psth_df.columns.levels[1].tolist()
    lambdas = psth_df.columns.levels[2].tolist()

    return(print(f'PSTH DataFrames saved as psths.csv saved in\n{data.data_path}\nchannels: {channels}\nscales: {scales}\nlambdas: {lambdas}'))
