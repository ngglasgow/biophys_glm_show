import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colors import bluered10


def open_vm(paths_object, channel, scale, trials=None):
    data_path = paths_object.biophys_output
    file_name = f'wns_{channel}_{scale}_v.dat'
    file_path = os.path.join(data_path, file_name)
    data = pd.read_csv(file_path, sep=r'\s+', header=None, usecols=trials)

    return data

def open_im(paths_object, channel, scale, trials=None):
    data_path = paths_object.biophys_output
    file_name = f'wns_{channel}_{scale}_i.dat'
    file_path = os.path.join(data_path, file_name)
    data = pd.read_csv(file_path, sep=r'\s+', header=None, usecols=trials)

    return data


class BiophysOutput:
    def __init__(self, paths_object, channel, scales, trials):
        v_df = pd.DataFrame()

        self.scales = list(set([str(float(file.split('_')[2])) for file in os.listdir(paths_object.biophys_output) if file.endswith('.dat')]))
        self.scales.sort()

        for scale in scales:
            scaler = str(float(scale))
            index = pd.DataFrame({'scale': scaler, 'trials': trials}, index=range(len(trials)))
            multiindex = pd.MultiIndex.from_frame(index)

            v = open_vm(paths_object, channel, scale, trials)
            v.columns = multiindex
            v_df = pd.concat([v_df, v], axis=1)

        i = open_im(paths_object, channel, scales[0], trials)
        
        dt = 0.1
        time = np.arange(0, len(v_df) * dt, dt)

        v_df.index = time
        i.index = time

        self.vm = v_df
        self.im = i
        
    

    def plot_vm(self, ax, scale, trials, xlim=(None, None)):
        start = xlim[0]
        stop = xlim[1]

        color_index = self.scales.index(scale)
        color = bluered10[color_index]

        vm = self.vm.loc[start:stop, (scale, trials)]
        ax.plot(vm, color=color, label=scale)

        return ax

    def plot_im(self, ax, trials, xlim=(None, None)):
        start = xlim[0]
        stop = xlim[1]

        im = self.im.loc[start:stop, trials]
        ax.plot(im, color='k', label=trials)

        return ax
