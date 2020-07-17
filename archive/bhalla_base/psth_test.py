import numpy as np
import quantities as pq
import pandas as pd
import elephant
import scipy
import matplotlib.pyplot as plt
% matplotlib

gaussian_kernel = elephant.statistics.make_kernel('GAU', 2*pq.ms, 1*pq.ms)
# plt.figure()
# plt.title(item + ' PSTH')



binned_spikes = channels[channel][scale]['binned_spiketrains']
mean_spikes = binned_spikes.apply(np.mean, axis=1)
psth = gaussian_kernel[1] * scipy.signal.fftconvolve(mean_spikes, gaussian_kernel[0])
psth = pd.DataFrame(psth, index=np.arange(550, 3110, 1), columns=['PSTH'])
channels[channel][scale]['psth'] = psth
# plt.plot(psth, label=item + '*' + key)
# plt.legend()
# plt.xlabel('Time (ms)')
# plt.ylabel('Firing Rate')
plt.plot(psth)
