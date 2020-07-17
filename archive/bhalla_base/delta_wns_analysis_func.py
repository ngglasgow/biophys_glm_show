#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
*********************************************************************************
24 Jan 2018
Ran first batch of delta_wns files through Bhalla and Bower 1993 model.

Simulations ran from bash script: ./bhalla_wns_batch.sh
It uses 100 trials of WNS generated with 600 DC amplitude with 15% SD in noise
and correlation coefficient between WNS sweeps of 0.8. In control model, this
yields firing rate ~35 Hz and a spike train correlation coefficient of ~0.17.

The run took about 6 hours to complete with multithreaded simulations. Will need
to test single-threaded simulations on subset to check veracity of simulations.

Simulation files of the form "wns_v_'conductance'_'delta'.dat"

***********
25 Jan 2018
Analysis of files. Most interested in firing rates, spike-triggered averages,
PSTH, CV ISI.

Made major change to I/O reading. I changed if/else statements to save into
dictionaries instead of into lists. This way, I don't have to make a bunch of
list variables and call all of them all the time. I make one list that will be
unique to the set of analysis I'm doing (g_list), where I define all the channel
types (conductances, g) as strings in a list.

Then I create two dictionaries by using the channel type strings as keys, with
values of empty lists. It's a very simple in line for loop to do this:
    g_dict = {item:[] for item in g_list}

Dict one is a list of the simulation delta level in that
simulation file (g_dict_d). This dictionary has values of lists, with 'delta_g ='
is the first value, and the string delta_g level as the second value.

The second dictionary is of lists of voltage (vm) df's simulated in response to
WNS injection sweeps.

For now, I have repeat control (delta_g = 1) for each channel type. This isn't
the most memory/data efficient way, but the files aren't that large, so fine. If
this ends up being a problem, I can add a second if/else statement to my .dat
statement.

NEW
Now I think I have a better formulation for organizing all the data and adding to
the same dict new data sets, it mibht be unwieldy. but we will see. I think g_dict2
might have the right organization.

***********
26 Jan 2016
The files here may be too memory intensive to be useful to do how I was initially
analysing. May be that it'smore efficient to def functions that point to certain
paths and files and I can do them on demand, do the analysis, get the output,
then kill the file again. Easy enoguh to call that function again if need be.
Try to implement tomorrow.

Alternatively, I could just write separate scripts for each channel type and run
each and then that would b fine. Could make them executable and it would work ok
all the analysies and outpouts were automatic.

*********************************************************************************
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neo.core import SpikeTrain, AnalogSignal
import elephant
import quantities as pq
from elephant.sta import spike_triggered_average
import scipy
import os
from neuron import h

#%% Read in sim dir. Need to navigate to dir where they exist before executing.
os.chdir('/Users/nate/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/simulations_8015x50')
dir_list = os.listdir(os.getcwd())
dir_list.sort()
print dir_list

#%%
x = 700

#%% Set up lists to import data into. Create time place holder in case is needed.

t = np.arange(0,3200,0.1)
tm = pd.DataFrame(t)

g_list = ['car', 'cah', 'bk', 'kslow', 'sk', 'na', 'iH', 'iA']
#g_dict = {item:[] for item in g_list}
#g_dict2 = {item:[] for item in g_list}
channels = {item:{} for item in g_list}


for k in range(len(dir_list)):
    if '.dat' in dir_list[k]:
        tmpv = pd.read_csv(dir_list[k],sep='\s+',header=None)
        tmpv = tmpv.iloc[:32000]
        tmpg = dir_list[k].split('_')[2]
        tmpd = dir_list[k].split('_')[3].split('.dat')[0]
        #g_dict[tmpg].append([tmpg,'delta_g',tmpd,tmpv])
        #g_dict2[tmpg].append([[tmpg,'delta_g =',tmpd,dir_list[k]],['Vm',tmpv]])
        channels[tmpg][tmpd] = {'filename':dir_list[k],'Vm':tmpv}
        tmpv = None
    else:
        wns_i = pd.read_csv(dir_list[k],sep='\t',header=None)
        wns_i = wns_i.iloc[:32000]
        wns_it = pd.concat([tm,wns_i],axis=1)

#%% Create spike time lists and neo.SpikeTrain lists from vm

for item in g_list:
    for key in channels[item]:
        tvmdf = channels[item][key]['Vm']
        tbst = pd.DataFrame()
        for i in range(50):
            tvm = h.Vector(tvmdf.iloc[:, i].values)
            tst = h.Vector()
            tst.spikebin(tvm, 0)
            tstdf = pd.DataFrame(tst.to_python())
            tbst = pd.concat([tbst, tstdf], axis=1)
            tst = None
            tstdf = None
            tvm = None
        tbst = pd.concat([tm, tbst], axis=1)
        tst_list = []
        tnst500_list = []
        tst500_list = []
        tstdet_list = []
        for i in range(1, 51):
            tst = tbst.iloc[:, [0, i]]
            tst500 = tst.iloc[6500:]
            tstdet = tst.iloc[15000:18000]
            tst = tst.iloc[:, 0][tst.iloc[:, 1] > 0]
            tst_list.append(tst)
            tst500 = tst500.iloc[:, 0][tst500.iloc[:, 1] > 0]
            tst500_list.append(tst500)
            tstdet = tstdet.iloc[:, 0][tstdet.iloc[:, 1] > 0]
            tstdet_list.append(tstdet)
            tnst500 = SpikeTrain(times=tst500.values*pq.ms,
                                 t_start=650., t_stop=3200)
            tnst500_list.append(tnst500)
            tst = None
            tst500 = None
            tnst500 = None
        channels[item][key]['spikeTimes'] = tst_list
        channels[item][key]['spikeTimes500'] = tst500_list
        channels[item][key]['neoSpkTrain500'] = tnst500_list
        channels[item][key]['spikeDetail'] = tstdet_list
        tbst = None
        tst_list = None
        tst500_list = None
        tnst500_list = None
        tvmdf = None
        tstdet = None
        tstdet_list = None

#%% Pull out Vm detail of 1500 to 1800 ms
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
for item in g_list:
    plt.figure()
    plt.title('Vm Traces for '+item)
    plt.xlim(1490,1810)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Voltage (mV)')
    for key in channels[item]:
        tvm = channels[item][key]['Vm'].iloc[15000:18000, 0]
        tvm.index = np.arange(1500,1800,0.1)
        plt.plot(tvm)
    #plt.show()
    plt.savefig(item+'_Vm_detail.png',dpi=600,format='png')
#%% Make raster plots for comparing all 4 delta values
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
colors = ['blue','orange','green','red']
for item in g_list:
    plt.figure()
    plt.title('Spike Times for '+item)
    plt.axis('tight')
    plt.xlim(0,3200)
    plt.xlabel('Time (ms)',fontsize=16)
    plt.ylim(0,81)
    plt.ylabel('Trial Number', fontsize=16)
    for i, key in zip(range(4), channels[item]):
        for k in range(20):
            t = channels[item][key]['spikeTimes'][k]
            plt.plot(t,((i*20)+(k+1))*np.ones_like(t),'|',markersize=2
                        ,color=colors[i])
    #plt.show()
    plt.savefig(item+'_rasters.png',dpi=600,format='png')
    t = None
#%% Make raster plots for comparing all 4 delta values in detail 1500-1800 ms
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
colors = ['blue','orange','green','red']
for item in g_list:
    plt.figure()
    plt.title('Spike Times for '+item)
    plt.xlabel('Time (ms)',fontsize=16)
    plt.ylabel('Trial Number', fontsize=16)
    plt.ylim(0,81)
    for i, key in zip(range(4), channels[item]):
        for k in range(20):
            t = channels[item][key]['spikeDetail'][k]
            plt.plot(t,((i*20)+(k+1))*np.ones_like(t),'|',markersize=2
                        ,color=colors[i])
    #plt.show()
    plt.savefig(item+'_rasters_detail.png',dpi=600,format='png')
    t = None

#%% Make raster plots for comparing each channel and each delta, save
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
for item in g_list:
    for key in channels[item]:
        plt.figure()
        plt.title('Spike Times of '+item+'*'+key)
        plt.axis('tight')
        plt.xlim(0,3200)
        plt.xlabel('Time (ms)',fontsize=16)
        plt.ylim(0,51)
        plt.ylabel('Trial Number', fontsize=16)
        for k in range(50):
            t = channels[item][key]['spikeTimes'][k]
            plt.plot(t,(k+1)*np.ones_like(t),'b|',markersize=2)
    #    plt.show()
        plt.savefig(item+'_'+key+'_raster.png',dpi=600,format='png')
        t = None



#%%
"""
        if 'nan' in cv_list:
            print 'nan'
        else:
            plt.figure()
            plt.hist(cv_list)
            plt.xlabel('CV', fontsize=16)
            plt.ylabel('count', fontsize=16)
            plt.show()
            plt.savefig(item+'*'+key+'_CVisi.png',dpi=600,format='png')
"""
#%%
"""
28Sep2017
NOTE: Work on conversion of discrete spike time lists to binary spike counts.
Use this conversion for PSTH and for spike-time correlations.
"""
##ADDED bstc_df to cut the first 500 ms of data out for correlations
###conversion of discrete spike times to binary counts

from elephant.conversion import BinnedSpikeTrain
bins = 1.0 #define the bin size here in ms
for item in g_list:
    for key in channels[item]:
        bst_list = BinnedSpikeTrain(channels[item][key]['neoSpkTrain500'],
                                    binsize = bins*pq.ms,
                                    t_start = 650.0*pq.ms,
                                    t_stop = 3160*pq.ms)


        bst_arr = bst_list.to_array()           #export binned spike times to an array
        bst_df = pd.DataFrame(bst_arr).T        #turn into a df and transpose (.T)
   #     bst_df.index = np.arange(550,3050,bins)
        bst_sum = bst_df.apply(np.sum,axis=1)   #sum by row across columns
      #  bst_sum.index = np.arange(550,3050,bins)
        channels[item][key]['binned_spike_list'] = bst_list
        channels[item][key]['binnedSpikes'] = bst_df
        channels[item][key]['binnedSpksSum'] = bst_sum
        bst_arr = None
        bst_df = None
        bst_sum = None

#%%
###PSTH with binned spike counts

#make gaussian kernel with 2 ms SD
gauKern = elephant.statistics.make_kernel('GAU',2*pq.ms,1*pq.ms)
#%%
#convolve kernel with summed spike times for full psth
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
for item in g_list:
    plt.figure()
    plt.title(item+' PSTH')
    plt.xlim(0,3200)
    for key in channels[item]:
        tmpbst_sum = channels[item][key]['binnedSpksSum']
        psth = scipy.convolve(gauKern[0],tmpbst_sum)
        plt.plot(psth,label=item+'*'+key)
        channels[item][key]['PSTH'] = psth
    plt.legend()
    plt.axis('tight')
    #plt.show()
    plt.savefig(item+'_psth.png',dpi=600,format='png')
    tmpbst_sum = None

#%% Do a detail psth (psth_detail) for some piece of 500 ms
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
for item in g_list:
    plt.figure()
    plt.title(item+' PSTH Detail')
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing Rate Normalized')
    for key in channels[item]:
        tmpbst_sum = channels[item][key]['binnedSpksSum']
        psth = scipy.convolve(gauKern[0],tmpbst_sum)
        psth = pd.DataFrame(psth)
        psth.index = np.arange(650,3170,1)
        plt.plot(psth.iloc[950:1250],label='conductance*'+key)
        channels[item][key]['PSTH'] = psth
    plt.legend()
    plt.axis('tight')
#    plt.show()
    plt.savefig(item+'_psth_detail.png',dpi=600,format='png')

#%%
###cross-trial correlations WITHIN-cell

from elephant.spike_train_correlation import corrcoef

for item in g_list:
    for key in channels[item]:
        tmpcorr = corrcoef(channels[item][key]['binned_spike_list'])
        tmpmean = np.mean(tmpcorr)
        channels[item][key]['corr'] = tmpmean
        tmpcorr = None
        tmpmean = None

#%%
"""
12 Dec 2017
The elephant BOX kernel doesn't seem to work really, it fails, not sure if me
or something wrong with the code itself. Honestly, could just make the array
myself for the most part as it's just a value of 1 for the width of 2delta.
Anyway, this first cell is how to do it on a single binary binned spike train.
Next cell will be doing it for all simulations.

This cell now does a loop through 8 delta ms values, through all the sim files
and through all trials in simulations, and puts out a df that includes info
about the WNS SD and correlation
"""
###do pairwise cross-trial correlations by Galan's window method
#make square boxcar kernel with sigma of delta ms
#delta_list = [1,2,3,4,5,6,7,8]
#mean_corr_df = pd.DataFrame()
#column_list = ['Filename','WNS SD','WNS Corr.','1 ms','2 ms','3 ms','4 ms',
#                         '5 ms','6 ms','7 ms','8 ms']
tmpbox = scipy.signal.boxcar(9)
for item in g_list:
    for key in channels[item]:
        tmpboxdf = pd.DataFrame()
        tmpbstdf = channels[item][key]['binnedSpikes']
        for i in range(50):
            tmpkern = scipy.signal.convolve(tmpbox, tmpbstdf.iloc[:,i])
            tmpkerndf = pd.DataFrame(tmpkern)
            tmpboxdf = pd.concat([tmpboxdf,tmpkerndf],axis=1)
        st_corr = tmpboxdf.corr()
        mean_st_corr = np.mean(st_corr.mean().values)
        channels[item][key]['boxcorr']= mean_st_corr
        st_corr = None
        mean_st_corr = None
        tmpboxdf = None
        tmpkern = None
        tmpkerndf = None
        tmpbstdf = None
tmpbox = None

#%% Put the correlations into a table that's easy to read
#Need to get correlation and possibly delta numbers in float format.
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/summary')
corrdf = pd.DataFrame()
for item in g_list:
    for key in channels[item]:
        tmpcorr = pd.Series(channels[item][key]['boxcorr'])
        tmpcorrdf = pd.DataFrame([item,key]).T
        tmpcorrdf = pd.concat([tmpcorrdf,tmpcorr],axis=1)
        corrdf = pd.concat([corrdf,tmpcorrdf])
        tmpcorr = None
        tmpcorrdf = None
corrdf.columns = ['Channel','Delta','Corr. Coef.']
corrdf.to_csv('wns_delta_boxcorr.csv',sep=',',float_format='%10.4f',index=False,header=True)

#%% Make a plot of correlations as a function of delta conductance
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
plt.figure()
plt.xlabel('Global Channel Conductance Factor')
plt.ylabel('Correlation Coefficient')
for item in g_list:
    x = corrdf['Delta'][corrdf['Channel']==item]
    y = corrdf['Corr. Coef.'][corrdf['Channel']==item]
    plt.plot(x,y, label=item)
plt.legend(loc=0)
#plt.show()
plt.savefig('spiketime_correlation.png',dpi = 600, format = 'png')

#%% Compute the firing frequency for each trace and append to corrdf
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/summary')
frdf = pd.Series()
for item in g_list:
    for key in channels[item]:
        spiketrain_list = channels[item][key]['neoSpkTrain500']
        fr = [elephant.statistics.mean_firing_rate(spiketrain.rescale(pq.s),
                                                 t_start=0.65, t_stop= 3.16)
              for spiketrain in spiketrain_list]
        frmean = np.mean(fr)
        frser = pd.Series(frmean)
        frdf = pd.concat([frdf,frser])
statsdf = pd.concat([corrdf,frdf],axis=1)
statsdf.columns = ['Channel','Delta','Corr. Coef.','FR (Hz)']
statsdf.to_csv('wns_delta_stats.csv',sep=',',float_format='%10.4f',index=False,header=True)


#%% Make a plot of firing rate as function of delta conductance
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
plt.figure()
plt.xlabel('Global Channel Conductance Factor')
plt.ylabel('Firing Rate (Hz)')
for item in g_list:
    x = statsdf['Delta'][statsdf['Channel']==item]
    y = statsdf['FR (Hz)'][statsdf['Channel']==item]
    plt.plot(x,y, label=item)
plt.ylim(0,90)
plt.yticks(np.arange(0,92,10))
plt.legend(loc=0)
#plt.show()
plt.savefig('spike_frequency.png',dpi = 600, format = 'png')


"""
Tried to do a few different fano factor calculations, not sure if what I have is
correct, but the numbers seem to be somewhat logical.
#%% Attempt fano factor calculation
ffdf = pd.DataFrame()
for item in g_list:
    for key in channels[item]:
        ff = elephant.statistics.fanofactor(channels[item][key]['neoSpkTrain500'])
        ffs = pd.Series(ff)
        ffdf = pd.concat([ffdf,ffs])
#%% another attempt at fano factor with rolling window stole from dautil
def fano_factor(arr, window):
    ''' Calculates the Fano factor a windowed variance-to-mean ratio.

    .. math:: F=\\frac{\\sigma_W^2}{\\mu_W}

    :param arr: Array-like list of values.
    :param window: Size of the window.

    :returns: The Fano factor.

    *See Also*

    https://en.wikipedia.org/wiki/Fano_factor
    '''
    return pd.rolling_var(arr, window)/pd.rolling_mean(arr, window)
#%% actual fano factor running
x = fano_factor(test[0],100)"""
#%% another attmept at fano_factor, a real one this time
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/summary')
fanodf = pd.Series()
for item in g_list:
    for key in channels[item]:
        tfanoser = pd.Series()
        for i in range(50):
            tmpser = channels[item][key]['binnedSpikes'].iloc[:,i]
            if np.sum(tmpser) == 0:
                continue
            else:
                window_spikes = tmpser.rolling(100).sum()
                tmpff = window_spikes.var()/window_spikes.mean()
                tmpffs = pd.Series(tmpff)
                tfanoser = pd.concat([tfanoser,tmpffs])
        tfanomean = pd.Series(tfanoser.mean())
        fanodf = pd.concat([fanodf,tfanomean])
statsdf2 = pd.concat([statsdf,fanodf],axis=1)
statsdf2.columns = ['Channel','Delta','Corr. Coef','FR (Hz)','Fano Factor']
statsdf2.to_csv('wns_delta_stats2.csv',sep=',',float_format='%10.4f',index=False,header=True)

#%% plotting Fano factor
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
plt.figure()
plt.xlabel('Global Channel Conductance Factor')
plt.ylabel('Fano Factor')
for item in g_list:
    x = statsdf2['Delta'][statsdf2['Channel']==item]
    y = statsdf2['Fano Factor'][statsdf2['Channel']==item]
    plt.plot(x,y, label=item)
plt.legend(loc=1)
#plt.show()
plt.savefig('fano_factor.png',dpi = 600, format = 'png')
#%% Set up WNS to only look at stimulus after the first 500 ms
wns_list=[]
for i in range(50):
    wns = wns_i.iloc[6500:,i]
    wns2 = AnalogSignal(signal=wns.values*pq.nA,
                        sampling_rate=10*pq.kHz,
                        t_start=650.0*pq.ms)
    wns_list.append(wns2)

#%% Set up STA excluding spikes in first 500 ms
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
for item in g_list:
    plt.figure()
    plt.xlabel('Time Before Spike (ms)')
    plt.ylabel('Current (nA)')
    plt.title(item+' Channel STA')
    for key in channels[item]:
        sta_ind = pd.DataFrame()                #starting time column to build on
        for i in range(50):                     #adding each columb of sta to df
            sta_i = spike_triggered_average(wns_list[i],
                                            channels[item][key]['neoSpkTrain500'][i],
                                            (-40*pq.ms,5.1*pq.ms))
            sta_i_df = pd.DataFrame(sta_i.as_array().flatten())
            sta_ind = pd.concat([sta_ind,sta_i_df],axis=1)
        sta_avg = sta_ind.apply(np.mean,axis=1) #average each row across columns
        t_sta = np.arange(-40,5.1,0.1)         #time from -50 to 10 ms inclusive by 0.1
        t_df = pd.DataFrame(t_sta)              #make a time data frame
        channels[item][key]['STA'] = sta_avg
        plt.plot(t_df,sta_avg,label='g'+item+'_bar*'+key) #easier to call as separate x,y
        sta_i = None
        sta_i_df = None
        sta_ind = None
        sta_avg = None
    plt.legend()
    #plt.show()
#    plt.savefig(item+'_sta.png',dpi=600,format='png')

# %% STA PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

% matplotlib
sta_df = pd.DataFrame()
pc_var_list = []
for item in g_list:
    temp_sta_df = pd.DataFrame()
    for key in channels[item]:
        temp_sta = pd.DataFrame(data = channels[item][key]['STA'])
        if temp_sta.isnull().values.any() == True:
            continue
        else:
            mean_sub_sta = temp_sta - 0.45
            mean_sub_sta = mean_sub_sta.T
            mean_sub_sta['channel'] = item
            mean_sub_sta['density'] = key
            temp_sta_df = pd.concat([temp_sta_df, mean_sub_sta])
    sta_df = pd.concat([sta_df, temp_sta_df])
sta_df.index = range(30) #need to reindex so that the concat works later
X = sta_df.iloc[:, :451]
meta = pd.DataFrame(sta_df.iloc[:, 451:])
pca = PCA(n_components=2)
pc = pca.fit_transform(X)
temp_pc_df = pd.DataFrame(pc, columns = ['PC1', 'PC2'])
final_pc_df = pd.concat([temp_pc_df, sta_df.iloc[:,451:]], axis=1)

fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('STA PCA', fontsize = 20)
colors = ['b', 'r', 'g', 'y', 'c', 'k', 'm', 'tab:gray']
for item, color in zip(g_list, colors):
    indicesToKeep = final_pc_df['channel'] == item
    ax.plot(final_pc_df.loc[indicesToKeep, 'PC1'],
            final_pc_df.loc[indicesToKeep, 'PC2'],
            c = color, marker = '.', markersize = 12)
ax.legend(g_list)
os.chdir('/Users/nate/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
fig.savefig('STA_PCA_L5pc_a.png', dpi = 300, format='png')

# %% For doing the first five components and figuring out the variance of each
pca5 = PCA(n_components=5)
pc5 = pca5.fit(X)
pc_var = pca5.explained_variance_ratio_

# %% For doing what Krishnan does in Fig. 4C of 2010 paper.
pcaT = PCA(n_components=3)
X_T = X.T
pcT = pcaT.fit_transform(X_T)
temp_pcT_df = pd.DataFrame(pcT)


#%% Set up mean subtracted WNS to only look at stimulus after the first 500 ms
wns_meansub_list=[]
for i in range(50):
    wns = wns_i.iloc[6500:,i]
    wns2 = AnalogSignal(signal=(wns.values - 0.45)*pq.nA,
                        sampling_rate=10*pq.kHz,
                        t_start=650.0*pq.ms)
    wns_meansub_list.append(wns2)

os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
for item in g_list:
    plt.figure()
    plt.xlabel('Time Before Spike (ms)')
    plt.ylabel('Current (nA)')
    plt.title(item+' Channel STA')
    for key in channels[item]:
        sta_ind = pd.DataFrame()                #starting time column to build on
        for i in range(50):                     #adding each columb of sta to df
            sta_i = spike_triggered_average(wns_meansub_list[i],
                                            channels[item][key]['neoSpkTrain500'][i],
                                            (-40*pq.ms,5.1*pq.ms))
            sta_i_df = pd.DataFrame(sta_i.as_array().flatten())
            sta_ind = pd.concat([sta_ind,sta_i_df],axis=1)
        sta_avg = sta_ind.apply(np.mean,axis=1) #average each row across columns
        t_sta = np.arange(-40,5.1,0.1)         #time from -50 to 10 ms inclusive by 0.1
        t_df = pd.DataFrame(t_sta)              #make a time data frame
        channels[item][key]['STA'] = sta_avg
        plt.plot(t_df,sta_avg,label='g'+item+'_bar*'+key) #easier to call as separate x,y
        sta_i = None
        sta_i_df = None
        sta_ind = None
        sta_avg = None
    plt.legend()
    #plt.show()
    plt.savefig(item+'_sta_meansubtracted.png',dpi=600,format='png')

#%% ISI and CV isi, has problem when there are no spikes at all.
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/summary')
from elephant.statistics import isi, cv
cvdf = pd.Series()
for item in g_list:
    for key in channels[item]:
        isi_list = [isi(spiketrain) for spiketrain in channels[item][key]['spikeTimes500']]
        cv_list = [cv(isi_list[i]) for i in range(len(isi_list))]
        channels[item][key]['isi'] = isi_list
        channels[item][key]['CVisi'] = np.mean(cv_list)
        cv_mean = pd.Series(np.mean(cv_list))
        cvdf = pd.concat([cvdf,cv_mean])
stats3df = pd.concat([statsdf2,cvdf],axis=1)
stats3df = stats3df.rename(columns = {0:'CV ISI'})
stats3df.to_csv('wns_delta_stats3.csv',float_format='%10.4f',index=False,header=True)

#%% plot cv isi as function of delta conductance
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/'
         'AlmogAndKorngreen2014/ModCell_5_thrdsafe/figures')
plt.figure()
plt.xlabel('Global Channel Conductance Factor')
plt.ylabel('CV ISI')
for item in g_list:
    x = stats3df['Delta'][stats3df['Channel']==item]
    y = stats3df['CV ISI'][stats3df['Channel']==item]
    plt.plot(x,y, label=item)
plt.legend(loc=0)
#plt.show()
plt.savefig('CV_ISI.png',dpi = 600, format = 'png')


'''*****************************************************************************
*********************************************************************************
********************************************************************************
###############################################################################
I'm not sure anything below here is real
'''
#%%%
st_df_list = []
for item in binnedst_list:
    st_df = pd.DataFrame(item.to_array()).T
    st_df_list.append(st_df)



#%%
psth_df = pd.DataFrame(psth_list).T
psth_df.to_csv('PSTH_20ms.txt',sep='\t',float_format='%10.6f',index=False,header=False)

#%%
for i in range(16):
    st_df_list[i].to_csv('binSpkTime_1ms_'+str(i+1)+'.txt',sep='\t',float_format='%10.6f',index=False,header=False)
#%%
for i in range(16):
    sta_list[i].to_csv('STA_'+str(i+1)+'.txt',sep='\t',float_format='%10.6f',index=False,header=False)
#%%
count = pd.DataFrame(range(1,17))
stcdf = pd.DataFrame(np.asarray(spkt_corr_list))
stc_df = pd.concat([count,stcdf],axis=1)
stc_df.to_csv('stCorr_20ms.txt',sep='\t',float_format='%10.6f',index=False,header=False)
#%%
sta_list = []
sta_dir = os.listdir(os.getcwd())
for i in range(len(sta_dir)):
    if 'STA' in sta_dir[i]:
        tmpsta = pd.read_csv(sta_dir[i],sep='\s+',header=None)
        sta_list.append(tmpsta)
#%%
#plotting all STAs together on real time axis
plt.figure()
for i in range(len(sta_list)):
    plt.plot(t_df,sta_list[i])
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.show()

#%%
#plotting STAs of same correlation together on real time axis
plt.figure()
for i in range(4):
    plt.plot(t_df,sta_list[i])
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.show()
plt.figure()
for i in range(4,8):
    plt.plot(t_df,sta_list[i])
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.show()
plt.figure()
for i in range(8,12):
    plt.plot(t_df,sta_list[i])
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.show()
plt.figure()
for i in range(12,16):
    plt.plot(t_df,sta_list[i])
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.show()
plt.figure()
for i in [3,7,11,15]:
    plt.plot(t_df,sta_list[i])
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.show()
#%%
#plotting all STAs together on index for measuring
plt.figure()
for i in range(len(sta_list)):
    plt.plot(sta_list[i])
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.show()
#%%
#plotting 5030 and 9030 STAs together on real time
plt.figure()
for i in [3,15]:
    plt.plot(t_df,sta_list[i], label='WNS Corr.=  ')
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.legend()
    plt.show()

#%%
#measuring peak of all STAs
sta_peak = []
for i in range(len(sta_list)):
    tmppeak = np.mean(sta_list[i].iloc[483:497])
    sta_peak.append(tmppeak)

#%%
#measuring slope of all STAs
sta_line = []
sta_slope = []
t = pd.Series(np.arange(-50,10.1,0.1))
for i in range(len(sta_list)):
    x = t.iloc[415:483]
    y = sta_list[i].iloc[415:483]
    tmpline = scipy.stats.linregress(x,y)
    sta_line.append(tmpline)
    sta_slope.append(tmpline.slope)


#%%
#plotting 5030 and 9030 wns together on real time
xstart = 10000
xend = 15000
t = pd.Series(np.arange(0,3100,0.1))
for k in [3,15]:
    plt.figure()
    plt.plot(t,i_list[k][0])
    plt.plot(t,i_list[k][1])
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.show()

    plt.figure()
    plt.plot(t.iloc[xstart:xend],i_list[k][0].iloc[xstart:xend])
    plt.plot(t.iloc[xstart:xend],i_list[k][1].iloc[xstart:xend])
    plt.axis('tight')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Current (nA)', fontsize=16)
    plt.show()
#%%
#plotting 5030 and 9030 vm together on real time
xstart = 15000
xend = 20000
t = pd.Series(np.arange(0,3100,0.1))
for k in [3,15]:
    plt.figure()
    plt.plot(t,vmt_list[k][0])
    plt.plot(t,vmt_list[k][1])
    plt.axis('tight')
    plt.xlim(0,3100)
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Vm (mV)', fontsize=16)
    plt.show()

    plt.figure()
    plt.plot(t.iloc[xstart:xend],vmt_list[k][0].iloc[xstart:xend])
    plt.plot(t.iloc[xstart:xend],vmt_list[k][1].iloc[xstart:xend])
    plt.axis('tight')
    plt.xlim(1500,2000)
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Vm (mV)', fontsize=16)
    plt.show()



#%%
#finding correlation between spike train 0,1 for wns3 and 15
from elephant.conversion import BinnedSpikeTrain
st3_subset = [spktrain_list[3][0],spktrain_list[3][1]]
bst3_list = BinnedSpikeTrain(st3_subset,
                             binsize=1.0*pq.ms,
                             t_start=0.0*pq.ms,
                             t_stop=3100*pq.ms)


st15_subset = [spktrain_list[15][0],spktrain_list[15][1]]
bst15_list = BinnedSpikeTrain(st15_subset,
                              binsize=1.0*pq.ms,
                              t_start=0.0*pq.ms,
                              t_stop=3100*pq.ms)

from elephant.spike_train_correlation import corrcoef
stcorr3 = corrcoef(bst3_list, binary=True)
stcorr3_mean = np.mean(stcorr3)
print(stcorr3_mean)
stcorr15 = corrcoef(bst15_list, binary=True)
stcorr15_mean = np.mean(stcorr15)
print(stcorr15_mean)
