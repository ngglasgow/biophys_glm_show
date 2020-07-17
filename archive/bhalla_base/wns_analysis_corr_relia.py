# -*- coding: utf-8 -*-
"""
******************************************************************************
9 Oct 2017
Second Run of 100 trial simulation with 600 pA WNS generated today.
Need more correlation in spike trains so increase correlation of WNS

WNS file "wns_600_b.txt" stats:
600 pA DC injection gives 35.238 Hz firing rate.
Correlation Coefficient: 0.836
Standard Deviation: 0.0344 nA

Simulation run of "init_wns_600_b.hoc"
Generates files: wns_i_600_b.dat; wns_st_600_b.dat; wns_v_600_b.dat;
Full run takes about 25 minutes.

Spike train correlation: 0.0663


******************************************************************************

"""
"""
NOTE: Need to change file saving notation from NEURON so that each number is
comma separated. Use format {
fi_f = new File("fi_i.csv")
fi_f.wopen()
fi_m.fprint(0, fi_f, "%8f\,")
fi_f.close()}
This gives an 8 point precision floating point (I think), and if saved to .csv
then can be read easily in pandas. However, it also separates each element
including the final element with a comma, so need to either slice it during the
read or after the read. Either should work fine. See code below.
************
NEW NOTE 22Sep2017
************
If sep='\s+' it means separated by at least 1 space and up to n spaces. Works
more elegantly than the method above, although that method is also valid.

"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neo.core import SpikeTrain
import elephant
import quantities as pq
from elephant.sta import spike_triggered_average
import scipy
import os

#%%
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/simulations')
os.chdir('/Users/nate/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/simulations')
dir_list = os.listdir(os.getcwd())
dir_list.sort()
print dir_list

t = np.arange(0,3100,0.1)
tm = pd.DataFrame(t)

i_list = []
it_list = []
i_files = []
vm_list = []
vmt_list = []
vm_files = []
st_list = []
st_files = []


#%%
#Read files into lists
for k in range(len(dir_list)):
    if 'i' in dir_list[k]:
        tmpi = pd.read_csv(dir_list[k],sep='\s+',header=None)
        tmpi = tmpi.iloc[:31000]
        i_list.append(tmpi)
        tmpi2 = pd.concat([tm,tmpi],axis=1)
        it_list.append(tmpi2)
        i_files.append(dir_list[k])
    elif 'v' in dir_list[k]:
        tmpv = pd.read_csv(dir_list[k],sep='\s+',header=None)
        tmpv = tmpv.iloc[:31000]
        vm_list.append(tmpv)
        tmpv2 = pd.concat([tm,tmpv],axis=1)
        vmt_list.append(tmpv2)
        vm_files.append(dir_list[k])
    else:
        tmpst = pd.read_csv(dir_list[k],sep='\s+',header=None)
        tmpst = tmpst.iloc[:31000]
        ttmpst = pd.concat([tm,tmpst],axis=1)
        st_list.append(ttmpst)
        st_files.append(dir_list[k])

#%%
#Make a list of neo.spiketrain lists
spktrain_list = []
for item in st_list:
    spkt_list = []
    for i in range(1,51):
        st = item.iloc[:,[0,i]]
        st2 = st.iloc[:,0][st.iloc[:,1]>0]
        st3 = SpikeTrain(times=st2.values*pq.ms,t_start=0.,t_stop=3100)
        spkt_list.append(st3)
    spktrain_list.append(spkt_list)

#%%

#To amake a graph raster plot like in elephant tutorial.
raster_list = []
for item in spktrain_list:
    plt.figure()
    for i, spiketrain in enumerate(item):
        t = spiketrain.rescale(pq.ms)
        raster_list.append(t)
        plt.plot(t,(i+1)*np.ones_like(t),'k.',markersize=2)
        plt.axis('tight')
        plt.xlim(0,3100)
        plt.xlabel('Time (ms)',fontsize=16)
        plt.ylim(0,51)
        plt.ylabel('Trial Number', fontsize=16)
        plt.gca().tick_params(axis='both',which='major',labelsize=14)
        plt.show()


#%%
#CV isi NEVER CONVERTED FOR MULTIPLE FILES###
from elephant.statistics import isi, cv
isi_list = [isi(spiketrain) for spiketrain in st_list]
cv_list = [cv(item) for item in isi_list]
plt.figure()
plt.hist(cv_list)
plt.xlabel('CV', fontsize=16)
plt.ylabel('count', fontsize=16)
plt.show()
plt.figure()
plt.plot(cv_list)
#%%
"""
28Sep2017
NOTE: Work on conversion of discrete spike time lists to binary spike counts.
Use this conversion for PSTH and for spike-time correlations.
"""
##ADDED bstc_df to cut the first 500 ms of data out for correlations
###conversion of discrete spike times to binary counts

from elephant.conversion import BinnedSpikeTrain
binnedst_list = []
bst_df_list = []
bstc_df_list = []
bst_sum_list = []
for item in spktrain_list:
    bst_list = BinnedSpikeTrain(item,
                                binsize=1.0*pq.ms,
                                t_start=0.0*pq.ms,
                                t_stop=3100*pq.ms)


    bst_arr = bst_list.to_array()           #export binned spike times to an array
    bst_df = pd.DataFrame(bst_arr).T        #turn into a df and transpose (.T)
    bst_sum = bst_df.apply(np.sum,axis=1)   #sum by row across columns
    bst_df_list.append(bst_df)
    bstc_df = bst_df.iloc[550:3050]
    bstc_df_list.append(bstc_df)
    #plt.figure()
    #plt.plot(bst_sum)
    binnedst_list.append(bst_list)
    bst_sum_list.append(bst_sum)

    bstc_df_list
#%%
###PSTH with binned spike counts

#make gaussian kernel with 2 ms SD
gauKern = elephant.statistics.make_kernel('GAU',2*pq.ms,1*pq.ms)
#%%
#convolve kernel with summed spike times
psth_list = []
for item in bst_sum_list:
    psth = scipy.convolve(gauKern[0],item)
    #plt.figure()
    #plt.plot(psth)
    #plt.axis('tight')
    #plt.xlim(0,3100)
    psth_list.append(psth)
#%%
###cross-trial correlations WITHIN-cell

from elephant.spike_train_correlation import corrcoef
spkt_corr_list = []
for item in binnedst_list:
    stcorr = corrcoef(item, binary=True)
    stcorr_mean = np.mean(stcorr)
    print(stcorr_mean)
    spkt_corr_list.append(stcorr_mean)
    stcorr_list = stcorr.flatten()
    #plt.figure()
    #plt.hist(stcorr_list,bins=200,normed=True,histtype='step')
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
###do pairwise cross-trial Pearson correlations by Galan's window method
#make square boxcar kernel with sigma of delta ms
# Pearson correlation coefficient
# NOTE not centered on spike time for convolution
delta_list = [1,2,3,4,5,6,7,8]
mean_corr_df = pd.DataFrame()
column_list = ['Filename', 'WNS SD %', 'WNS Corr.', '0 ms', '1 ms', '2 ms',
               '3 ms', '4 ms', '5 ms', '6 ms', '7 ms', '8 ms']
for k in range(8):
    tmpbox = scipy.signal.boxcar((delta_list[k]*2+1))
    mean_corr_list = []
    for item in bstc_df_list:
        tmpboxdf = pd.DataFrame()
        for i in range(50):
            tmpkern = scipy.signal.convolve(tmpbox,item.iloc[:,i])
            tmpkerndf = pd.DataFrame(tmpkern)
            tmpboxdf = pd.concat([tmpboxdf,tmpkerndf],axis=1)
        st_corr = tmpboxdf.corr()
        mean_st_corr = np.mean(st_corr.mean().values)
        mean_corr_list.append(mean_st_corr)
    tmparr = np.asarray(mean_corr_list)
    tmpdf = pd.DataFrame(tmparr)
    mean_corr_df = pd.concat([mean_corr_df,tmpdf],axis=1)
wns_sd = pd.DataFrame([10, 15, 20, 30,
                       10, 15, 20, 30,
                       10, 15, 20, 30,
                       10, 15, 20, 30])
wns_corr = pd.DataFrame([0.5, 0.5, 0.5, 0.5,
                         0.7, 0.7, 0.7, 0.7,
                         0.8, 0.8, 0.8, 0.8,
                         0.9, 0.9, 0.9, 0.9])
spktrain_corr = pd.concat([pd.DataFrame(np.asarray(st_files)),wns_sd,wns_corr,mean_corr_df],axis=1)
spktrain_corr.columns = column_list

# save to file
os.chdir('/Users/nate/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/output')
spktrain_corr.to_csv('spktrain_notcentered_pearsoncorr_ones_galan.txt',sep=',',float_format='%10.6f',index=False,header=True)

###do pairwise cross-trial Pearson correlations by Galan's window method
#make square boxcar kernel with sigma of delta ms
# Pearson correlation coefficient
# NOTE IS centered on spike time for convolution
spktrain_corr = None
delta_list = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]

mean_corr_df = pd.DataFrame()
column_list = ['Filename', 'WNS SD %', 'WNS Corr.', '0 ms', '0.5 ms', '1 ms', '2 ms',
               '3 ms', '4 ms', '5 ms', '6 ms', '7 ms', '8 ms']
for k in range(len(delta_list)):
    tmpbox = scipy.signal.boxcar(int((delta_list[k]*2+1)))
    mean_corr_list = []
    for item in bstc_df_list:
        tmpboxdf = pd.DataFrame()
        for i in range(50):
            tmpkern = scipy.ndimage.convolve1d(item.iloc[:, i], tmpbox)
            tmpkerndf = pd.DataFrame(tmpkern)
            tmpboxdf = pd.concat([tmpboxdf, tmpkerndf], axis=1)
        st_corr = tmpboxdf.corr()
        st_corr_no_ones = st_corr[st_corr < 1]
        mean_st_corr = np.mean(st_corr_no_ones.mean().values)
        mean_corr_list.append(mean_st_corr)
    tmparr = np.asarray(mean_corr_list)
    tmpdf = pd.DataFrame(tmparr)
    mean_corr_df = pd.concat([mean_corr_df, tmpdf], axis=1)
wns_sd = pd.DataFrame([10, 15, 20, 30,
                       10, 15, 20, 30,
                       10, 15, 20, 30,
                       10, 15, 20, 30])
wns_corr = pd.DataFrame([0.5, 0.5, 0.5, 0.5,
                         0.7, 0.7, 0.7, 0.7,
                         0.8, 0.8, 0.8, 0.8,
                         0.9, 0.9, 0.9, 0.9])
spktrain_corr = pd.concat([pd.DataFrame(np.asarray(st_files)), wns_sd, wns_corr, mean_corr_df], axis=1)
spktrain_corr.columns = column_list
spktrain_corr
spktrain_corr = None
mean_corr_df
# save to file
os.chdir('/Users/nate/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/output')
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/output')
spktrain_corr.to_csv('spktrain_centered_pearsoncorr_no_ones_galan.txt',sep=',',float_format='%10.6f',index=False,header=True)


# GALAN NON-MEAN-SUBTRACTED RELIABILIY/CORRELATION COEFFICIENT CALCULATION
# box kernel with width 8 ms (2*jitter) and amplitude 1
# NOTE not centered on spike time for convolution
spktrain_corr = None
delta_list = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
mean_corr_df = pd.DataFrame()
corr_mat_df_df = pd.DataFrame()
column_list = ['Filename', 'WNS SD %', 'WNS Corr.', '0 ms', '0.5 ms', '1 ms', '2 ms',
               '3 ms', '4 ms', '5 ms', '6 ms', '7 ms', '8 ms']
len(bstc_df_list)
for k in range(len(delta_list)):
    tmpbox = scipy.signal.boxcar(int((delta_list[k]*2 + 1)))
    corr_mat_list = []
    mean_corr_list = []
    for item in bstc_df_list:
            # the binary spike time data frame
            tmpbstdf = item

            # set number of neurons for trial to trial correlations/reliability
            num_spiketrains = len(tmpbstdf.columns)

            # pre-allocate correlaion matrix
            corr_mat = np.zeros((num_spiketrains, num_spiketrains))

            # set up empty data frame for convolved spike train data frame
            tmpcstdf = pd.DataFrame()

            # convolve box kernel with binary spike train data frame to make convolved
            # signal data frame
            for i in range(num_spiketrains):
                tmpkern = scipy.ndimage.convolve1d(tmpbstdf.iloc[:, i], tmpbox)
                tmpkerndf = pd.DataFrame(tmpkern)
                tmpcstdf = pd.concat([tmpcstdf, tmpkerndf], axis=1)

            for i in range(num_spiketrains):
                for j in range(i, num_spiketrains):
                    # numerator = dot product of (cst_i, cst_j),
                    # where cst_i is the ith binary spike train convolved with a box kernel
                    # and cst_j is the jth binary spike train convolved with a box kernel
                    cst_i = tmpcstdf.iloc[:, i]
                    cst_j = tmpcstdf.iloc[:, j]
                    numerator = np.dot(cst_i, cst_j)

                    # denominator = product of norm(cst_i, cst_j),
                    # where norm = the sqrt(sum(x**2)) for cst_i and cst_j
                    cst_i_norm = np.linalg.norm(cst_i)
                    cst_j_norm = np.linalg.norm(cst_j)
                    denominator = cst_i_norm * cst_j_norm

                    # calculate the correlation coefficient and build the matrix
                    corr_mat[i, j] = corr_mat[j, i] = numerator / denominator

            corr_mat_df = pd.DataFrame(corr_mat)
            #corr_mat_list.append(corr_mat_df)
            #corr_mat_df_df = pd.concat([corr_mat_df_df, corr_mat_df], axis=2)

            # take out 1's diaganol for mean calculation and maybe plotting
            corr_mat_df_no_ones = corr_mat_df[corr_mat_df < 1]
            mean_corr = np.mean(corr_mat_df_no_ones.mean().values)
            mean_corr_list.append(mean_corr)

            # null out data frames to not overflow memory
            tmpbstdf = None
            num_spiketrains = None
            corr_mat = None
            tmpcstdf = None
            tmpkern = None
            tmpkerndf = None
            cst_i = None
            cst_j = None
            numerator = None
            cst_i_norm = None
            cst_j_norm = None
            denominator = None
            corr_mat_df = None
            corr_mat_df_no_ones = None
            mean_corr = None
    tmparr = np.asarray(mean_corr_list)
    tmpdf = pd.DataFrame(tmparr)
    mean_corr_df = pd.concat([mean_corr_df, tmpdf], axis=1)
    tmpbox = None
wns_sd = pd.DataFrame([10, 15, 20, 30,
                       10, 15, 20, 30,
                       10, 15, 20, 30,
                       10, 15, 20, 30])
wns_corr = pd.DataFrame([0.5, 0.5, 0.5, 0.5,
                         0.7, 0.7, 0.7, 0.7,
                         0.8, 0.8, 0.8, 0.8,
                         0.9, 0.9, 0.9, 0.9])
spktrain_corr = pd.concat([pd.DataFrame(np.asarray(st_files)),wns_sd,wns_corr,mean_corr_df],axis=1)
spktrain_corr.columns = column_list

# save to file
os.chdir('/Users/nate/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/output')
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/output')
spktrain_corr.to_csv('spktrain_notcentered_reliabilitycorr_no_ones_galan.txt',sep=',',float_format='%10.6f',index=False,header=True)


# GALAN NON-MEAN-SUBTRACTED RELIABILIY/CORRELATION COEFFICIENT CALCULATION
# box kernel with width 8 ms (2*jitter) and amplitude 1
# NOTE IS centered on spike time for convolution
spktrain_corr = None
delta_list = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
mean_corr_df = pd.DataFrame()
corr_mat_df_df = pd.DataFrame()
column_list = ['Filename', 'WNS SD %', 'WNS Corr.', '0 ms', '0.5 ms', '1 ms', '2 ms',
               '3 ms', '4 ms', '5 ms', '6 ms', '7 ms', '8 ms']
corr_mat_df_list = []
for k in range(len(delta_list)):
    tmpbox = scipy.signal.boxcar(int((delta_list[k]*2 + 1)))
    corr_mat_list = []
    mean_corr_list = []
    for item in bstc_df_list:
            # the binary spike time data frame
            tmpbstdf = item

            # set number of neurons for trial to trial correlations/reliability
            num_spiketrains = len(tmpbstdf.columns)

            # pre-allocate correlaion matrix
            corr_mat = np.zeros((num_spiketrains, num_spiketrains))

            # set up empty data frame for convolved spike train data frame
            tmpcstdf = pd.DataFrame()

            # convolve box kernel with binary spike train data frame to make convolved
            # signal data frame
            for i in range(num_spiketrains):
                tmpkern = scipy.ndimage.convolve1d(tmpbstdf.iloc[:, i], tmpbox)
                tmpkerndf = pd.DataFrame(tmpkern)
                tmpcstdf = pd.concat([tmpcstdf, tmpkerndf], axis=1)

            for i in range(num_spiketrains):
                for j in range(i, num_spiketrains):
                    # numerator = dot product of (cst_i, cst_j),
                    # where cst_i is the ith binary spike train convolved with a box kernel
                    # and cst_j is the jth binary spike train convolved with a box kernel
                    cst_i = tmpcstdf.iloc[:, i]
                    cst_j = tmpcstdf.iloc[:, j]
                    numerator = np.dot(cst_i, cst_j)

                    # denominator = product of norm(cst_i, cst_j),
                    # where norm = the sqrt(sum(x**2)) for cst_i and cst_j
                    cst_i_norm = np.linalg.norm(cst_i)
                    cst_j_norm = np.linalg.norm(cst_j)
                    denominator = cst_i_norm * cst_j_norm

                    # calculate the correlation coefficient and build the matrix
                    corr_mat[i, j] = corr_mat[j, i] = numerator / denominator

            corr_mat_df = pd.DataFrame(corr_mat)
            corr_mat_df_list.append(corr_mat_df)
            #corr_mat_list.append(corr_mat_df)
            #corr_mat_df_df = pd.concat([corr_mat_df_df, corr_mat_df], axis=2)

            # take out 1's diaganol for mean calculation and maybe plotting
            corr_mat_df_no_ones = corr_mat_df[corr_mat_df < 1]
            mean_corr = np.mean(corr_mat_df_no_ones.mean().values)
            mean_corr_list.append(mean_corr)

            # null out data frames to not overflow memory
            tmpbstdf = None
            num_spiketrains = None
            corr_mat = None
            tmpcstdf = None
            tmpkern = None
            tmpkerndf = None
            cst_i = None
            cst_j = None
            numerator = None
            cst_i_norm = None
            cst_j_norm = None
            denominator = None
            corr_mat_df = None
            corr_mat_df_no_ones = None
            mean_corr = None
    tmparr = np.asarray(mean_corr_list)
    tmpdf = pd.DataFrame(tmparr)
    mean_corr_df = pd.concat([mean_corr_df, tmpdf], axis=1)
    tmpbox = None
wns_sd = pd.DataFrame([10, 15, 20, 30,
                       10, 15, 20, 30,
                       10, 15, 20, 30,
                       10, 15, 20, 30])
wns_corr = pd.DataFrame([0.5, 0.5, 0.5, 0.5,
                         0.7, 0.7, 0.7, 0.7,
                         0.8, 0.8, 0.8, 0.8,
                         0.9, 0.9, 0.9, 0.9])
spktrain_corr = pd.concat([pd.DataFrame(np.asarray(st_files)), wns_sd, wns_corr, mean_corr_df], axis=1)
spktrain_corr.columns = column_list
spktrain_corr
mean_corr_list
#save as a csv file
os.chdir('/Users/nate/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/output')
os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/output')
spktrain_corr.to_csv('spktrain_centered_reliabilitycorr_no_ones_galan.txt',sep=',',float_format='%10.6f',index=False,header=True)
len(corr_mat_df_list[0:16])
corr_mat_list_05 = corr_mat_df_list[16:32]
len(corr_mat_list_05)
corr_mat_list_4 = corr_mat_df_list[96:112]
len(corr_mat_list_4)
import seaborn

os.chdir('/home/ngg1/ngglasgow@gmail.com/Data_Urban/NEURON/Analysis/Bhalla1993_wns/figures')

plt.figure()
seaborn.heatmap(corr_mat_list_05[3], vmin=0.0, vmax=0.2)
plt.title('Jitter 0.5 ms; WNS corr. 0.5; 30% SD')
plt.savefig('05jitter05corr30sd_corr.png', dpi=300, format='png')

plt.figure()
seaborn.heatmap(corr_mat_list_05[15], vmin=0.0, vmax=0.2)
plt.title('Jitter 0.5 ms; WNS corr. 0.9; 30% SD')
plt.savefig('05jitter09corr30sd_corr.png', dpi=300, format='png')
#%%
#open saved csv of spike train correlations
spktrain_corr = pd.read_csv("spktrain_corr_cut.txt",sep=',')
#%%
#plot correlations in some logical manner
#plt.plot(spktrain_corr.iloc[:,2:])
corr_list = [0.5,0.7,0.8,0.9]
for i in range(4):
    plt.figure()
    plt.plot(spktrain_corr.iloc[i*4:i*4+4,1],spktrain_corr.iloc[i*4:i*4+4,3:])
    plt.xlabel('WNS %SD')
    plt.ylabel('Correlation Coefficient')
    plt.title('WNS Correlation of ' +str(corr_list[i]))
    plt.legend(('1 ms', '2 ms', '3 ms', '4 ms', '5 ms','6 ms','7 ms','8 ms'),bbox_to_anchor=(1.02,0.02,1,1),loc=3)
    plt.show()



#%%

#convolve kernel with binned spike times
st_kern_list = []
for item in st_kern_list:
    st_kern = scipy.convolve(sqKern[0],item)
    plt.figure()
    plt.plot(st_kern)
    st_kern_list.append(st_kern)
#%%
"""
26Sep2017
NOTE: none of this seems to work at all. Something to do with the
spike times and analog signals not working together.

NEW NOTE: As of now, for unknown reasons, it works perfectly well. Although,
I think i tried it earlier, I think the only difference was that I put a []
around the spiketime values in the spiketime_list for loop. Holy god.

NOTE: Work well to have an empty df obj pd.DataFrame() to fill in, as long as
the object is called each time to be concatanated. The df.apply() function
works by default down through all rows,
     then across columns, axis=1 for across
rows by column.
"""
#sta
#need to read in current signal and spiketrains and link them in neo
#inj = pd.read_csv('wns_i_600_b.dat',sep='\s+',header=None)
i#nj.head(600)

wns_list_list = []
for item in i_list:
    wns_list=[]
    for i in range(1,51):
        wns = item.iloc[:,i]
        wns2 = AnalogSignal(signal=wns.values*pq.nA,
                        sampling_rate=10*pq.kHz,
                        t_start=0*pq.ms)
        wns_list.append(wns2)
    wns_list_list.append(wns_list)
#%%
#stavg = spike_triggered_average(wns_list[0],st_list[0],(-50*pq.ms,10.1*pq.ms))

#Set up a for loop to do an STA for all trials
sta_list = []
for k in range(16):
    sta_ind = pd.DataFrame()                #starting time column to build on
    for i in range(50):                     #adding each columb of sta to df
        sta_i = spike_triggered_average(wns_list_list[k][i],spktrain_list[k][i],(-50*pq.ms,10.1*pq.ms))
        sta_i_df = pd.DataFrame(sta_i.as_array().flatten())
        sta_ind = pd.concat([sta_ind,sta_i_df],axis=1)
    sta_avg = sta_ind.apply(np.mean,axis=1) #average each row across columns
    t_sta = np.arange(-50,10.1,0.1)         #time from -50 to 10 ms inclusive by 0.1
    t_df = pd.DataFrame(t_sta)              #make a time data frame
    sta_list.append(sta_avg)
    plt.figure()
    plt.plot(t_df,sta_avg)                  #easier to call as separate x,y
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
