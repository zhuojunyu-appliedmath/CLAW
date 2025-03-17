import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
import os
import pickle
import glob
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,LabelBinarizer,Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, GroupKFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import scipy as sp
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import scipy.stats as sp_st
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import sys

input_channels = ['CxI_common', 'Cx_left', 'Cx_right', 'dSPN_left', 'dSPN_right',
       'iSPN_left', 'iSPN_right', 'FSI_common', 'GPeP_left',
       'GPeP_right', 'GPeA_left', 'GPeA_right', 'GPi_left', 'GPi_right', 
       'STN_left', 'STN_right', 'Th_left', 'Th_right']

selected_nuc = ['dSPN_left', 'dSPN_right','iSPN_left', 'iSPN_right','GPi_left', 'GPi_right', 'GPeP_left', 'GPeP_right', 'Th_left', 'Th_right']

conf = dict({1.0:"No",0.9:"Low",0.75:"High"})

binned_firing_rates = pd.DataFrame()

bin_size = 10

seeds = [1]  #network indices, ranging from 1 to 300

for seed in seeds:

    print(seed)

    data_dir = r"./Data/network_"+str(seed)+"/" 
    
    reward_q_df = pd.compat.pickle_compat.load(open(data_dir+"reward_q_df.pickle","rb"))
    firing_rate = pd.compat.pickle_compat.load(open(data_dir+"firing_rates.pickle","rb"))
    nw_data = pickle.load(open(data_dir+"network_"+str(seed)+".pickle","rb"))
    
    for i in np.arange(len(reward_q_df)):
        rew = reward_q_df[i]
        fr = firing_rate[i]
        nw = nw_data[i]
        datatabs = nw['datatables']
        
        times = np.array(datatabs["stimulusstarttime"].values).astype(float)
        decision_times = np.array(datatabs["decisiontime"].values).astype(float)
        
        bins = np.arange(0,times[-1],bin_size)
        block = datatabs['correctdecision']
        chosen_action = list(datatabs['decision'].values)
        trial_nums = list(datatabs.index)
        temp = dict()
        temp["bin_num"] = []
        temp["Time(ms)"] = []
        temp["chosen_action"] = []
        temp["seed"] = []
        temp["phase"] = []
        temp["block"] = []
        temp["trial_num"] = []
        for j,dt in enumerate(bins[:-1]):
            temp["bin_num"].append(j)
            temp["Time(ms)"].append(dt)

            ind = np.digitize(dt,times)-1
            ca = chosen_action[ind]
            temp["chosen_action"].append(ca)
            temp["block"].append(block[ind])
            temp["trial_num"].append(trial_nums[ind])
            
            phase = 0 if dt >= times[ind] and dt <= decision_times[ind] else 1 if dt >= decision_times[ind] and dt <= decision_times[ind]+300 else 2
            temp["phase"].append(phase)
            for ip_ch in input_channels:
                dat_slice = fr.loc[(fr["Time (ms)"] >=dt) & (fr["Time (ms)"] <=bins[j+1]) & (fr["variable"] == ip_ch)]
                if ip_ch not in temp.keys():
                    temp[ip_ch] = []
                temp[ip_ch].append(dat_slice["firing_rate"].mean())
            temp["seed"].append(str(seed)+"_"+str(i))
        
        temp_df = pd.DataFrame(temp)
        temp_df["conflict"] = conf[np.unique(fr["conflict"])[0][0]]
        
        binned_firing_rates = pd.concat([binned_firing_rates,temp_df])

binned_firing_rates.to_csv(data_dir+"binned_firing_rates.csv")