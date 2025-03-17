import ray
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

seeds = [1]  #network indices, ranging from 1 to 300

for seed in seeds:

    print(seed)
    data_dir = r"./Data/network_"+str(seed)+"/" 
    data_conf = pd.DataFrame()

    binarized_firing_rates = pd.read_csv(data_dir+"binarized_firing_rates.csv")
    binarized_firing_rates = binarized_firing_rates.loc[:,~binarized_firing_rates.columns.str.contains("Unnamed")]
    
    for grp in binarized_firing_rates.groupby("conflict"):
        for i in np.arange(len(grp[1])):
            df = grp[1].iloc[i].copy()
            temp = df[selected_nuc]
            num = int("".join(str(x) for x in df[selected_nuc].astype(int).values), 2)
            num_full = int("".join(str(x) for x in df[input_channels].astype(int).values), 2)
            temp["state_num"] = num
            temp["state_num_full"] = num_full
            temp["chosen_action"] = df["chosen_action"]
            temp["trial_num"] = df["trial_num"]
            temp["bin_num"] = df["bin_num"]
            temp["seed"] = df["seed"]
            temp["phase"] = df["phase"]
            temp["block"] = df["block"]
            data_conf = pd.concat([data_conf,temp],axis=1)
    
    data_conf = data_conf.transpose()
    data_conf.to_csv(data_dir+"data_conf.csv")
