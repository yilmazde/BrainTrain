#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:11:04 2023

@author: denizyilmaz
"""

# %%  0. Import Packages 

import mne
import os
import numpy as np
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
from mne_icalabel import label_components
from autoreject import AutoReject # for rejecting bad channels
from autoreject import get_rejection_threshold  


import matplotlib
matplotlib.use('Qt5Agg')  # You can try different backends (Qt5Agg, TkAgg, etc.)

# %% example code

ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose=True)
ar.fit(epochs[:20])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

# %% 


# C. Detect & interpolate noisy channels
# INITIAL code
raw_resampled_line_reref_interp = raw_resampled_line_reref.copy()
raw_resampled_line_reref_interp.interpolate_bads()
# AUTO_REJECT 
# You have to either mark bads first during recording by clicking on the channel or making a list and then importing it
# Then u can use this info for interp or you have to do an auto detecting with auto_reject
raw_resampled_line_reref_interp = raw_resampled_line_reref.copy()
# do a dummy epoching for channel rejection
reject_epoch_duration = 1.0
# Create epochs
reject_epochs = mne.make_fixed_length_epochs(raw_resampled_line_reref_interp, duration=reject_epoch_duration, preload=True)
# check out epochs
reject_epochs.average().detrend().plot_joint()
# Create auto_reject object
auto_reject = AutoReject(n_interpolate=[1, 2, 3, 4], 
                         random_state=11,
                         n_jobs=1, 
                         verbose=True)
# get cleaned epochs
# clean_epochs = auto_reject.fit_transform(reject_epochs)  
# OR
auto_reject.fit(reject_epochs)  # fit on a few epochs to save time
epochs_ar, reject_log = auto_reject.transform(reject_epochs, return_log=True)
# visualize
reject_epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
reject_log.plot('horizontal', show_names=1)
# get rejection dictionary
reject = get_rejection_threshold(reject_epochs)  
# intrepolate the detected bad channels
raw_resampled_line_reref_interp.interpolate_bads()
