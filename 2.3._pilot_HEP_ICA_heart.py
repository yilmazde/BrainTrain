#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:17:44 2024

@author: denizyilmaz
"""

# %%  0. Import Packages 

import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
from mne_icalabel import label_components
from autoreject import AutoReject # for rejecting bad channels
from autoreject import get_rejection_threshold  
from collections import Counter
from pyprep.find_noisy_channels import NoisyChannels
#from pyprep import PreprocessingPipeline


import matplotlib
# matplotlib.use('TkAgg')  # You can try different backends (Qt5Agg, TkAgg, etc.)


# gives you info on whole packages in mne env
# mne.sys_info()

#  dir where data preprocessed until ıca is stored
prep_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_until_ICA"
os.chdir(prep_dir)

# name of file
file_name = 'BTSCZ022_V1_hct_prep_until_ICA.fif'

# construct the full file path
file_path = os.path.join(prep_dir, file_name)

# Load the FIF file
prep_data = mne.io.read_raw_fif(file_path, preload=True)


# %%  1. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

# Define the column names for the DF
column_names = ['subject_id', 'session', 'task', 
                'total_heart_epoch_nr', 'rejected_heart_epoch_nr', 'percent_rejected_heart_epoch', 
                'heart_ICA_labels', 'heart_ICA_percentage']

# Initialize an empty DF with columns
prep_outputs = pd.DataFrame(columns=column_names)


# %% Epoch around the R-peak, ALTERNATIVE to 1 sec epochs: TRY on heart epochs!

# For ICA to perform better we need a highpass filter 1
prep_heart_ica = prep_data.copy().filter(l_freq=1.0, h_freq=45) 

# Define Heartbeat Events
heartbeat_events, event_id = mne.events_from_annotations(prep_heart_ica) # event_id defines what kind of an event it is
# NEW: Extract only those that are heart-events, not from other annotations!
heartbeat_events = heartbeat_events[heartbeat_events[:, 2] == event_id['R-peak']]
heartbeat_event_id = event_id['R-peak']

# Describe a  bad epoch
reject_criteria = dict(eeg=1e-4)  #dict(eeg=100e-6)  # 100 µV, same as Antonin,(exclude chand that reflected eye movements, i.e.,Fp1, Fp2, F7, F8) 
flat_criteria = dict(eeg=1e-6)  # 1 µV

# Temporarily set Fp1, Fp2, F7, F8 as eye-chans so that they do not get involved in reject
temporary_eog_mapping = {
    'Fp1': 'eog',
    'Fp2': 'eog',
    'F7': 'eog',
    'F8': 'eog',
}
prep_heart_ica.set_channel_types(temporary_eog_mapping)




# Make heart epochs, you can already exclude bad epochs by adding reject argument!
heartbeat_epochs = mne.Epochs(
    prep_heart_ica, heartbeat_events,
    event_id=heartbeat_event_id, tmin=-0.3, tmax=0.8, 
    baseline=None, preload=True, # baseline should be None! see: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
    reject=reject_criteria, flat = flat_criteria, 
    event_repeated='drop'
)

# check drop log 
heartbeat_epochs.plot_drop_log()
print(heartbeat_epochs.drop_log)

# Turn near-eye chans back to eeg chans
back_eeg__mapping = {
    'Fp1': 'eeg',
    'Fp2': 'eeg',
    'F7': 'eeg',
    'F8': 'eeg',
}
heartbeat_epochs.set_channel_types(back_eeg__mapping)



# %%  Run the ICA on Heart-Epochs cleaned above from bad epochs 

# Do ICA on the epochs we prepared
ica = ICA(n_components=27, max_iter="auto", random_state=97) # initially 30 components
ica.fit(heartbeat_epochs)
ica


# print explained vars for ICAs
explained_var_ratio = ica.get_explained_variance_ratio(heartbeat_epochs)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
    )

# plot ICs on the original data: From here use the original prep data!
prep_data.load_data()
ica.plot_sources(prep_data, show_scrollbars=False) # time series 
ica.plot_components(inst= heartbeat_epochs)        # topoplots, inst paramter makes it interactive 
# ica.plot_components(inst= prep_data) does not show the interactive plots???
ica.plot_overlay(prep_data)  # overlay seems almost identical...
ica.plot_properties(prep_data, picks=[0, 1])


# Automatically label components using the 'iclabel' method
component_dict = label_components(inst=raw_ica, ica=ica, method='iclabel')
# component_labels gives the labels

# Print the results
print("Component Labels:", component_dict['labels'])
print("Predicted Probabilities:", component_dict['y_pred_proba'])







heartbeat_evoked = heartbeat_epochs.average()
heartbeat_evoked.plot

# YAY WORKS UNTIL HERE !!!

# OR from inbuilt function

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_resampled, ch_name='ECG', tmin = -0.3, tmax = -)
ecg_evoked = ecg_epochs.average()
ecg_evoked.plot_psd()
ecg_epochs.plot_image(combine="mean")
