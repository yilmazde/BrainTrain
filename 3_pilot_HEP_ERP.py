#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:18:25 2024

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
mne.sys_info()

#  Dir where data preprocessed and ICA cleaned is stored
prep_ica_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_ICA_applied"
os.chdir(prep_ica_dir)

# name of file
prep_ica_file_name = 'BTSCZ022_V1_eyes-open_prep_ICA.fif'

# construct the full file path
file_path = os.path.join(prep_ica_dir, prep_ica_file_name)

# Load the FIF file
prep_ica_data = mne.io.read_raw_fif(file_path, preload=True)


# %% 1. Creating ECG/HEP epochs: mne.preprocessing.create_ecg_epochs

heartbeat_events, event_id = mne.events_from_annotations(prep_ica_data)

heartbeat_epochs = mne.Epochs(
    prep_ica_data, heartbeat_events,
    event_id=event_id, tmin=-0.3, tmax=0.8, 
    baseline=(None, 0), preload=True, event_repeated='drop'
)

heartbeat_evoked = heartbeat_epochs.average()
heartbeat_evoked.plot()

# %% 2. Comparison of Conditions : CHECK FROM HERE ON, you need to import V3 as well to compare!

# Define time window of interest (e.g., 100-200 ms)
tmin, tmax = 0.45, 0.5

# Compute ERP for each condition
condition_1_data = epochs_data['condition_1'].average()
condition_2_data = epochs_data['condition_2'].average()

# Select data within time window of interest
time_mask = (condition_1_data.times >= tmin) & (condition_1_data.times <= tmax)
condition_1_amplitudes = np.mean(condition_1_data.data[:, :, time_mask], axis=2)
condition_2_amplitudes = np.mean(condition_2_data.data[:, :, time_mask], axis=2)

# Perform statistical comparison (e.g., t-test)
t_stat, p_value = stats.ttest_ind(condition_1_amplitudes, condition_2_amplitudes)

# Print results
print("T-statistic:", t_stat)
print("P-value:", p_value)

#### Plot HEPs

# Plot ERP waveforms
plt.figure(figsize=(10, 6))
plt.plot(condition_1_data.times, condition_1_data.data.mean(axis=0), color='blue', label='Condition 1')
plt.plot(condition_2_data.times, condition_2_data.data.mean(axis=0), color='red', label='Condition 2')

# Highlight time window of interest
plt.axvline(x=tmin, color='gray', linestyle='--', label='Time Window of Interest')
plt.axvline(x=tmax, color='gray', linestyle='--')

# Add labels, legend, and title
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('ERP Waveforms for Conditions')

# Show plot
plt.show()


# YAY WORKS UNTIL HERE !!!

# OR from inbuilt function
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_resampled, ch_name='ECG', tmin = -0.3, tmax = -)
ecg_evoked = ecg_epochs.average()
ecg_evoked.plot_psd()
ecg_epochs.plot_image(combine="mean")