#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:39:36 2024

Continuation of 1_pilot_HEP
Performing ICA

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

#  dir where dta preprocessed until ıca is stored
prep_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_until_ICA"
os.chdir(prep_dir)

# name of file
file_name = 'BTSCZ022_V1_eyes-open_prep_until_ICA.fif'

# construct the full file path
file_path = os.path.join(prep_dir, file_name)

# Load the FIF file
prep_data = mne.io.read_raw_fif(file_path, preload=True)


# %%  1. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

# Define the column names for the DF
column_names = ['subject_id', 'session', 'task', 
                'total_ICA_epoch_nr', 'rejected_ICA_epoch_nr', 'percent_rejected_ICA_epoch', 
                'epoch_ICA_labels', 'epoch_ICA_percentage']

# Initialize an empty DF with columns
prep_outputs = pd.DataFrame(columns=column_names)


# %% 8.1 Preparation for ICA: EPOCHING AND CLEANING FOR ICA

# For ICA to perform better we need a highpass filter 1 
# THEN after ICA copy weights back to initiAL preprocessed data with filter .3
prep_data_ica_filtered = prep_data.copy().filter(l_freq=1.0, h_freq=45)  # you can try and see lower lowpass (e.g. 0.1, 0.3,..)for heart artifacts but other components may get worse

# For epoching data for 1 sec to reject  bad epochs
epoch_duration = 1.0  # Define the duration of each epoch in seconds
# Create epochs on the data you prepared
epochs = mne.make_fixed_length_epochs(prep_data_ica_filtered, duration=epoch_duration, preload=True)

### Step 1: Define Criteria

# Copy epochs to prep for ica 
epochs_cleaned = epochs.copy()
# Set Criteria: describe a  bad epoch: CHECK THESE VALS! 
reject_criteria = dict(eeg=1e-4)  #dict(eeg=100*1e-6)  # 100 µV, same as Antonin
# HERE for reject: (excluding channels that reflected eye movements, i.e., EOG channels, Fp1, Fp2, F7, F8) 
flat_criteria = dict(eeg=1e-6)  # 1 µV

# Temporarily set Fp1, Fp2, F7, F8 as eye-chans so that they do not get involved in reject
# Check also without EOG conversion!
temporary_eog_mapping = {
    'Fp1': 'eog',
    'Fp2': 'eog',
    'F7': 'eog',
    'F8': 'eog',
}
epochs_cleaned.set_channel_types(temporary_eog_mapping)

# Drop bad epochs 
epochs_cleaned = epochs_cleaned.drop_bad(reject_criteria, flat_criteria)

# check drop log 
epochs_cleaned.plot_drop_log()
print(epochs_cleaned.drop_log)

# Turn near-eye chans back to eeg chans
back_eeg__mapping = {
    'Fp1': 'eeg',
    'Fp2': 'eeg',
    'F7': 'eeg',
    'F8': 'eeg',
}
epochs_cleaned.set_channel_types(back_eeg__mapping)




# %% 8.2.  DO ICA ON CLEANED EPOCHS

# Do ICA on the epochs we prepared
ica = ICA(
    n_components=27, 
    max_iter="auto", 
    method="infomax", 
    random_state=97,
    fit_params=dict(extended=True)
    ) # initially 30... ICLabel requires extended infomax!
ica.fit(epochs_cleaned)
ica

# print explained vars for ICs
explained_var_ratio = ica.get_explained_variance_ratio(epochs_cleaned)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
    )


# plot ICs on the original data: From here use the original prep data!
prep_data.load_data()
ica.plot_sources(prep_data, show_scrollbars=False)
ica.plot_components()


# Automatically label components using the 'iclabel' method
component_dict = label_components(inst=prep_data, ica=ica, method='iclabel')
# component_labels gives the labels

# Print the results
print("Component Labels:", component_dict['labels'])
print("Predicted Probabilities:", component_dict['y_pred_proba'])

### Change what follows with the correct var names 

# Extract non-brain labels' index to exclude them from original data
# only those labels where algorithm assigns above chance probability to the label, as per Berkan's suggestion
labels = ic_labels["labels"]
exclude_index = [
    index for index, label in enumerate(labels) if label not in ["brain", "other"] and predicted_probabilities[index] > 0.50
]
print(f"Excluding these ICA components: {exclude_index}")

# ADD: CORRELATION WITH ECG signal!!


# Reconstruct the original data without noise components
# ica.apply() changes the Raw object in-place, so let's make a copy first:
prep_ica_data = prep_data.copy()
ica.apply(prep_ica_data, exclude=exclude_index)

# compare ica cleaned and before
prep_data.plot()
prep_ica_data.plot()

# Save the data, yay!
ica_file_name = file_name.replace('_until_', '_')
file_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_ICA_applied', ica_file_name)
prep_ica_data.save(file_path)


"""
extract labels and reconstruct raw data
https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html#sphx-glr-generated-examples-00-iclabel-py

"""

#  baseline correction ??  FIRST TRY THAT AND CHECK MICROVOLTS AGAIN
baseline = (start_time, end_time)  # Define the baseline period
epochs.apply_baseline(baseline=(start_time, end_time))




# %% ARCHIVE

# ### Step 2: Exclude eye Chans from reject criteria

# # Channels to exclude from threshold check (eye movement channels):exclude channels close to the eyes: we want to detect eye movements w ICA 
# exclude_channels = ['Fp1', 'Fp2', 'F7', 'F8']
# epochs_cleaned.drop_channels(exclude_channels)

# """
# # Iterate over epochs to identify epochs with channels other than ['Fp1', 'Fp2', 'F7', 'F8'] exceeding the voltage threshold
# epochs_to_drop = []
# for index in range(len(epochs)):
#     # Check if any channels other than ['Fp1', 'Fp2', 'F7', 'F8'] exceed the voltage threshold
#     epoch = epochs[index] # iterating over all epochs using their index
#     #print(epoch.ch_names)
#     if any(epoch.get_data()[0, epoch.ch_names.index(ch_name)].max() > reject_criteria['eeg'] for ch_name in epoch.ch_names if ch_name not in exclude_channels):
#         print(epoch.ch_names)
#         epochs_to_drop.append(index)



# # Iterate over epochs to identify epochs with channels other than ['Fp1', 'Fp2', 'F7', 'F8'] exceeding the voltage threshold
# epochs_to_drop = []
# for index, epoch in enumerate(epochs):
#     # Check if any channels other than ['Fp1', 'Fp2', 'F7', 'F8'] exceed the voltage threshold
#     if any(epoch[ch_name].max() > reject_criteria['eeg'] for ch_name in epoch.ch_names if ch_name not in exclude_channels):
#         epochs_to_drop.append(index)
# """

# ### Step 3: Apply rejection and flat criteria to epochs of non-eye chans
    
# epochs_cleaned = epochs_cleaned.drop_bad(reject_criteria, flat_criteria)
# # check drop log 
# epochs_cleaned.plot_drop_log()
# print(epochs_cleaned.drop_log)

# ### Step 4: Select only the clean epochs for ICA 

# ica_epochs = epochs_cleaned.copy()


# """
# # lets drop channels where we had to exclude more than 33% of epochs
# total_epoch_nr = epochs.get_data().shape[0]
# # Convert the drop log to binary values (1 if dropped, 0 if not)
# binary_drop_log = [[1 if ch in log else 0 for ch in epochs_cleaned.ch_names] for log in epochs_cleaned.drop_log]

# # Calculate the percentage of epochs dropped for each channel
# channel_drop_percentage = {ch_name: sum(log) / total_epoch_nr * 100 
#                            for ch_name, log in zip(epochs_cleaned.ch_names, zip(*binary_drop_log))}

# # Set the threshold for drop percentage
# threshold_percentage = 33.0

# # Print channel names with drop percentage above the threshold
# bad_channels = []
# for ch_name, percentage in channel_drop_percentage.items():
#     if percentage > threshold_percentage:
#         print(f"Channel {ch_name}: {percentage:.2f}% epochs dropped")
#         bad_channels.append(ch_name)
# """

# # When running the code below something is wrong with the amplitudes below


# """


# #plot data
# # epochs.average().detrend().plot_joint()

# # Extract EEG data
# eeg_data = epochs.get_data()

# # Check the voltage range
# min_voltage = np.min(eeg_data)
# max_voltage = np.max(eeg_data)

# print(f'Minimum Voltage: {min_voltage} microvolts')
# print(f'Maximum Voltage: {max_voltage} microvolts') 


# # Try to reject bad spans (based on non-eye chans) of raw data and then perform ICA
# exclude_channels = ['Fp1', 'Fp2', 'F7', 'F8']
# ica_data_no_eyes = prep_data_ica_filtered.drop_channels(exclude_channels) # maybe delete?

# # reject bad spans based on amplitude
# annotations, bads = annotate_amplitude(
#     ica_data_no_eyes,
#     )

# """
