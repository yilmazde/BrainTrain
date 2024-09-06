#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:18:25 2024

@author: denizyilmaz

This script is loading preprocessed and ICA-cleaned data and calculating HEPs per session...
Hopefully.

# e.g. if u do manual rejection specify bad epochs
# remove_epochs = [57, 63,64]
# epochs.drop(remove_epochs)
"""

# %%  0. Import Packages & Load Data

import mne
import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
from mne_icalabel import label_components
from autoreject import AutoReject # for rejecting bad channels
from autoreject import get_rejection_threshold  
from collections import Counter
from pyprep.find_noisy_channels import NoisyChannels
#from pyprep import PreprocessingPipeline

# matplotlib.use('TkAgg')  # You can try different backends (Qt5Agg, TkAgg, etc.)

# gives you info on whole packages in mne env
# mne.sys_info()

#  Dir where data preprocessed and ICA cleaned is stored
prep_ica_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_ICA_applied_on_raw"
os.chdir(prep_ica_dir)

# name of file
prep_ica_file_name_v1 = 'BTSCZ011_V1_eyes-closed_prep_ICA.fif'
#prep_ica_file_name_v3= 'BTSCZ011_V3_hct_prep_ICA.fif'


# construct the full file path
file_path_v1 = os.path.join(prep_ica_dir, prep_ica_file_name_v1)
#file_path_v3 = os.path.join(prep_ica_dir, prep_ica_file_name_v3)

# Load the FIF file
prep_ica_data_v1 = mne.io.read_raw_fif(file_path_v1, preload=True)
#prep_ica_data_v3 = mne.io.read_raw_fif(file_path_v3, preload=True)

# inspect annotations
prep_ica_data_v1.annotations.description
np.unique(prep_ica_data_v1.annotations.description, return_counts=True)

# Filter out irrelevant annotations (you need to define which ones are irrelevant)
irrelevant_annotations = ['Comment/actiCAP Data On', 'New Segment/']  # Adjust this list as needed

# find indices of irrelevant annots
irrelevant_indices = np.where(np.isin(prep_ica_data_v1.annotations.description, irrelevant_annotations))[0]

# delete irrelevant ones
prep_ica_data_v1.annotations.delete(irrelevant_indices)

# check whether it worked
prep_ica_data_v1.annotations.description
print("Remaining annotations:", np.unique(prep_ica_data_v1.annotations.description, return_counts=True))


"""

prep_ica_data_v1.annotations
prep_ica_data_v1.annotations.onset
prep_ica_data_v1.annotations.duration


prep_ica_data_v3.annotations.delete(0)
prep_ica_data_v3.annotations.delete(0)


prep_ica_data_v3.annotations
prep_ica_data_v3.annotations.onset
prep_ica_data_v3.annotations.duration


# describe and plot
prep_ica_data.describe()
prep_ica_data.plot_sensors(show_names= True)
prep_ica_data.plot_psd


prep_ica_data.annotations
prep_ica_data.annotations.onset
prep_ica_data.annotations.duration
"""



# %% 1. Creating HEP evoked: mne.preprocessing.create_ecg_epochs

### V1: this works for non-HCT tasks!

heartbeat_events_v1, event_id_v1 = mne.events_from_annotations(prep_ica_data_v1)

baseline = (-0.2, -0.1)
heartbeat_epochs_v1 = mne.Epochs(
    prep_ica_data_v1, heartbeat_events_v1,
    event_id=event_id_v1, tmin=-0.3, tmax=0.8, 
    baseline=baseline, preload=True, event_repeated='drop'
)

# plot epochs
heartbeat_epochs_v1.plot(events = heartbeat_events_v1, event_id=event_id_v1)

# drop bad epochs
heartbeat_epochs_v1.drop_bad(reject=dict(eeg=150e-6))


# create evoked
heartbeat_evoked_v1 = heartbeat_epochs_v1.average()

# plot evoked
# heartbeat_evoked.plot(gfp=True)
heartbeat_evoked_v1.plot_joint().show()
heartbeat_evoked_v1.plot()
heartbeat_evoked_v1.plot_psd()
heartbeat_evoked_v1.plot(picks=['F4'])
heartbeat_evoked_v1.plot(picks=['F8'])
heartbeat_evoked_v1.plot(picks=['Fp2'])
heartbeat_evoked_v1.plot(picks=['Pz'])
heartbeat_evoked_v1.plot(picks=['T7'])


# Extract relevant variables and values....


# we are interested in a time window from 450 to 500 ms
time_window = (0.45, 0.50)  # in seconds

# Select the time indices for this range
mask = (heartbeat_evoked_v1.times >= time_window[0]) & (heartbeat_evoked_v1.times <= time_window[1])

# Extract data for all channels within this time window, then average across the time window
mean_amplitudes = heartbeat_evoked_v1.data[:, mask].mean(axis=1)

# get peak values

evoked_times = heartbeat_evoked_v1.times

# max
max_amplitudes = heartbeat_evoked_v1.data[:, mask].max(axis=1)
max_indices = heartbeat_evoked_v1.data[:, mask].argmax(axis=1)
max_latencies = evoked_times[mask][max_indices]

# min
min_amplitudes = heartbeat_evoked_v1.data[:, mask].min(axis=1)
min_indices = evoked.data[:, mask].argmin(axis=1)
min_latencies = evoked.times[mask][min_indices]

# You need to add channel names too, to be able to match mean amps to chans...
channels = heartbeat_evoked_v1.ch_names

# Focus on specific channels, e.g., 'Fz', 'Cz', 'Pz'
# Find indices of the channels
channel_indices = [heartbeat_evoked_v1.ch_names.index(ch) for ch in ['Fp2', 'F4', 'F8']]
selected_mean_amplitude_data = mean_amplitudes[channel_indices]

Fp2_mean_amplitude = selected_mean_amplitude_data[0]
F4_mean_amplitude = selected_mean_amplitude_data[1]
F8_mean_amplitude = selected_mean_amplitude_data[2]


#### Save !
hep_file_name = "BTSCZ011_V1_eyes-closed_hep-ave.fif"
#file_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/HEPs', hep_file_name)
file_path = os.path.join('/Users/denizyilmaz/Desktop', hep_file_name)


mne.write_evokeds(file_path, heartbeat_evoked_v1, on_mismatch='raise', overwrite=False,)   # evokedEvoked instance, or list of Evoked instance; to load it back: evokeds_list = mne.read_evokeds(evk_file, verbose=False)



"""
Continue here....
"""


### V3

heartbeat_events_v3, event_id_v3 = mne.events_from_annotations(prep_ica_data_v3)

heartbeat_epochs_v3 = mne.Epochs(
    prep_ica_data_v3, heartbeat_events_v3,
    event_id=event_id_v3, tmin=-0.3, tmax=0.8, 
    baseline=baseline, preload=True, event_repeated='drop'
)

# plot epochs
# heartbeat_epochs.plot(events = heartbeat_events, event_id=event_id)

# drop bad epochs
heartbeat_epochs_v3.drop_bad(reject=dict(eeg=150e-6))


# create evoked
heartbeat_evoked_v3 = heartbeat_epochs_v3.average()

# plot evoked
# heartbeat_evoked.plot(gfp=True)
heartbeat_evoked_v3.plot_joint()
heartbeat_evoked_v3.plot()
heartbeat_evoked_v3.plot(picks=['F4'])
heartbeat_evoked_v3.plot(picks=['F8'])
heartbeat_evoked_v3.plot(picks=['Fp2'])
heartbeat_evoked_v3.plot(picks=['Pz'])
heartbeat_evoked_v3.plot(picks=['T7'])

# %% 2. Comparison of Conditions : Visualization.....  These may rather belong to script 4_pilot_HEP_stats  !!!

# Organize the evoked objects into a dictionary
evokeds = {
    'V1': heartbeat_evoked_v1,
    'V3': heartbeat_evoked_v3
}


# Plot comparison for the 'F4' electrode
mne.viz.plot_compare_evokeds(evokeds, picks='F4')

# difference plotted on topo
hep_diff = mne.combine_evoked([heartbeat_evoked_v1, heartbeat_evoked_v3], weights=[1, -1])  # or -1, 1
hep_diff.pick(picks="eeg").plot_topo(color="r")

### Or plot with the time window highlighted ##### 
##################################################


# Plot comparison for the 'F4' electrode with highlighted time window
plt.figure(figsize=(10, 6))
colors = {'V1': 'blue', 'V3': 'red'}  # Define colors for each condition
for session, evoked in evokeds.items():
    plt.plot(evoked.times, evoked.data[evoked.ch_names.index('F4'), :], color=colors[session], label=session)

# Highlight time window of interest
tmin, tmax = 0.45, 0.5
plt.axvline(x=tmin, color='gray', linestyle='--', label='Time Window of Interest')
plt.axvline(x=tmax, color='gray', linestyle='--')

# Add labels, legend, and title
plt.xlabel('Time (s)')
plt.ylabel('Mean Amplitude (µV)')
plt.legend()
plt.title('Comparison of ERP Waveforms Across Visits V1 and V3 for electrode F4 ')

# Add axes lines for x=0 and y=0
plt.axhline(y=0, color='black', linewidth=0.5)  # Horizontal line at y=0
plt.axvline(x=0, color='black', linewidth=0.5)  # Vertical line at x=0

# Show plot
plt.show()


# %% 3.  Stats Comparison of Conditions 

# Find the indices corresponding to the time window
tmin, tmax = 0.45, 0.5

# Select data within the time window of interest
time_mask_v1 = (heartbeat_evoked_v1.times >= tmin) & (heartbeat_evoked_v1.times <= tmax)
time_mask_v3 = (heartbeat_evoked_v3.times >= tmin) & (heartbeat_evoked_v3.times <= tmax)

v1_amplitudes = heartbeat_evoked_v1.data[:, time_mask_v1]
v3_amplitudes = heartbeat_evoked_v3.data[:, time_mask_v3]


# Perform statistical comparison (e.g., t-test)
t_stat, p_value = stats.ttest_ind(v1_amplitudes, v3_amplitudes)

# Print results
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Check for significance
if p_value < 0.05:
    print("The difference between conditions is significant.")
else:
    print("There is no significant difference between conditions.")
    
### but now i have 13 values.... 13 time points? why.....

# Check for significance
if any(p_value < 0.05):
    print("At least one electrode shows a significant difference between conditions.")
else:
    print("No electrode shows a significant difference between conditions.")


###  delete this bit?? 

# Assuming you're interested in a specific channel, e.g., 'F4'
ch_idx_v1 = heartbeat_evoked_v1.ch_names.index('F4')
ch_idx_v3 = heartbeat_evoked_v3.ch_names.index('F4')

v1_amplitudes = np.mean(heartbeat_evoked_v1.data[ch_idx_v1, time_mask_v1])
v3_amplitudes = np.mean(heartbeat_evoked_v3.data[ch_idx_v3, time_mask_v3])

#  OR Select data within time window of interest
v1_amplitudes = np.mean(heartbeat_evoked_v1.data[:, time_mask], axis=1)
v3_amplitudes = np.mean(heartbeat_evoked_v3.data[:, time_mask], axis=1)

######



# VISUALIZE !! 

# Let's focus on channel Pz and visualize both conditions in one plot:
# bc we are interested in difference

mne.viz.plot_compare_evokeds(dict(rare=rare, frequent=frequent), picks="Pz")

# This shows a nice P300 for the rare condition. To improve this visualization further, we could add a 95% confidence interval around each evoked time course:
mne.viz.plot_compare_evokeds(
    dict(rare=list(epochs["x"].iter_evoked()), frequent=list(epochs["o"].iter_evoked())),
    picks="Pz"
)


### Example to correlate HEP and beh

# pip install pandas scipy statsmodels
 

# Assuming 'heartbeat_evoked' is your MNE Evoked object
# Let's say you are interested in a time window from 250 to 350 ms
time_window = (0.45, 0.50)  # in seconds
# Select the time indices for this range
mask = (heartbeat_evoked.times >= time_window[0]) & (heartbeat_evoked.times <= time_window[1])

# Extract data for all channels within this time window, then average across the time window
# This gives you a single value per channel
amplitude_data = heartbeat_evoked.data[:, mask].mean(axis=1)

# Optionally, you might only want to focus on specific channels, e.g., 'Fz', 'Cz', 'Pz'
# Find indices of the channels
channel_indices = [heartbeat_evoked.ch_names.index(ch) for ch in ['Fz', 'Cz', 'Pz']]
selected_amplitude_data = amplitude_data[channel_indices]

# Create a DataFrame for analysis
df = pd.DataFrame({
    'Fz_amplitude': [selected_amplitude_data[0]],
    'Cz_amplitude': [selected_amplitude_data[1]],
    'Pz_amplitude': [selected_amplitude_data[2]],
    'symptoms': [your_symptom_score]  # Make sure to replace `your_symptom_score` with actual data
})


##### ARCHIVE #####
# OR from inbuilt function
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_resampled, ch_name='ECG', tmin = -0.3, tmax = -)
ecg_evoked = ecg_epochs.average()
ecg_evoked.plot_psd()
ecg_epochs.plot_image(combine="mean")