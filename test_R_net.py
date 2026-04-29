# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

useful link: https://cbrnr.github.io/blog/importing-eeg-data/

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
from collections import Counter
from pyprep.find_noisy_channels import NoisyChannels
#from pyprep import PreprocessingPipeline
import matplotlib


# pilot analysis dir
pilot_dir = '/Users/denizyilmaz/Desktop/eeg_test_data/RNet and Slim Test'

#pilot_dir = "/Users/denizyilmaz/Desktop/BrainTrain/pilot_analysis/BrainTrain_pilot_data_Michelle"
os.chdir(pilot_dir)

# %% 1. Import data

# I run this to get the name of my file I want to read
os.listdir()

# read data that you already put in your wd
#raw = mne.io.read_raw("BTSCZ008_V1_eyes-open.vhdr", preload=True)
raw = mne.io.read_raw("test_r_net_eo.vhdr", preload=True)

# inspect data visually
raw.plot()

# %%  2. Get the relevant info of data

# you can get specific info by calling an attribute like a dict
print(raw.info["sfreq"])  # u get sampling frequency
print(raw.info["bads"])  # u get the bad channels IF marked beforehand

# plot power spectral density
raw.plot_psd()

# describe raw data
raw.describe()
print(type(raw._data))
print(raw._data.shape)
#raw_data = raw.get_data()
#print(raw_data.shape)

# %% 3. Montage

# check our channel names
print(raw.ch_names)

# Load the montage (check if its truw by checking the cap)
montage = mne.channels.make_standard_montage('standard_1020')

# Apply the montage to your raw data
raw.set_montage(montage)

# plot montage
raw.plot_sensors(show_names=True)


# %% 4. Resampling to 250 Hz, by 2 bc the initial sampling is 500

# Define the new sampling rate you want
new_sampling_rate = 250 

# Resample the EEG data to the new sampling rate
raw_resampled = raw.copy()
raw_resampled.resample(sfreq=new_sampling_rate)

print(raw_resampled.info["sfreq"])

# %% 7. Preprocess the EEG data: DO NOT INCLUDE the ECG ND RESP channel here !!

# A. Remove line noise
# Apply notch filter to remove line noise (e.g., 50 Hz from Antonin's manuscript)
line_freq = 50  # Set the line frequency to 50, as Antonin did: 
raw_resampled_line= raw_resampled.copy()
raw_resampled_line.notch_filter(freqs=line_freq)  # Apply notch filter to EEG channels only ?? OR: 49.5 to 50.5 in a method ??
# Plot the data to visualize the effect of the notch filter
raw_resampled_line.plot_psd()

# B.Robust average rereferencing
raw_resampled_line_reref = raw_resampled_line.copy()
raw_resampled_line_reref.set_eeg_reference(ref_channels='average')
raw_resampled_line_reref.plot()
raw_resampled_line_reref.plot_psd()


# C. Detect & interpolate noisy channels
raw_resampled_line_reref_interp = raw_resampled_line_reref.copy()
# Assign the mne object to the NoisyChannels class. The resulting object will be the place where all following methods are performed.
noisy_data = NoisyChannels(raw_resampled_line_reref, random_state=1337)
# find bad by corr
noisy_data.find_bad_by_correlation()
print("Bad channels by correlation:", noisy_data.bad_by_correlation)
# find bad by deviation
noisy_data.find_bad_by_deviation()
print("Bad channels by deviation:", noisy_data.bad_by_deviation)
#find bad by ransac: finds nothing, do I first have to mark bads from the methods before?  acc to Bigdely-Shamlo paper most bads are found in corr and dev
noisy_data.find_bad_by_ransac(channel_wise=True, max_chunk_size=1) 
print("Bad channels by RANSAC:", noisy_data.bad_by_ransac)

# get channel names marked as bad and assign them into bads of the data from the step before
raw_resampled_line_reref_interp.info["bads"] = noisy_data.get_bads()
bads = noisy_data.get_bads() 
# Interpolate noisy Channels
raw_resampled_line_reref_interp.interpolate_bads()
# plot psd
raw_resampled_line_reref_interp.plot_psd()

# D. Bandpass filter [0.3  45]: Do this before all other steps?
raw_resampled_line_reref_interp_filt = raw_resampled_line_reref_interp.copy()
# Define the bandpass filter frequency range
low_freq = 0.3  # Lower cutoff frequency (in Hz)
#low_freq = 1
high_freq = 45.0  # Upper cutoff frequency (in Hz)
# Apply the bandpass filter
raw_resampled_line_reref_interp_filt.filter(l_freq=low_freq, h_freq=high_freq, method='fir', phase='zero') # check method and phase
# plot psd
raw_resampled_line_reref_interp_filt.plot_psd()


# %% 8.2.  DO ICA
# for ICA to perform better we need filter 1, THEN copy weights back to filter .3
raw_ica = raw_resampled_line_reref_interp_filt.copy().filter(l_freq=1.0, h_freq=None)

# set up and fit the ICA
ica = ICA(n_components=27, max_iter="auto", random_state=97) # initially 30
ica.fit(raw_ica)
ica

# print explained vars for ICAs
explained_var_ratio = ica.get_explained_variance_ratio(raw_ica)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
    )

# plot ICs
raw_ica.load_data()
ica.plot_sources(raw_ica, show_scrollbars=False)
ica.plot_components()

# Automatically label components using the 'iclabel' method
component_dict = label_components(inst=raw_ica, ica=ica, method='iclabel')
# component_labels gives the labels

# Print the results
print("Predicted Probabilities:", component_dict['y_pred_proba'])
print("Component Labels:", component_dict['labels'])

# Extract the labels and probabilities
labels = component_dict['labels']
probabilities = component_dict['y_pred_proba']

# Identify components to exclude (not labeled as 'brain' with probability >= 0.5)
exclude_components = [
    idx for idx, (label, prob) in enumerate(zip(labels, probabilities))
    if label != 'brain' or prob < 0.5
]

# Print excluded components for verification
print(f"Excluding components: {exclude_components}")

# Remove the identified components from the preprocessed data
ica.exclude = exclude_components
raw_cleaned = ica.apply(raw_resampled_line_reref_interp_filt.copy())

# Plot the cleaned data
raw_cleaned.plot()

# Plot data before ICA cleaning
raw_resampled_line_reref_interp_filt.plot(title="Before ICA Cleaning")

# Plot data after ICA cleaning
raw_cleaned.plot(title="After ICA Cleaning")

# Compare PSD before and after ICA cleaning
raw_resampled_line_reref_interp_filt.plot_psd()
raw_cleaned.plot_psd()

# overlay plot to compare before and after ICA cleaning
ica.plot_overlay(raw_resampled_line_reref_interp_filt, exclude=ica.exclude)

