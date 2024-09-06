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
#matplotlib.use('TkAgg')  # You can try different backends (Qt5Agg, TkAgg, etc.)


# gives you info on whole packages in mne env
mne.sys_info()

# pilot analysis dir
pilot_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data"

#pilot_dir = "/Users/denizyilmaz/Desktop/BrainTrain/pilot_analysis/BrainTrain_pilot_data_Michelle"
os.chdir(pilot_dir)

# %% 1. Import data
"""
.eeg: This file contains the actual EEG data. It's the raw EEG signal data you want to analyze.

.vhdr: This is the header file, and it contains metadata and information about the EEG recording, such as channel names, sampling rate, and electrode locations. This file is essential for interpreting the EEG data correctly.

.vmrk: This file contains event markers or annotations that correspond to events in the EEG data. Event markers can be used to mark the timing of specific events or stimuli during the EEG recording. This file is useful for event-related analyses.
"""

# I run this to get the name of my file I want to read
os.listdir()

# read data that you already put in your wd
#raw = mne.io.read_raw("BTSCZ008_V1_eyes-open.vhdr", preload=True)
raw = mne.io.read_raw("BTSCZ021_V1_eyes-open.vhdr", preload=True)


# alternative way of reading data
# raw = mne.io.read_raw_brainvision("BTSCZ009_V1_eyes-open.vhdr", preload=True, scale=1)
# read_raw_brainvision(scale=1e6)

# inspect data visually
raw.plot()

"""
You can use raw.apply_function() for this purpose. 
MNE expects the unit of EEG signals to be V, so because your data is presumably in µV you could convert it like this: 
raw.apply_function(lambda x: x * 1e-6)
https://mne.discourse.group/t/rescale-data-import-from-fieldtrip/3402
"""

# %%  2. Get the relevant info of data

# you can get specific info by calling an attribute like a dict
print(raw.info["sfreq"])  # u get sampling frequency
print(raw.info["bads"])  # u get the bad channels IF marked beforehand

"""
# first run an algoritm to detect bads:
# works only on epochs!? should I do this on epochs Im so confuseddddd
# aNYWAY check the MNE post u made
raw_check_bads = raw.copy()

auto_reject = AutoReject()
raw_no_bads = auto_reject.fit_transform(raw_check_bads)  

"""

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

# some channels are named according to the old nomenclature, change them
# Create a dictionary to map old channel names to new channel names
channel_renaming = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8',
}

# Rename the channels using rename_channels
raw.rename_channels(channel_renaming)

# Define the channel type mapping
channel_type_mapping = {
    'ECG': 'ecg',
    'RESP': 'resp'
}

# Set the channel type for the "ECG" channel
raw.set_channel_types(channel_type_mapping)

# Load the easycap-M1 montage (check if its truw by checking the cap)
montage = mne.channels.make_standard_montage('easycap-M1')

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

# %% 5. Find ECG events & add annotations: some r peaks not detected, reduce treshold &try again!

# find ECG events
ecg_events,_,_ = mne.preprocessing.find_ecg_events(raw_resampled, ch_name='ECG')
# play w the detection treshold and see it performs better!


# Create annotations time-locked to R-peak with event descriptions
descriptions = ['R-peak'] * len(ecg_events)
R_peak_annotations = mne.Annotations(onset=ecg_events[:, 0] / raw_resampled.info['sfreq'],  # Convert sample indices to seconds
                              duration=0.0,  # Set the duration to zero for point events
                              description=descriptions)

# Define existing annotations
hct_annotations = raw.annotations.copy()

# Combine initial and R-peaks annotations
combined_annotations = mne.Annotations(
    onset=np.concatenate((hct_annotations.onset, R_peak_annotations.onset)),
    duration=np.concatenate((hct_annotations.duration, R_peak_annotations.duration)),
    description=np.concatenate((hct_annotations.description, R_peak_annotations.description))
)

# Apply annotations to raw data
raw_resampled.set_annotations(combined_annotations)

# Plot ECG with Annotations
raw_resampled.plot(n_channels=1, scalings={'ECG': 1e-3})
raw_resampled.plot(n_channels=1)

                
# YAY WORKS WELL UNTIL HERE!

# ecg_events.plot_image(combine="mean")

# %% 6. Separate the EEG from ECG & RESP data

# Create a copy of the original RawBrainVision object
raw_eeg = raw_resampled.copy()

# Remove the ECG channel
raw_eeg.drop_channels(['ECG', 'RESP'])


# %% 7. Preprocess the EEG data: DO NOT INCLUDE the ECG ND RESP channel here !!

# A. Remove line noise
# Apply notch filter to remove line noise (e.g., 50 Hz from Antonin's manuscript)
line_freq = 50  # Set the line frequency to 50, as Antonin did: 
raw_resampled_line= raw_eeg.copy()
raw_resampled_line.notch_filter(freqs=line_freq)  # Apply notch filter to EEG channels only ?? OR: 49.5 to 50.5 in a method ??
# Plot the data to visualize the effect of the notch filter
raw_resampled_line.plot_psd()

# B.Robust average rereferencing
raw_resampled_line_reref = raw_resampled_line.copy()
raw_resampled_line_reref.set_eeg_reference(ref_channels='average')
raw_resampled_line_reref.plot()

# C. Detect & interpolate noisy channels
raw_resampled_line_reref_interp = raw_resampled_line_reref.copy()
# Assign the mne object to the NoisyChannels class. The resulting object will be the place where all following methods are performed.
noisy_data = NoisyChannels(raw_resampled_line_reref_interp, random_state=1337)
# find bad by corr
noisy_data.find_bad_by_correlation()
print(noisy_data.bad_by_correlation)
# find bad by deviation
noisy_data.find_bad_by_deviation()
print(noisy_data.bad_by_deviation)
#find bad by ransac: finds nothing, do I first have to mark bads from the methods before??
noisy_data.find_bad_by_ransac(channel_wise=True, max_chunk_size=1) 
print(noisy_data.bad_by_ransac)
# get channel names marked as bad and assign them into bads of the data from the step before
raw_resampled_line_reref_interp.info["bads"] = noisy_data.get_bads()
# Interpolate noisy Channels
raw_resampled_line_reref_interp.interpolate_bads()

# D. Bandpass filter [0.3  45]: Do this before all other steps?
raw_resampled_line_reref_interp_filt = raw_resampled_line_reref_interp.copy()
# Define the bandpass filter frequency range
low_freq = 0.3  # Lower cutoff frequency (in Hz)
#low_freq = 1
high_freq = 45.0  # Upper cutoff frequency (in Hz)
# Apply the bandpass filter
raw_resampled_line_reref_interp_filt.filter(l_freq=low_freq, h_freq=high_freq, method='fir', phase='zero') # check method and phase


# E. SAVE the Preprocessed Data

# Replace 'filename' with the desired file name (without extension)
filename = os.path.splitext(os.path.basename(raw.filenames[0]))[0]   # Specify the desired file name (without extension)
folder_name = 'Preprocessed_until_ICA'  # Specify the folder name

# Construct the full file path
file_path = os.path.join(os.getcwd(), folder_name, filename + '_prep_until_ICA.fif')

# Save the raw data
#raw_resampled_line_reref_interp_filt.save(file_path, overwrite=True)




# From here on, transfer to SCRIPT 2
 
# %% 8.1 EPOCHING FOR ICA : TRY ON (HEART?) EPOCHS ! Antonin did on 1 sec epochs, not around the heart

# for ICA to perform better we need filter 1, THEN copy weights back to filter .3
raw_ica = raw_resampled_line_reref_interp_filt.copy().filter(l_freq=1.0, h_freq=45) # you can try and see lower lowpass (e.g. 0.1, 0.3,..)for heart artifacts but other components may get worse

# Channels to exclude from threshold check (eye movement channels, close to eyes): we will anyway want to detect eye movements w ICA 
exclude_channels = ['Fp1', 'Fp2', 'F7', 'F8']    #include_channels = mne.pick_channels(epochs.ch_names, include = [], exclude=exclude_channels)
raw_ica_no_eyes = raw_ica.drop_channels(exclude_channels)

# epoch data for 1 sec to reject  bad epochs => define the duration of each epoch in seconds
epoch_duration = 1.0
# Create epochs on the data you prepared
epochs = mne.make_fixed_length_epochs(raw_ica_no_eyes, duration=epoch_duration, preload=True)

# describe a  bad epoch: CHECK THESE VALS! 
reject_criteria = dict(eeg=100e-6)  # 100 µV, same as Antonin
# HERE for reject: (excluding channels that reflected eye movements, i.e., EOG channels, Fp1, Fp2, F7, F8) 
flat_criteria = dict(eeg=1e-6)  # 1 µV
# reject bad epochs
epochs_cleaned = epochs.copy().drop_bad(reject_criteria, flat_criteria)
epochs_cleaned.plot_drop_log()

# print(epochs.drop_log) # shows none
print(epochs_cleaned.drop_log) # shows the dopped ones

# CONTINUE HERE!!!
# Get the indices of the dropped epochs without eye & noisy channels 
# THEN drop these epochs from the data prepared for ICA 
# THEN run ICA !!!

# Access the drop log for EEG channels
eeg_drop_log = epochs_cleaned.drop_log
# Get the indices of the dropped epochs without eye channels
dropped_epochs_indices = [idx for idx_list in eeg_drop_log.values() for idx in idx_list]
non_empty_indices = [idx for idx, elem in enumerate(eeg_drop_log) if elem is not None and elem != '']


# Then, drop these epochs from the data prepared for ICA
epochs_for_ICA = epochs_no_eyes.drop(dropped_epochs_indices)

# Now, you can proceed with the ICA decomposition on the cleaned epochs
# For example:
from mne.preprocessing import ICA

ica = ICA(n_components=20, method='fastica')
ica.fit(epochs_for_ICA)

# Continue with ICA decomposition and further analysis as needed...

"""

IF you want to exclude channels based on % dropped epochs..

# lets drop channels where we had to exclude more than 33% of epochs
total_epoch_nr = epochs.get_data().shape[0]
# Convert the drop log to binary values (1 if dropped, 0 if not)
binary_drop_log = [[1 if ch in log else 0 for ch in epochs_cleaned.ch_names] for log in epochs_cleaned.drop_log]

# Calculate the percentage of epochs dropped for each channel
channel_drop_percentage = {ch_name: sum(log) / total_epoch_nr * 100 
                           for ch_name, log in zip(epochs_cleaned.ch_names, zip(*binary_drop_log))}

# Set the threshold for drop percentage
threshold_percentage = 33.0

# Print channel names with drop percentage above the threshold
bad_channels = []
for ch_name, percentage in channel_drop_percentage.items():
    if percentage > threshold_percentage:
        print(f"Channel {ch_name}: {percentage:.2f}% epochs dropped")
        bad_channels.append(ch_name)

"""



#plot data
epochs_cleaned.plot(n_epochs=len(epochs_cleaned), title='All Cleaned Epochs')
epochs.average().detrend().plot_joint()

# Extract EEG data
eeg_data = epochs.get_data()
# OR eeg_data = epochs_cleaned.get_data()
eeg_data = epochs_cleaned.get_data()

# Check the voltage range
min_voltage = np.min(eeg_data)
max_voltage = np.max(eeg_data)

print(f'Minimum Voltage: {min_voltage} microvolts')
print(f'Maximum Voltage: {max_voltage} microvolts')



# %% 8.2.  DO ICA ON CLEANED EPOCHS




# ICA on continous data !

# for ICA to perform better we need filter 1, THEN copy weights back to filter .3
raw_ica = raw_resampled_line_reref_interp_filt.copy().filter(l_freq=1.0, h_freq=None)
# you can try and see lower for heart artifacts but other components may get worse
#raw_ica = raw_resampled_line_reref_interp_filt.copy().filter(l_freq=0.1, h_freq=None)


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

"""
extract labels and reconstruct raw data
https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html#sphx-glr-generated-examples-00-iclabel-py

"""

# NOW time to take the weights from the ICA and put it on the correctly preprocessed data!
# HOW?
# based on probability usually .5 


# WHAT TO DO ABOUT HUGE OFFFSETS?

#  baseline correction ??  FIRST TRY THAT AND CHECK MICROVOLTS AGAIN
baseline = (start_time, end_time)  # Define the baseline period
epochs.apply_baseline(baseline=(start_time, end_time))

# %% Creating ECG/HEP epochs: mne.preprocessing.create_ecg_epochs

heartbeat_events, event_id = mne.events_from_annotations(raw_resampled)

heartbeat_epochs = mne.Epochs(
    raw_resampled_line_reref_interp_filt, heartbeat_events,
    event_id=event_id, tmin=-0.3, tmax=0.8, 
    baseline=(None, 0), preload=True, event_repeated='drop'
)

heartbeat_evoked = heartbeat_epochs.average()
heartbeat_evoked.plot

# YAY WORKS UNTIL HERE !!!

# OR from inbuilt function

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_resampled, ch_name='ECG', tmin = -0.3, tmax = -)
ecg_evoked = ecg_epochs.average()
ecg_evoked.plot_psd()
ecg_epochs.plot_image(combine="mean")


# %%% ARCHIVE LOOLL

# finding channels with > 33% epochs dropped: DID NOT WORK !!!!
epoch_duration = 1.0
# Create epochs on the data you prepared
epochs = mne.make_fixed_length_epochs(raw_resampled_line_reref_interp, duration=epoch_duration, preload=True)
# describe a  bad epoch
reject_criteria = dict(eeg=100e-6)  # 100 µV, same as Antonin
flat_criteria = dict(eeg=1e-6)  # 1 µV
# reject bad epochs
epochs_cleaned = epochs.copy().drop_bad(reject_criteria, flat_criteria)
epochs_cleaned.plot_drop_log()
print(epochs_cleaned.drop_log) # or epochs_cleaned?




# lets drop channels where we had to exclude more than 33% of epochs
total_epoch_nr = epochs.get_data().shape[0]
# Convert the drop log to binary values (1 if dropped, 0 if not)
binary_drop_log = [[1 if ch in log else 0 for ch in epochs_cleaned.ch_names] for log in epochs_cleaned.drop_log]

# Calculate the percentage of epochs dropped for each channel
channel_drop_percentage = {ch_name: sum(log) / total_epoch_nr * 100 
                           for ch_name, log in zip(epochs_cleaned.ch_names, zip(*binary_drop_log))}

# Set the threshold for drop percentage
threshold_percentage = 33.0

# Print channel names with drop percentage above the threshold
bad_channels = []
for ch_name, percentage in channel_drop_percentage.items():
    if percentage > threshold_percentage:
        print(f"Channel {ch_name}: {percentage:.2f}% epochs dropped")
        bad_channels.append(ch_name)


