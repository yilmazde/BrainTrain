#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 09:43:47 2024

@author: denizyilmaz
"""

# %%  0. Import Packages 

import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
import neurokit2 as nk
import matplotlib
from collections import Counter
import pandas as pd



# from mne_icalabel import label_components
# from autoreject import AutoReject # for rejecting bad channels
# from autoreject import get_rejection_threshold  
# from pyprep.find_noisy_channels import NoisyChannels
# from pyprep import PreprocessingPipeline
# matplotlib.use('TkAgg')  # You can try different backends (Qt5Agg, TkAgg, etc.)


# gives you info on whole packages in mne env
mne.sys_info()

# pilot analysis dir
# pilot_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data"
pilot_dir = '/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/'

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
raw = mne.io.read_raw("BHC010_eyes-closed.vhdr", preload=True)


### OR you can directly downsample EEG then extract ECG

# Resample the EEG data to the new sampling rate
raw_resampled = raw.copy()
raw_resampled.resample(sfreq=new_sfreq)
print(raw_resampled.info["sfreq"])




#%% Extract ecg parameters: HR, HRV (RMSSD),  Amplitude of R wave,  QT interval, QTc interval,


# Extract ECG signal (assuming the channel name is 'ECG')
ecg_data = raw.copy().pick_channels(['ECG']).get_data()[0]
sfreq = raw.info['sfreq']  # Sampling frequency
new_sfreq = 250

# Convert the ECG signal to a NeuroKit2-compatible format
ecg_downsampled = nk.signal_resample(ecg_data, sampling_rate=sfreq, desired_sampling_rate=new_sfreq) 
time_vector = np.arange(len(ecg_downsampled)) / new_sfreq
ecg_df = pd.DataFrame({'ECG': ecg_downsampled}, index=time_vector)




# Process the downsampled ECG signal with NeuroKit2
ecg_processed  = nk.ecg_process(ecg_df['ECG'], sampling_rate=new_sfreq)

ecg_processed_df = ecg_processed[0]
ecg_processed_df.columns
ecg_processed_dict = ecg_processed[1]
ecg_processed_dict.keys()

# plot
nk.ecg_plot(ecg_processed_df)

ecg_analyzed = nk.ecg_analyze(ecg_processed, method = "interval-related")


#### 1. HR

hr = ecg_processed_df['ECG_Rate']
# Calculate the average heart rate
average_heart_rate = hr.mean()
average_heart_rate = float(average_heart_rate)
print(f"Average Heart Rate: {average_heart_rate} bpm")


#### 2. HRV

# extract R-peaks
R_peaks = ecg_processed[1]['ECG_R_Peaks']
# get hrv
hrv = nk.hrv(R_peaks)
rmssd = hrv['HRV_RMSSD'].values[0]
rmssd_value = float(rmssd)
print(f"HRV (RMSSD): {rmssd_value}")



#### 3. Amplitude of the R-wave

ecg_events,_,_ = mne.preprocessing.find_ecg_events(ecg_data, ch_name='ECG')

baseline = (-0.25,-0.2)
ecg_epochs = mne.preprocessing.create_ecg_epochs(ecg_data, ch_name='ECG', tmin = -0.25, tmax = 0.55, baseline = baseline)
# Access the epochs data
epoch_data = ecg_epochs.get_data()

# Define the time point within the epoch to extract the R-peak amplitude
# Calculate index for time point 0s, since we're focusing on the peak amplitude in the epoch
r_peak_amplitude_index = int(0.0 * ecg_epochs.info['sfreq'])  # Index for the R-peak position within the epoch

# Extract R-peak amplitudes from all epochs
# Assuming single channel, adjust channel index if multiple channels are used
r_peak_amplitudes = epoch_data[:, 0, r_peak_amplitude_index]

# Calculate the average R-peak amplitude
average_r_peak_amplitude = np.mean(r_peak_amplitudes)

average_r_peak_amplitude_microvolts = average_r_peak_amplitude * 1e6
average_r_peak_amplitude_microvolts = float(average_r_peak_amplitude_microvolts)
print(f'Average R Peak Amplitude: {average_r_peak_amplitude * 1e6:.6f} µV')  # Convert from V to µV

"""
from neurokit:
    # Baseline correction
ecg_baseline_corrected = ecg_downsampled - np.mean(ecg_downsampled)

# Process the baseline-corrected ECG signal
ecg_processed_corrected = nk.ecg_process(ecg_baseline_corrected, sampling_rate=new_sfreq)

# Extract R-wave amplitudes
r_wave_amplitude = ecg_processed_corrected[0]['ECG_Clean'].loc[ecg_processed_corrected[1]['ECG_R_Peaks'] == 1].mean()
print(f'Average Amplitude of R Wave: {r_wave_amplitude:.2f} µV')
"""


#### 4. QT Interval

# Calculate QT interval
# Assuming 'ECG_Q_Peaks' and 'ECG_T_Offsets' are in the same unit (e.g., seconds or samples)
T_offsets = np.array(ecg_processed_dict['ECG_T_Offsets'])
#T_offsets = float(T_offsets)
len(T_offsets)

Q_peaks = np.array(ecg_processed_dict['ECG_Q_Peaks'])
#Q_peaks = float(Q_peaks)
len(Q_peaks)

ecg_processed_dict['QT_Interval'] = T_offsets - Q_peaks
QT_intervals = ecg_processed_dict['QT_Interval']
# Convert the QT intervals from samples to milliseconds
QT_intervals = (QT_intervals / new_sfreq) * 1000
average_QT_interval = np.nanmean(QT_intervals)
print(f"average_QT_interval: {average_QT_interval}")




#### 5. QTc Interval

# The corrected QT interval (QTc) was calculated using the Bazett formula (Bazett, 1997).








