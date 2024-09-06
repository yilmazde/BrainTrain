
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


#  Dir where data preprocessed and ICA cleaned is stored
prep_ica_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_until_ICA"
os.chdir(prep_ica_dir)

# %%  1. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

# Define the column names for the DF
column_names = ['subject_id', 'session', 'task', 
                                 'heart_rate_bpm',
                                 'hrv_rmssd_ms',
                                 'R_peak_amplitude_mV', 
                                 'QT_interval_ms',
                                 'QTc_interval_ms', 
                                 'baseline'
                ]


# Initialize an empty DF with columns
ecg_outputs = pd.DataFrame(columns=column_names)

# Lists to store evoked objects
ecg_list_eyes_closed = []
ecg_list_eyes_open = []
ecg_list_hct = []

# %% 2. loopidiebooo

# Get a list of filenames from the current directory
all_files = os.listdir()

# Filter filenames that end with ".eeg"
eeg_file_names = [filename for filename in all_files if filename.endswith('.fif')]

# Initialize Participant, Session, Task
participant_numbers = []
for file in eeg_file_names:
    num = file[5:8]
    print(num)
    participant_numbers.append(num)
    
participant_numbers =  list(set(participant_numbers))

# turn into an array
participant_numbers = np.array(sorted(participant_numbers))

### Define all file relevant variables

sessions= ['V1', 'V3']

tasks = ['eyes-closed', 'eyes-open', 'hct']
# tasks = ['eyes-closed', 'eyes-open', 'hct']  # maybe first try without hct, corr w questionnaires then move forward with hct.....

for participant_no in participant_numbers:
    for session in sessions:
        for task in tasks: 
                        
                        
            # HERE raise an error for non existing files!! ON CHATGPT !!
            
            # Format the filename string
            filename = f"BTSCZ{participant_no}_{session}_{task}_prep_until_ICA.fif"
            
            try:
                # Load the raw data
                raw = mne.io.read_raw(filename, preload=True)
            
                # Proceed with further processing here
                print(f"Successfully loaded: {filename}")
                
            except FileNotFoundError:
                #print(f"File not found: {filename}. Skipping...")
                continue
            
            ### OR you can directly downsample EEG then extract ECG
            
            # Resample the EEG data to the new sampling rate
            # raw_resampled = raw.copy()
            # raw_resampled.resample(sfreq=new_sfreq)
            # print(raw_resampled.info["sfreq"])
            
            
            
            
            #%% Extract ecg parameters: HR, HRV (RMSSD),  Amplitude of R wave,  QT interval, QTc interval,
            
            
            # Extract ECG signal (assuming the channel name is 'ECG')
            ecg_data = raw.copy().pick_channels(['ECG']).get_data()[0]
            sfreq = raw.info['sfreq']  # Sampling frequency
            new_sfreq = 250
            print(new_sfreq)
            
            # Convert the ECG signal to a NeuroKit2-compatible format
            ecg_downsampled = nk.signal_resample(ecg_data, sampling_rate=sfreq, desired_sampling_rate=new_sfreq) 
            time_vector = np.arange(len(ecg_downsampled)) / new_sfreq
            ecg_df = pd.DataFrame({'ECG': ecg_downsampled}, index=time_vector)
            ecg_df['ECG'] = ecg_df['ECG'].astype(float)

            
            
            # Process the downsampled ECG signal with NeuroKit2
            ecg_processed  = nk.ecg_process(ecg_df['ECG'], sampling_rate=new_sfreq)
            
            ecg_processed_df = ecg_processed[0]
            #ecg_processed_df.columns
            ecg_processed_dict = ecg_processed[1]
            #ecg_processed_dict.keys()
            
            # plot
            # nk.ecg_plot(ecg_processed_df)
            # ecg_analyzed = nk.ecg_analyze(ecg_processed, method = "interval-related")
            
            
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
            rmssd_value_ms = float(rmssd)
            print(f"HRV (RMSSD): {rmssd_value_ms}")
            
            
            
            #### 3. Amplitude of the R-wave
            
            #ecg_events,_,_ = mne.preprocessing.find_ecg_events(ecg_data, ch_name='ECG')
            ecg_data_e = raw.copy().pick_channels(['ECG'])
            baseline = (-0.25,-0.2)
            ecg_epochs = mne.preprocessing.create_ecg_epochs(ecg_data_e, ch_name='ECG',picks = 'ECG', tmin = -0.25, tmax = 0.55, baseline = baseline)
            # Access the epochs data
            epoch_data = ecg_epochs.get_data()
            
            # Define the time point within the epoch to extract the R-peak amplitude
            # Calculate index for time point 0s, since we're focusing on the peak amplitude in the epoch
            r_peak_amplitude_index = int(0.25 * ecg_epochs.info['sfreq'])  # Index for the R-peak position within the epoch
            
            # Extract R-peak amplitudes from all epochs
            # Assuming single channel, adjust channel index if multiple channels are used
            r_peak_amplitudes = epoch_data[:, 0, r_peak_amplitude_index]
            
            # Calculate the average R-peak amplitude
            average_r_peak_amplitude = np.mean(r_peak_amplitudes)
            
            average_r_peak_amplitude_microvolts = average_r_peak_amplitude * 1e6
            average_r_peak_amplitude_microvolts = float(average_r_peak_amplitude_microvolts)
            average_r_peak_amplitude_mV = average_r_peak_amplitude_microvolts/1000
            print(f'Average R Peak Amplitude in mV: {average_r_peak_amplitude_mV} mV') 
            print(f'Average R Peak Amplitude in µV: {average_r_peak_amplitude_microvolts * 1e6:.6f} µV')  # Convert from V to µV
            
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
            
            ecg_mean = ecg_epochs.average(picks = 'ECG')
            # ecg_mean.plot
            
            # Append to list
            if task == 'eyes-closed':
                ecg_list_eyes_closed.append(ecg_mean)
            elif task == 'eyes-open':
                ecg_list_eyes_open.append(ecg_mean)
            elif task == 'hct':
                ecg_list_hct.append(ecg_mean)
            
            
            
            
            
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
            average_QT_interval_ms = np.nanmean(QT_intervals)
            average_QT_interval_ms = float(average_QT_interval_ms)
            print(f"average_QT_interval: {average_QT_interval_ms}") 
            
            
            #### 5. QTc Interval
            
            # The corrected QT interval (QTc) was calculated using the Bazett formula (Bazett, 1997).
            
            # The corrected QT interval (QTc) was calculated using the Bazett formula (Bazett, 1997).
            
            # Calculate RR intervals
            R_peaks_list = list(R_peaks)
            RRs = []
            for i in range(len(R_peaks_list)-1):
                RRs.append(R_peaks_list[i+1]- R_peaks_list[i])
            
            # Convert RR intervals to numpy array
            RRs = np.array(RRs)
            
            # Calculate average RR interval in samples
            average_RR = np.mean(RRs)
            average_RR = float(average_RR)
            
            # Convert average RR interval to milliseconds
            average_RR_msec = (average_RR / new_sfreq) * 1000
            
            
            qtc_seconds = (average_QT_interval_ms / 1000) / np.sqrt(average_RR_msec / 1000)
            
            qtc_seconds = float(qtc_seconds)
            
            qtc_ms = qtc_seconds*1000

             
            #### 6. Prepare the CSVto Save 
            
            # participant id should be BTSCZ...
            participant_id = f"BTSCZ{participant_no}"
            
            
            # Create a dictionary representing the new row
            new_row = pd.Series({'subject_id': participant_id, 
                                 'session': session, 
                                 'task': task,
                                 'heart_rate_bpm':average_heart_rate,
                                 'hrv_rmssd_ms': rmssd_value_ms, 
                                 'R_peak_amplitude_mV': average_r_peak_amplitude_mV, 
                                 'QT_interval_ms': average_QT_interval_ms, 
                                 'QTc_interval_ms': qtc_ms, 
                                 'baseline': baseline
                                 })
            
            # convert row to df
            new_row =  new_row.to_frame().T
            
            # add to existing df the current data outputs
            ecg_outputs = pd.concat([ecg_outputs, new_row], ignore_index=True)
            
            # Print the DataFrame
            print(ecg_outputs)
                        
            ### Save the Data 
            
            #### Save !                        
            #ecg_file_name = filename.replace('_prep_ICA', '_ecg-ave')
            #file_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/ECG', ecg_file_name)
            #mne.write_evokeds(file_path, ecg_mean, on_mismatch='raise', overwrite=True,)   # evokedEvoked instance, or list of Evoked instance; to load it back: evokeds_list = mne.read_evokeds(evk_file, verbose=False)
            
            # Save all Evokeds
            if task == 'eyes-closed' and session == 'V1':
                mne.write_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/ECG/ecgs_list_V1_eyes_closed-ave.fif', ecg_list_eyes_closed, overwrite=True)
            elif task == 'eyes-open' and session == 'V1':
                mne.write_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/ECG/ecgs_list_V1_eyes_open-ave.fif', ecg_list_eyes_open, overwrite=True)
            elif task == 'hct' and session == 'V1':
                mne.write_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/ECG/ecgs_list_V1_hct-ave.fif', ecg_list_hct, overwrite=True)
            elif task == 'eyes-closed' and session == 'V3':
                mne.write_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/ECG/ecgs_list_V3_eyes_closed-ave.fif', ecg_list_eyes_closed, overwrite=True)
            elif task == 'eyes-open' and session == 'V3':
                mne.write_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/ECG/ecgs_list_V3_eyes_open-ave.fif', ecg_list_eyes_open, overwrite=True)
            elif task == 'hct' and session == 'V3':
                mne.write_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/ECG/ecgs_list_V3_hct-ave.fif', ecg_list_hct, overwrite=True)
           
            
            ### CSV : can be in or out of the loop
            
            # Construct the csv path
            csv_filename = 'ecg_outputs_patients.csv'
            csv_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/ECG',csv_filename)
            ecg_outputs.to_csv(csv_path, mode='w', sep=',', index=False)
            
            
            
            
            
            
            
            
