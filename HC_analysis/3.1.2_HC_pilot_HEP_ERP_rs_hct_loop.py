#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:22:56 2024

@author: denizyilmaz
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
import datetime
#from pyprep import PreprocessingPipeline

# matplotlib.use('TkAgg')  # You can try different backends (Qt5Agg, TkAgg, etc.)

# gives you info on whole packages in mne env
# mne.sys_info()

#  Dir where data preprocessed and ICA cleaned is stored
prep_ica_dir = '/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HC_Preprocessed_ICA_applied_on_raw'
os.chdir(prep_ica_dir)

# %%  1. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

# Define the column names for the DF
column_names = ['subject_id', 'task', 'event_type',  'event_times',
                'good_epoch_count','percentage_dropped_epochs', 'epoch_time_window', 'baseline_correction', 
                'hep_time_window',  'channels', 'hep_max_amplitudes', 'hep_max_latencies', 'hep_min_amplitudes', 'hep_min_latencies',
                'hep_mean_amplitudes', 'hep_amplitudes_sd', 'Fp2_mean_amplitude', 'F4_mean_amplitude', 'F8_mean_amplitude',
                'Fp2_max_amplitude', 'F4_max_amplitude', 'F8_max_amplitude', 'Fp2_min_amplitude', 'F8_min_amplitude', 
                'Fp2_max_latency', 'F4_max_latency', 'F8_max_latency', 'Fp2_min_latency', 'F4_min_latency', 'F8_min_latency',
                'mean_HEP_accross_channels',
                'start_time_of_analysis', 'analysis_duration'
                ]


# Initialize an empty DF with columns
hep_outputs = pd.DataFrame(columns=column_names)

# Lists to store evoked objects
hep_list_eyes_closed = []
hep_list_eyes_open = []
hep_list_hct = []

# %% 2. loopidiebooo

# Get a list of filenames from the current directory
all_files = os.listdir()

# Filter filenames that end with ".eeg"
eeg_file_names = [filename for filename in all_files if filename.endswith('.fif')]

# Initialize Participant, Task
participant_numbers = []
for file in eeg_file_names:
    num = file[3:6]
    print(num)
    participant_numbers.append(num)
    
participant_numbers =  list(set(participant_numbers))

# turn into an array
participant_numbers = np.array(sorted(participant_numbers))

### Define all file relevant variables


tasks = ['eyes-closed', 'eyes-open', 'hct']
# tasks = ['eyes-closed', 'eyes-open', 'hct']  # maybe first try without hct, corr w questionnaires then move forward with hct.....

for participant_no in participant_numbers:
    for task in tasks: 
                    
        # HERE raise an error for non existing files!! ON CHATGPT !!
        
        # Format the filename string
        filename = f"BHC{participant_no}_{task}_prep_ICA.fif"
        
        try:
            # Load the raw data
            prep_ica_data = mne.io.read_raw(filename, preload=True)
        
            # Proceed with further processing here
            print(f"Successfully loaded: {filename}")
            
        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping...")
            continue
                    
        # track time
        start_time = datetime.datetime.now()
        
        
        # Filter out irrelevant annotations 
        irrelevant_annotations = ['Comment/actiCAP Data On', 'New Segment/', 'Comment/actiCAP USB Power On', 'ControlBox is not connected via USB', 'actiCAP USB Power On', 'Comment/ControlBox is not connected via USB']  # Adjust this list as needed
        
        # find indices of irrelevant annots
        irrelevant_indices = np.where(np.isin(prep_ica_data.annotations.description, irrelevant_annotations))[0]
        
        # delete irrelevant ones
        prep_ica_data.annotations.delete(irrelevant_indices)
        
        # check whether it worked
        # prep_ica_data.annotations.description
        print("Remaining annotations:", np.unique(prep_ica_data.annotations.description, return_counts=True))
        
        ### Continue with epoching & creating evoked
        
        if task == 'hct':

            # Get event ids of heartbeat_events
            events, event_id = mne.events_from_annotations(prep_ica_data)
            R_peak_id = event_id['R-peak']
            heartbeat_events = events[events[:, 2] == R_peak_id]
            
        else: 
            
            # This below works ONLY for non-HCT tasks (RESTING-STATE) because in HCT we also have the other annots, which wont be dropped, just by dropping the first 2 irrelevants!
            heartbeat_events, R_peak_id = mne.events_from_annotations(prep_ica_data)
        
        # define epoch parameters    
        heart_epoch_tmin = -0.25
        heart_epoch_tmax = 0.55
        epoch_time_window = (heart_epoch_tmin, heart_epoch_tmax)
        baseline = (-0.25, -0.20) # same as: KOREKI: -250 - -200
        
        # create epochs
        heartbeat_epochs = mne.Epochs(
            prep_ica_data, heartbeat_events,
            event_id=R_peak_id, tmin=heart_epoch_tmin, tmax=heart_epoch_tmax,  # here I select the timing same as Koreki et al., 23
            baseline=baseline, preload=True, event_repeated='drop'
        )
        
        # do I need to apply_baseline again ?!
        
        # drop bad epochs
        heartbeat_epochs.drop_bad(reject=dict(eeg=150e-6))
        percentage_dropped_epochs = heartbeat_epochs.drop_log_stats()
        
        # plot epochs
        # heartbeat_epochs.plot(events = heartbeat_events_v1, event_id=event_id_v1)
        
        # create evoked
        heartbeat_evoked = heartbeat_epochs.average()
        
        # Append to list
        if task == 'eyes-closed':
            hep_list_eyes_closed.append(heartbeat_evoked)
        elif task == 'eyes-open':
            hep_list_eyes_open.append(heartbeat_evoked)
        elif task == 'hct':
            hep_list_hct.append(heartbeat_evoked)
        
        ##############     Plot evoked ###############
        
        # directory for plots
        plot_dir = '/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HEPs/plots/'
        
        # time window to highlight on plot
        highlight = (0.45, 0.50)
        
        hep_joint_plot = heartbeat_evoked.plot_joint(show=False)
        hep_joint_plot_path = os.path.join(plot_dir, f'BHC{participant_no}_{task}_joint_plot.jpg' )
        hep_joint_plot.savefig(hep_joint_plot_path, format='jpg')
        
        hep_plot = mne.viz.plot_evoked(heartbeat_evoked, show=False, highlight=highlight)
        hep_plot_path = os.path.join(plot_dir, f'BHC{participant_no}_{task}_hep_plot.jpg' )
        hep_plot.savefig(hep_plot_path, format='jpg')
        
        hep_plot_3chans = mne.viz.plot_evoked(heartbeat_evoked, highlight=highlight, picks = ['F4', 'F8', 'Fp2'], show=False)
        hep_plot_3chans_path = os.path.join(plot_dir, f'BHC{participant_no}_{task}_hep_plot_3chans.jpg' )
        hep_plot_3chans.savefig(hep_plot_3chans_path, format='jpg')
        
        hep_psd_plot = heartbeat_evoked.plot_psd(show=False)
        hep_psd_plot_path = os.path.join(plot_dir, f'BHC{participant_no}_{task}_hep_psd_plot.jpg' )
        hep_psd_plot.savefig(hep_psd_plot_path, format='jpg')
        
        # Compute the average HEP across all fronto-central channels
        frontal_central_regions= ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8','Fz', 'FC1', 'FC2', 'FC5', 'FC6', 'C3', 'C4', 'Cz']
        picks = mne.pick_channels(heartbeat_evoked.ch_names, frontal_central_regions)
        average_frontal_central = heartbeat_evoked.data[picks, :].mean(axis=0)
        # Create an Info object for the averaged data
        info_avg = mne.create_info(['Frontal-Central Average'], heartbeat_evoked.info['sfreq'], ch_types='eeg')
        # Create an Evoked object for the averaged data
        average_frontal_central = mne.EvokedArray(average_frontal_central[np.newaxis, :], info_avg, tmin=heartbeat_evoked.times[0])
        average_frontal_central_plot = mne.viz.plot_evoked(average_frontal_central,  highlight=highlight, show=False)
        average_frontal_central_plot_path = os.path.join(plot_dir, f'BHC{participant_no}_{task}_hep_average_frontal_central_plot.jpg' )
        average_frontal_central_plot.savefig(average_frontal_central_plot_path, format='jpg')

            
        # heartbeat_evoked.plot(gfp=True)
        # heartbeat_evoked_v1.plot_joint().show()
        # heartbeat_evoked_v1.plot()
        # heartbeat_evoked_v1.plot_psd()
        # heartbeat_evoked_v1.plot(picks=['F4'])
        # heartbeat_evoked_v1.plot(picks=['F8'])
        # heartbeat_evoked_v1.plot(picks=['Fp2'])
        
        ########       Extract relevant variables and values.... ##############
        
        
        # we are interested in a time window from 450 to 500 ms
        time_window = (0.45, 0.50)  # in seconds
        
        # Select the time indices for this range
        mask = (heartbeat_evoked.times >= time_window[0]) & (heartbeat_evoked.times <= time_window[1])
        
        # Extract data for all channels within this time window, then average across the time window also get peak values
        hep_mean_amplitudes = heartbeat_evoked.data[:, mask].mean(axis=1)
        hep_amplitudes_sd = heartbeat_evoked.data[:, mask].mean(axis=1)
        
        # Mean HEP accross channels
        mean_HEP_accross_channels = hep_mean_amplitudes.mean()

        
        ## get peak values
        
        evoked_times = heartbeat_evoked.times # define all timepoints of the evoked object
        
        # max
        max_amplitudes = heartbeat_evoked.data[:, mask].max(axis=1)
        max_indices = heartbeat_evoked.data[:, mask].argmax(axis=1)
        max_latencies = evoked_times[mask][max_indices]
        
        # min
        min_amplitudes = heartbeat_evoked.data[:, mask].min(axis=1)
        min_indices = heartbeat_evoked.data[:, mask].argmin(axis=1)
        min_latencies = heartbeat_evoked.times[mask][min_indices]
        
        # You need to add channel names too, to be able to match mean amps to chans...
        channels = heartbeat_evoked.ch_names
        
        # Focus on specific channels, e.g., 'Fz', 'Cz', 'Pz'
        # Find indices of the channels
        channel_indices = [heartbeat_evoked.ch_names.index(ch) for ch in ['Fp2', 'F4', 'F8']]
        selected_mean_amplitude_data = hep_mean_amplitudes[channel_indices]
        selected_max_amplitude_data = max_amplitudes[channel_indices]
        selected_min_amplitude_data = min_amplitudes[channel_indices]
        selected_max_latency_data = max_latencies[channel_indices]
        selected_min_latency_data = min_latencies[channel_indices]


        Fp2_mean_amplitude = selected_mean_amplitude_data[0]
        F4_mean_amplitude = selected_mean_amplitude_data[1]
        F8_mean_amplitude = selected_mean_amplitude_data[2]
        
        Fp2_max_amplitude = selected_max_amplitude_data[0]
        F4_max_amplitude = selected_max_amplitude_data[1]
        F8_max_amplitude = selected_max_amplitude_data[2]
        
        Fp2_min_amplitude = selected_min_amplitude_data[0]
        F4_min_amplitude = selected_min_amplitude_data[1]
        F8_min_amplitude = selected_min_amplitude_data[2]
        
        Fp2_max_latency = selected_max_latency_data[0]
        F4_max_latency = selected_max_latency_data[1]
        F8_max_latency = selected_max_latency_data[2]
        
        Fp2_min_latency = selected_min_latency_data[0]
        F4_min_latency = selected_min_latency_data[1]
        F8_min_latency = selected_min_latency_data[2]
        

        ### ADD!!: Exclude the participant if too many epochs are bad, etc.
         
        ### Prepare the CSVto Save 
        
        # participant id should be BTSCZ...
        participant_id = f"BHC{participant_no}"
        
        # record end_time 
        end_time = datetime.datetime.now()
        
        # Calculate the duration of the analysis
        duration = end_time - start_time
        
        # Create a dictionary representing the new row
        new_row = pd.Series({'subject_id': participant_id, 
                             'task': task,
                             'event_type': 'R-peaks for HEP',
                             'event_times': heartbeat_events[:,0],
                             'good_epoch_count': len(heartbeat_epochs), 
                             'percentage_dropped_epochs': percentage_dropped_epochs, 
                             'epoch_time_window': epoch_time_window,
                             'baseline_correction': baseline,
                             'hep_time_window': time_window,
                             'channels': channels,
                             'hep_max_amplitudes': max_amplitudes,
                             'hep_max_latencies': max_latencies,
                             'hep_min_amplitudes': min_amplitudes,
                             'hep_min_latencies': min_latencies,
                             'hep_mean_amplitudes': hep_mean_amplitudes,
                             'hep_amplitudes_sd': hep_amplitudes_sd,
                             'Fp2_mean_amplitude': Fp2_mean_amplitude, 
                             'F4_mean_amplitude': F4_mean_amplitude, 
                             'F8_mean_amplitude': F8_mean_amplitude,
                             'Fp2_max_amplitude': Fp2_max_amplitude, 
                             'F4_max_amplitude': F4_max_amplitude, 
                             'F8_max_amplitude': F8_max_amplitude, 
                             'Fp2_min_amplitude': Fp2_min_amplitude, 
                             'F8_min_amplitude': F8_min_amplitude, 
                             'Fp2_max_latency': Fp2_max_latency, 
                             'F4_max_latency': F4_max_latency, 
                             'F8_max_latency': F8_max_latency, 
                             'Fp2_min_latency': Fp2_min_latency, 
                             'F4_min_latency': F4_min_latency, 
                             'F8_min_latency': F8_min_latency,
                             'mean_HEP_accross_channels': mean_HEP_accross_channels,
                             'start_time_of_analysis': start_time,
                             'analysis_duration': duration
                             })
        
        # convert row to df
        new_row =  new_row.to_frame().T
        
        # add to existing df the current data outputs
        hep_outputs = pd.concat([hep_outputs, new_row], ignore_index=True)
        
        # Print the DataFrame
        print(hep_outputs)
                    
        ### Save the Data 
        
        #### Save !                        
        hep_file_name = filename.replace('_prep_ICA', '_hep-ave')
        file_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HEPs/', hep_file_name)
        mne.write_evokeds(file_path, heartbeat_evoked, on_mismatch='raise', overwrite=True,)   # evokedEvoked instance, or list of Evoked instance; to load it back: evokeds_list = mne.read_evokeds(evk_file, verbose=False)
        
        # Save all Evokeds
        if task == 'eyes-closed':
            mne.write_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HEPs/HC_heps_list_V1_eyes_closed-ave.fif', hep_list_eyes_closed, overwrite=True)
        elif task == 'eyes-open':
            mne.write_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HEPs/HC_heps_list_V1_eyes_open-ave.fif', hep_list_eyes_open, overwrite=True)
        elif task == 'hct':
            mne.write_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HEPs/HC_heps_list_V1_hct-ave.fif', hep_list_hct, overwrite=True)
   
        
        ### CSV : can be in or out of the loop
        
        # Construct the csv path
        csv_filename = 'HC_hep_outputs_new.csv'
        csv_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HEPs/',csv_filename)
        hep_outputs.to_csv(csv_path, mode='w', sep=',', index=False)
        

        
         
