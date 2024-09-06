#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:05:41 2024

@author: denizyilmaz

Change within loop to add epoched stuff instead of raw!!!
"""




# %%  0. Import Packages 

import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
from mne_icalabel import label_components
from mne.viz import plot_ica_sources
from autoreject import AutoReject # for rejecting bad channels
from autoreject import get_rejection_threshold  
from collections import Counter
from pyprep.find_noisy_channels import NoisyChannels
#from pyprep import PreprocessingPipeline


import matplotlib
# matplotlib.use('TkAgg')  # You can try different backends (Qt5Agg, TkAgg, etc.)


# gives you info on whole packages in mne env
# mne.sys_info()

#  dir where dta preprocessed until ıca is stored
prep_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_until_ICA"
os.chdir(prep_dir)

# # name of file
# file_name = 'BTSCZ022_V1_eyes-open_prep_until_ICA.fif'
# # construct the full file path
# file_path = os.path.join(prep_dir, file_name)
# # Load the FIF file
# prep_data = mne.io.read_raw_fif(file_path, preload=True)

# %%  1. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

# Define the column names for the DF
column_names = ['subject_id', 'session', 'task', 'total_explained_var_ratio',
                'epoched_ICA_labels', 'epoched_ICA_percentage', 
                'epoched_ICA_heart_found', 'epoched_ICA_blink_found', 'epoched_ICA_muscle_found', 
                'num_components_total', 'num_components_excluded', 'rejection_criteria', 'ica_filter'
                ]

# Initialize an empty DF with columns
ica_epoched_outputs = pd.DataFrame(columns=column_names)


# %% Perform ICA on RAW data

### Find Participant numbers

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


for participant_no in participant_numbers:
    for session in sessions:
        for task in tasks: 
                        
            # HERE raise an error for non existing files!! ON CHATGPT !!
            
            # Format the filename string
            filename = f"BTSCZ{participant_no}_{session}_{task}_prep_until_ICA.fif"
            
            try:
                # Load the raw data
                prep_data = mne.io.read_raw(filename, preload=True)
            
                # Proceed with further processing here
                print(f"Successfully loaded: {filename}")
                
            except FileNotFoundError:
                print(f"File not found: {filename}. Skipping...")
                continue
            
            
            prep_data_ica_filtered = prep_data.copy().filter(l_freq=1.0, h_freq=100)  # you can try and see lower lowpass (e.g. 0.1, 0.3,..)for heart artifacts but other components may get worse
            
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
            
            # decide on component number for ICA
            good_channels = mne.pick_types(prep_data_ica_filtered.info, meg=False, eeg=True, exclude='bads')
            best_n_components = len(good_channels) - 1
            
            # Do ICA on the epochs we prepared
            ica = ICA(
                n_components= best_n_components, 
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
                print(f"Fraction of {channel_type} variance explained by all components: " f"{ratio}")
            
            
            # plot ICs on the original data: From here use the original prep data!
            # prep_data.load_data()
            # ica.plot_sources(prep_data, show_scrollbars=False)
            # ica.plot_components()
            
            
            # Automatically label components using the 'iclabel' method
            ic_labels = label_components(prep_data, ica, method='iclabel')
            component_labels = ic_labels['labels']
            predicted_probabilities = ic_labels['y_pred_proba']
            
            # Print the results
            print("Predicted Probabilities:", ic_labels['y_pred_proba'])
            print("Component Labels:", ic_labels['labels'])
            
            # Check whether heart component was found 
            nr_heart_components = 0
            nr_blink_components = 0
            nr_muscle_components = 0
            
            for label in component_labels:
                if label == 'heart beat':
                    nr_heart_components = nr_heart_components + 1
                elif label == 'eye blink':
                    nr_blink_components = nr_blink_components + 1
                elif label == 'muscle artifact':
                    nr_muscle_components = nr_muscle_components + 1
            
            
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
            
            # # compare ica cleaned and before
            # prep_data.plot()
            # prep_ica_data.plot()
            
            
            ### Prepare the CSVto Save 
            
            # participant id should be BTSCZ...
            participant_id = f"BTSCZ{participant_no}"
            
            # Create a dictionary representing the new row
            new_row = pd.Series({'subject_id': participant_id, 
                                 'session': session, 
                                 'task': task,
                                 'total_explained_var_ratio': explained_var_ratio['eeg'],
                                 'epoched_ICA_labels': component_labels,
                                 'epoched_ICA_percentage': predicted_probabilities, 
                                 'epoched_ICA_heart_found': nr_heart_components,
                                 'epoched_ICA_blink_found': nr_blink_components,
                                 'epoched_ICA_muscle_found': nr_muscle_components,
                                 'num_components_total': best_n_components,
                                 'num_components_excluded': len(exclude_index),
                                 'rejection_criteria': 'iclabel =! brain or other (with predicted_probabilities > .50)',
                                 'ica_filter': 'l_freq=1.0, h_freq=100'
                                 })
           
            # convert row to df
            new_row =  new_row.to_frame().T

            # add to existing df the current data outputs
            ica_epoched_outputs = pd.concat([ica_epoched_outputs, new_row], ignore_index=True)

            # Print the DataFrame
            print(ica_epoched_outputs)
                        
            ### Save the Data 
            # IF u need to save data again uncomment below!
            
            ica_file_name = filename.replace('_until_', '_epoched_')
            file_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_ICA_applied_on_1sec_epochs', ica_file_name)
            prep_ica_data.save(file_path, overwrite=True)
            
            
            ### CSV : can be in or out of the loop
            
            # Construct the csv path
            csv_filename = 'ica_output_epoched_preprocessed.csv'
            csv_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_ICA_applied_on_1sec_epochs',csv_filename)
            ica_epoched_outputs.to_csv(csv_path, mode='w', sep=',', index=False)


