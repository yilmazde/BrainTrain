#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:22:45 2024

@author: denizyilmaz
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
                'cont_ICA_labels', 'cont_ICA_percentage', 
                'cont_ICA_heart_found', 'cont_ICA_blink_found', 'cont_ICA_muscle_found', 
                'bads_ecg_indices', 'bads_ecg_scores_corr', 'n_bads_ecg', 
                'num_components_total', 'num_components_excluded', 'rejection_criteria'
                ]

# Initialize an empty DF with columns
ica_raw_outputs = pd.DataFrame(columns=column_names)


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



            # For ICA to perform better we need filter 1, THEN copy weights back to filter .3
            raw_ica_filtered = prep_data.copy().filter(l_freq=1.0, h_freq=100)  # you can try and see lower lowpass (e.g. 0.1, 0.3,..)for heart artifacts but other components may get worse
            
            """
            Richard: 
                you can simply stick with the default values here – we automatically account for rank deficiency, 
                which seems to work in almost all cases. So: don’t set n_components, don’t set n_pca_components, 
                and you should be good in 99% of the cases. 
                After fitting, check ica.n_components_ (mind the trailing underscore) to find out how many components were kept.
                
                # decide on component number for ICA
                good_channels = mne.pick_types(raw_ica_filtered.info, meg=False, eeg=True, exclude='bads')
                best_n_components = len(good_channels) - 1 # -1 for the acerage rereferencing beforehand
                print("Components after having accounted for rank deficiency: ", best_n_components)
                
                # now reset bads because theyve been interpolated anyway
                raw_ica_filtered.info["bads"] = []
                prep_data.info["bads"] = []
                print("Number of bad channels for raw_ica_filtered: ", raw_ica_filtered.info["bads"])
                print("Number of bad channels for prep_data: ", prep_data.info["bads"])
            """
            
            
            # Set up and fit the ICA
            ica = ICA(             #  n_components=best_n_components, AUTOMATICALLY DONE reduce the dim (by 1 for average reference and 1 for each interpolated channel) for optimal ICA performance
                max_iter="auto", 
                method="infomax", 
                random_state=97,
                fit_params=dict(extended=True)
                ) # n_components should be fit to the # interpolated channels ... ICLabel requires extended infomax!
            ica.fit(raw_ica_filtered)
            ica
            
            # Print explained var for ICA
            explained_var_ratio = ica.get_explained_variance_ratio(raw_ica_filtered)
            for channel_type, ratio in explained_var_ratio.items():
                print(f"Fraction of {channel_type} variance explained by all components: " f"{ratio}")
            
            # # Plot ICs: From here use the original prep data!
            # prep_data.load_data()
            # ica.plot_sources(prep_data, show_scrollbars=False) # you can call the original unfiltered raw object
            # ica.plot_components(inst = prep_data)
            # ica.plot_overlay(prep_data)
            # ica.plot_properties(prep_data, picks=[4])  # visualize a randomly selected component
            
            # Automatically label components using the 'iclabel' method
            ic_labels = label_components(prep_data, ica, method='iclabel')
            component_labels = ic_labels['labels']
            predicted_probabilities = ic_labels['y_pred_proba']
                                
            # Print the results
            print("Predicted Probabilities:", ic_labels['y_pred_proba'])
            print("Component Labels:", ic_labels['labels'])
            # Maybe: Create a dictionary mapping component labels to their probabilities
            
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
            
            
            # Extract non-brain labels' index to exclude them from original data
            # only those labels where algorithm assigns above chance probability to the label, as per Berkan's suggestion
            labels = ic_labels["labels"]
            exclude_index = [
                index for index, label in enumerate(labels) if label not in ["brain", "other"] and predicted_probabilities[index] > 0.50
            ]
            
            # # ADD: CORRELATION WITH ECG signal!!  find which ICs match the ECG pattern and exclude those too
            ecg_indices, ecg_scores = ica.find_bads_ecg(prep_data, method="correlation", threshold="auto")
            n_bads_ecg = len(ecg_indices) # number of ecg related ICs found by find_bads_ecg
            exclude_index.extend(ecg_indices)

            # Assign those bads ICs to ica.exclude
            ica.exclude = exclude_index
            print(f"Excluding these ICA components: {exclude_index}")
            
            
            # Exclude the bad Components: Reconstruct the original data without noise components
            # ica.apply() changes the Raw object in-place, so let's make a copy first:
            prep_ica_data = prep_data.copy()
            ica.apply(prep_ica_data, exclude=exclude_index) #  no need:  n_pca_components=best_n_components
            
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
                                 'cont_ICA_labels': component_labels,
                                 'cont_ICA_percentage': predicted_probabilities, 
                                 'cont_ICA_heart_found': nr_heart_components,
                                 'cont_ICA_blink_found': nr_blink_components,
                                 'cont_ICA_muscle_found': nr_muscle_components,
                                 'bads_ecg_indices': ecg_indices, 
                                 'bads_ecg_scores_corr': ecg_scores,
                                 'n_bads_ecg': n_bads_ecg,
                                 'num_components_total': ica.n_components_, # best_n_components,
                                 'num_components_excluded': len(exclude_index),
                                 'rejection_criteria': 'iclabel =! brain or other (with predicted_probabilities > .50)'
                                 })
           
            # convert row to df
            new_row =  new_row.to_frame().T

            # add to existing df the current data outputs
            ica_raw_outputs = pd.concat([ica_raw_outputs, new_row], ignore_index=True)

            # Print the DataFrame
            print(ica_raw_outputs)
            
            
            # SAVE the Preprocessed Data & Prep Ouputs in a CSV
            
            ### Save the Data 
            
            # IF u need to save data again uncomment below!
            
            # Save the data, yay!
            ica_file_name = filename.replace('_until_', '_')
            file_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_ICA_applied_on_raw', ica_file_name)
            prep_ica_data.save(file_path, overwrite=True)
            
            
            ### CSV : can be in or out of the loop
            
            # Construct the csv path
            csv_filename = 'ica_output_with_find-bads-ecg_raw_preprocessed.csv'
            csv_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/Preprocessed_ICA_applied_on_raw',csv_filename)
            ica_raw_outputs.to_csv(csv_path, mode='w', sep=',', index=False)

            
            # ARCHIVE
            # # Open a CSV file in write mode ADD as one col amount of heart components found!
            # with open('predicted_probabilities.csv', 'w', newline='') as csvfile:
            #     # Define CSV writer
            #     csvwriter = csv.writer(csvfile)
                
            #     # Write the header row
            #     csvwriter.writerow(['subject_no', 'labels', 'probabilities'])
                
            #     # Write data rows
            #     for subject_no, label, probability in zip(subject_numbers, component_labels, predicted_probabilities):
            #         csvwriter.writerow([subject_no, label, probability])
            
            # print("CSV file created successfully.")

