#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:16:56 2024

@author: denizyilmaz

Note:
    
For rank-deficient data such as EEG data after average reference or interpolation, 
it is recommended to reduce the dimensionality 
(by 1 for average reference and 1 for each interpolated channel) 
for optimal ICA performance

Note to self: Think of number of components that is good for ICA!
You can think of making it 24 because 6 will be max number of channel excluded 
if you exclude subject with 20% bad chans. (31 - 1 for average - 6 for interpolated max.)

RuntimeWarning: 
    
The provided ICA instance was fitted with a 'fastica' algorithm. 
ICLabel was designed with extended infomax ICA decompositions. 
To use the extended infomax algorithm, use the 'mne.preprocessing.ICA' instance with the arguments 
'ICA(method='infomax', fit_params=dict(extended=True))' (scikit-learn) or 
'ICA(method='picard', fit_params=dict(ortho=False, extended=True))' (python-picard).

Note: 
Look into find_bads_ecg() method of ICA....


Tutorial to extract labels and reconstruct raw data:
https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html#sphx-glr-generated-examples-00-iclabel-py


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

# name of file
file_name = 'BTSCZ022_V1_eyes-open_prep_until_ICA.fif'

# construct the full file path
file_path = os.path.join(prep_dir, file_name)

# Load the FIF file
prep_data = mne.io.read_raw_fif(file_path, preload=True)

# %%  1. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

# Define the column names for the DF
column_names = ['subject_id', 'session', 'task', 
                'cont_ICA_labels', 'cont_ICA_percentage']

# Initialize an empty DF with columns
prep_outputs = pd.DataFrame(columns=column_names)

# %% Perform ICA on RAW data


# For ICA to perform better we need filter 1, THEN copy weights back to filter .3
raw_ica_filtered = prep_data.copy().filter(l_freq=1.0, h_freq=100)  # you can try and see lower lowpass (e.g. 0.1, 0.3,..)for heart artifacts but other components may get worse

# Set up and fit the ICA
ica = ICA(
    n_components=27, 
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

# Plot ICs: From here use the original prep data!
prep_data.load_data()
ica.plot_sources(prep_data, show_scrollbars=False) # you can call the original unfiltered raw object
ica.plot_components(inst = prep_data)
ica.plot_overlay(prep_data)
ica.plot_properties(prep_data, picks=[4])  # visualize a randomly selected component

# Automatically label components using the 'iclabel' method
ic_labels = label_components(prep_data, ica, method='iclabel')
component_labels = ic_labels['labels']
predicted_probabilities = ic_labels['y_pred_proba']
                    
# Print the results
print("Predicted Probabilities:", ic_labels['y_pred_proba'])
print("Component Labels:", ic_labels['labels'])
# Maybe: Create a dictionary mapping component labels to their probabilities


# Extract non-brain labels' index to exclude them from original data
# only those labels where algorithm assigns above chance probability to the label, as per Berkan's suggestion
labels = ic_labels["labels"]
exclude_index = [
    index for index, label in enumerate(labels) if label not in ["brain", "other"] and predicted_probabilities[index] > 0.50
]
print(f"Excluding these ICA components: {exclude_index}")


# Exclude the bad Components: Reconstruct the original data without noise components
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



# Open a CSV file in write mode
with open('predicted_probabilities.csv', 'w', newline='') as csvfile:
    # Define CSV writer
    csvwriter = csv.writer(csvfile)
    
    # Write the header row
    csvwriter.writerow(['subject_no', 'labels', 'probabilities'])
    
    # Write data rows
    for subject_no, label, probability in zip(subject_numbers, component_labels, predicted_probabilities):
        csvwriter.writerow([subject_no, label, probability])

print("CSV file created successfully.")



# # ADD: CORRELATION WITH ECG signal!! 
# # This is from the mne tutorial : https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#sphx-glr-auto-tutorials-preprocessing-40-artifact-correction-ica-py

# ica.exclude = []
# # find which ICs match the ECG pattern
# ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method="correlation", threshold="auto")
# ica.exclude = ecg_indices

# # barplot of ICA component "ECG match" scores
# ica.plot_scores(ecg_scores)

# # plot diagnostics
# ica.plot_properties(raw, picks=ecg_indices)

# # plot ICs applied to raw data, with ECG matches highlighted
# ica.plot_sources(raw, show_scrollbars=False)

# # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
# ica.plot_sources(ecg_evoked)






# #### Get mixing matrix 
# ###The mixing_matrix obtained here represents the weights applied to each sensor channel to obtain the independent components. 
# ###Each row of the mixing matrix corresponds to one independent component, and each column corresponds to one sensor channel.

# # Get the mixing matrix (also known as the unmixing matrix)
# mixing_matrix = ica.get_components()

# # The mixing matrix will have the shape (n_components, n_channels), where:
# # - n_components is the number of independent components
# # - n_channels is the number of sensor channels

# # You can inspect or use the mixing matrix as needed for further analysis or visualization


