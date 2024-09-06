#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:34:22 2024

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
#from pyprep import PreprocessingPipeline

# matplotlib.use('TkAgg')  # You can try different backends (Qt5Agg, TkAgg, etc.)

# gives you info on whole packages in mne env
# mne.sys_info()

#  Dir where the HEP data is : make it working dir..
hep_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/HEPs"
os.chdir(hep_dir)

# import all HEP objects for patients
eyes_closed_heps = mne.read_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/HEPs/heps_list_V1_eyes_closed-ave.fif')
eyes_open_heps = mne.read_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/HEPs/heps_list_V1_eyes_open-ave.fif')
hct_heps = mne.read_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/HEPs/heps_list_V1_hct-ave.fif')

# for HCs
eyes_closed_heps_hc = mne.read_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HEPs/HC_heps_list_V1_eyes_closed-ave.fif')
eyes_open_heps_hc = mne.read_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HEPs/HC_heps_list_V1_eyes_open-ave.fif')
hct_heps_hc = mne.read_evokeds('/Users/denizyilmaz/Desktop/BrainTrain/Healthy Controls_BHC/BrainTrain_EEG_data_HC/HEPs/HC_heps_list_V1_hct-ave.fif')


# Compute the grand average for patients
grand_average_eyes_closed_hep_evoked = mne.grand_average(eyes_closed_heps)
grand_average_eyes_open_hep_evoked = mne.grand_average(eyes_open_heps)
grand_average_hct_hep_evoked = mne.grand_average(hct_heps)


# Compute the grand average for HCs
grand_average_eyes_closed_hep_evoked_hc = mne.grand_average(eyes_closed_heps_hc)
grand_average_eyes_open_hep_evoked_hc = mne.grand_average(eyes_open_heps_hc)
grand_average_hct_hep_evoked_hc = mne.grand_average(hct_heps_hc)

# Organize the evoked objects into a dictionary
evokeds_sz_hc = {
    'Eyes-closed SZ': grand_average_eyes_closed_hep_evoked,
    'HCT SZ': grand_average_hct_hep_evoked,
    'Eyes-closed HC': grand_average_eyes_closed_hep_evoked_hc,
    'HCT HC': grand_average_hct_hep_evoked_hc,
}

# Define All conditions
evokeds_sz_hc_all = {
    'Eyes-open SZ': grand_average_eyes_open_hep_evoked,
    'Eyes-closed SZ': grand_average_eyes_closed_hep_evoked,
    'HCT SZ': grand_average_hct_hep_evoked,
    
    'Eyes-open HC': grand_average_eyes_open_hep_evoked_hc,
    'Eyes-closed HC': grand_average_eyes_closed_hep_evoked_hc,
    'HCT HC': grand_average_hct_hep_evoked_hc    
}

# Define Color Shades
colors = {
    'Eyes-open SZ': '#9ACD32',    # Light Green
    'Eyes-closed SZ': '#4caf50',  # Medium Green
    'HCT SZ': '#004d00',          # Very Dark Green

    'Eyes-open HC': '#FFD700',    # Gold (light orange)
    'Eyes-closed HC': '#FF8C00',  # DarkOrange (medium orange)
    'HCT HC': '#FF4500'           # OrangeRed (dark orange)
}

# vertical lines to highlight window
vlines = [0, 0.45, 0.5]

### Plot F4 using the specified colors
f4_comparison_plot = mne.viz.plot_compare_evokeds(
    evokeds_sz_hc_all, 
    picks='F4', 
    colors=colors,  # Apply the custom colors
    vlines=vlines   
)
# save
f4_comparison_plot[0].savefig('/Users/denizyilmaz/Desktop/BrainTrain/Results/f4_comparison_plot_all_tasks.jpg', format='jpg')

### Plot F8 using the specified colors
f8_comparison_plot = mne.viz.plot_compare_evokeds(
    evokeds_sz_hc_all, 
    picks='F8', 
    colors=colors,  # Apply the custom colors
    vlines=vlines   
)
# save
f8_comparison_plot[0].savefig('/Users/denizyilmaz/Desktop/BrainTrain/Results/f8_comparison_plot_all_tasks.jpg', format='jpg')

### Plot Fp2 using the specified colors
fp2_comparison_plot = mne.viz.plot_compare_evokeds(
    evokeds_sz_hc_all, 
    picks='Fp2', 
    colors=colors,  # Apply the custom colors
    vlines=vlines   
)
# save
fp2_comparison_plot[0].savefig('/Users/denizyilmaz/Desktop/BrainTrain/Results/fp2_comparison_plot_all_tasks.jpg', format='jpg')




## Or using defaults...

f4_comparison_plot = mne.viz.plot_compare_evokeds(evokeds_sz_hc, picks='F4', vlines = vlines)
f4_comparison_plot[0].savefig('/Users/denizyilmaz/Desktop/BrainTrain/Results/f4_comparison_plot.jpg', format='jpg')
f8_comparison_plot = mne.viz.plot_compare_evokeds(evokeds_sz_hc, picks='F8', vlines = vlines)
f8_comparison_plot[0].savefig('/Users/denizyilmaz/Desktop/BrainTrain/Results/f8_comparison_plot.jpg', format='jpg')
fp2_comparison_plot = mne.viz.plot_compare_evokeds(evokeds_sz_hc, picks='Fp2', vlines = vlines)
fp2_comparison_plot[0].savefig('/Users/denizyilmaz/Desktop/BrainTrain/Results/fp2_comparison_plot.jpg', format='jpg')


frontal_central_regions= ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8','Fz', 'FC1', 'FC2', 'FC5', 'FC6', 'C3', 'C4', 'Cz']
all_frontal_comparison_plot = mne.viz.plot_compare_evokeds(evokeds_sz_hc, picks=frontal_central_regions, vlines = vlines, combine = 'mean') # combine default is GFP
all_frontal_comparison_plot[0].savefig('/Users/denizyilmaz/Desktop/BrainTrain/Results/all_frontal_comparison_plot.jpg', format='jpg')


three_picks = ['Fp2','F4', 'F8']
three_chans_comparison_plot = mne.viz.plot_compare_evokeds(evokeds_sz_hc, picks = three_picks, vlines = vlines, combine = 'mean')
three_chans_comparison_plot[0].savefig('/Users/denizyilmaz/Desktop/BrainTrain/Results/three_chans_comparison_plot.jpg', format='jpg')



# Plot comparison for the 'F4' electrode
# Organize the evoked objects into a dictionary
evokeds = {
    'Eyes-closed': grand_average_eyes_closed_hep_evoked,
    'HCT': grand_average_hct_hep_evoked
}
# vertical lines to highlight window
vlines = [0, 0.45, 0.5]
#plot
mne.viz.plot_compare_evokeds(evokeds, picks='F4', vlines = vlines)
mne.viz.plot_compare_evokeds(evokeds, picks='F8', vlines = vlines)
mne.viz.plot_compare_evokeds(evokeds, picks='FP2', vlines = vlines)
mne.viz.plot_compare_evokeds(evokeds, picks='FP2', vlines = vlines)




# import hep_output csv 
hep_outputs_file_path = '/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/HEPs/hep_outputs.csv'
hep_outputs = pd.read_csv(hep_outputs_file_path)

# only need v1 and eyes-closed for now 
v1_eyes_closed_hep_data = hep_outputs[(hep_outputs['task'] == 'eyes-closed') & (hep_outputs['session'] == 'V1')]



# import the EEG spreadsheet
beh_outputs_file_path = '/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_beh_data/EEG Behavioral Data & Preprocessing Log & Data Checklist - BT_behavioral_data.csv'
beh_outputs = pd.read_csv(beh_outputs_file_path)

# Change the colnames to the 1st row names, and delete the useless current colnames
new_column_names = beh_outputs.iloc[0]
beh_outputs.columns = new_column_names
beh_outputs = beh_outputs[1:]
beh_outputs.reset_index(drop=True, inplace=True)

# now only need session v1
v1_beh_outputs = beh_outputs[beh_outputs['Session'] == 'V1']

# make subject_id colname compatible
v1_beh_outputs.rename(columns={'BT_ID': 'subject_id'}, inplace=True)

# make subject_id s compatible, right now hep is BTSCZ001 but beh is BT001
v1_beh_outputs['subject_id'] = v1_beh_outputs['subject_id'].str.replace(r'^(BT)', r'\1SCZ', regex=True)


# Merge BEH and HEP outputs
hep_beh_data = pd.merge(v1_beh_outputs, v1_eyes_closed_hep_data, on='subject_id', how='inner')

#### correlate stuff

# get rid of the stupid comma
hep_beh_data['CDS'] = [x.split(',')[0] if isinstance(x, str) else x for x in hep_beh_data['CDS']]

# make qustionnaire score numeric
hep_beh_data['CDS'] = pd.to_numeric(hep_beh_data['CDS'])

# corr
cds_f4_hep_corr = hep_beh_data['CDS'].corr(hep_beh_data['F4_mean_amplitude'])
# or
hep_beh_data[['CDS', 'F4_mean_amplitude']].corr()

### Plot
# need to either apply log_transformaiton to hep_amplitude or rescale or zoom in bcs its scale is soooo tiny...

# Extract relevant data (assuming both columns are numeric and contain no NaN values)
x = hep_beh_data['CDS']
y = hep_beh_data['F4_mean_amplitude']

# Create the scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(x, y, color='blue', alpha=0.7, label='Data points')

# Add a line of identity (y=x)
identity_line = [min(min(x), min(y)), max(max(x), max(y))]
plt.plot(identity_line, identity_line, color='red', linestyle='--', label='Line of identity (y=x)')

# Set plot labels and title
plt.xlabel('CDS')
plt.ylabel('F4_mean_amplitude')
plt.title('Scatter Plot of CDS vs. F4_mean_amplitude with Line of Identity')
plt.legend()
plt.grid(True)
plt.show()




# %% 2. Comparison of Conditions : Visualization.....  These may rather belong to script 4_pilot_HEP_stats  !!!


# import HEP files?
prep_ica_file_name_v1 = 'BTSCZ011_V1_eyes-closed_prep_ICA.fif'

# construct the full file path
file_path_v1 = os.path.join(prep_ica_dir, prep_ica_file_name_v1)
#file_path_v3 = os.path.join(prep_ica_dir, prep_ica_file_name_v3)


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


# %% CLUSTER PERMUTATION TEST TO CORRECT FOR MULTIPLE COMPARISONS

import numpy as np
from scipy.stats import ttest_ind
from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt

# Simulate EEG-like data
np.random.seed(42)

n_channels = 64  # Number of EEG channels
n_samples = 100  # Number of time points
n_subjects_per_group = 30

# Simulate data for two groups
group1_data = np.random.randn(n_subjects_per_group, n_channels, n_samples)
group2_data = np.random.randn(n_subjects_per_group, n_channels, n_samples)

# Introduce a significant effect in a cluster of electrodes (e.g., channels 30-40)
group2_data[:, 30:40, 40:60] += 1.5  # Add a signal to group 2

# Perform t-test on each channel and time point
t_values, p_values = ttest_ind(group1_data, group2_data, axis=0)

# Set adjacency matrix (assuming 8-neighbor connectivity in a 2D grid)
adjacency_matrix = np.zeros((n_channels, n_channels))
for i in range(n_channels):
    for j in range(i + 1, n_channels):
        if abs(i - j) == 1 or abs(i - j) == 8:  # 8-neighbor connectivity
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1

# Cluster-based permutation test
cluster_stats = permutation_cluster_test(
    [group1_data, group2_data],
    n_permutations=1000,
    tail=0,
    adjacency=adjacency_matrix,
    threshold=None,
    n_jobs=1,
    out_type='mask'
)

T_obs, clusters, cluster_p_values, H0 = cluster_stats

# Plot the results
plt.figure(figsize=(12, 6))
for i, c in enumerate(clusters):
    c = c[0]  # Extract the boolean mask
    if cluster_p_values[i] < 0.05:  # Check if the cluster is significant
        plt.subplot(2, 1, 1)
        plt.title('Cluster-based Permutation Test')
        plt.imshow(T_obs * c, aspect='auto', origin='lower',
                   extent=[0, n_samples, 0, n_channels],
                   cmap='RdBu_r')
        plt.colorbar(label='T-value')
        plt.xlabel('Time points')
        plt.ylabel('Channels')

plt.tight_layout()
plt.show()
