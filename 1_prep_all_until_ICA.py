#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:35:17 2024

@author: denizyilmaz

Spyder Editor

This is a temporary script file.

useful link: https://cbrnr.github.io/blog/importing-eeg-data/

"""

# %%  00. Import Packages & Define WD

import mne
import os
import numpy as np
import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
from mne_icalabel import label_components
from autoreject import AutoReject # for rejecting bad channels
from autoreject import get_rejection_threshold  
from collections import Counter
from pyprep.find_noisy_channels import NoisyChannels
#from pyprep import PreprocessingPipeline
import datetime



import matplotlib
#matplotlib.use('TkAgg')  # You can try different backends (Qt5Agg, TkAgg, etc.)


# gives you info on whole packages in mne env
# mne.sys_info()

# pilot analysis dir
pilot_dir = "/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data"

#pilot_dir = "/Users/denizyilmaz/Desktop/BrainTrain/pilot_analysis/BrainTrain_pilot_data_Michelle"
os.chdir(pilot_dir)

# %%  0. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

# Define the column names for the DF
column_names = ['subject_id', 'session', 'task', 
                'montage', 'new_sampling_rate', 
                'removed_line_noise_freq', 'rereferencing', 
                'total_electrode_nr', 'interpolated_electrode_nr', 'percent_interpolated_electrode', 
                'interpolated_chans',   'bad_chan_detect_method',
                'bandpass_filtering', 'bandpass_filter_method',
                'start_time_of_analysis', 'analysis_duration'
                ]


# Initialize an empty DF with columns
prep_outputs = pd.DataFrame(columns=column_names)


# %% 1. Import data
"""
.eeg: This file contains the actual EEG data. It's the raw EEG signal data you want to analyze.

.vhdr: This is the header file, and it contains metadata and information about the EEG recording, such as channel names, sampling rate, and electrode locations. This file is essential for interpreting the EEG data correctly.

.vmrk: This file contains event markers or annotations that correspond to events in the EEG data. Event markers can be used to mark the timing of specific events or stimuli during the EEG recording. This file is useful for event-related analyses.
"""
### Find Participant numbers

# Get a list of filenames from the current directory
all_files = os.listdir()

# Filter filenames that end with ".eeg"
eeg_file_names = [filename for filename in all_files if filename.endswith('.eeg')]

# Initialize Participant, Session, Task
participant_numbers = []
for file in eeg_file_names:
    num = file[5:8]
    print(num)
    participant_numbers.append(num)
    
participant_numbers =  list(set(participant_numbers))

# turn into an array
participant_numbers = np.array(sorted(participant_numbers))

# define excluded participants, append if necessary
excluded_participants = ['025']

### Define all file relevant variables

sessions= ['V1', 'V3']

tasks = ['eyes-closed', 'eyes-open', 'hct']


for participant_index, participant_no in enumerate(participant_numbers):
    
    if participant_no in excluded_participants:
        print(f"Skipping participant {participant_no} because they are excluded.")
        continue  # skip this iteration if participant is excluded
    
    for session in sessions:
        for task in tasks: 
            
            
            # HERE raise an error for non existing files!! ON CHATGPT !!
            
            # Format the filename string
            filename = f"BTSCZ{participant_no}_{session}_{task}.vhdr"
            
            try:
                # Load the raw data
                raw = mne.io.read_raw(filename, preload=True)
            
                # Proceed with further processing here
                print(f"Successfully loaded: {filename}")
                
            except FileNotFoundError:
                print(f"File not found: {filename}. Skipping...")
                continue
            
                                    
            # track time
            start_time = datetime.datetime.now()
            

            # Load the raw data using the formatted filename
            #raw = mne.io.read_raw(filename, preload=True)
            
            # read data that you already put in your wd
            #raw = mne.io.read_raw("BTSCZ008_V1_eyes-open.vhdr", preload=True)
            #raw = mne.io.read_raw("BTSCZ011_V1_eyes-open.vhdr", preload=True)
            
            
            # alternative way of reading data
            # raw = mne.io.read_raw_brainvision("BTSCZ009_V1_eyes-open.vhdr", preload=True, scale=1)
            # read_raw_brainvision(scale=1e6)
            
            # inspect data visually
            # raw.plot()
            
            """
            You can use raw.apply_function() for this purpose. 
            MNE expects the unit of EEG signals to be V, so because your data is presumably in µV you could convert it like this: 
            raw.apply_function(lambda x: x * 1e-6)
            https://mne.discourse.group/t/rescale-data-import-from-fieldtrip/3402
            """
            
            # %%  2. Get the relevant info of data, if needed uncomment this section
            
            # # you can get specific info by calling an attribute like a dict
        
            # print(raw.info["sfreq"])  # u get sampling frequency
            # print(raw.info["bads"])  # u get the bad channels IF marked beforehand
            
            # """
            # # first run an algoritm to detect bads:
            # # works only on epochs!? should I do this on epochs Im so confuseddddd
            # # aNYWAY check the MNE post u made
            # raw_check_bads = raw.copy()
            
            # auto_reject = AutoReject()
            # raw_no_bads = auto_reject.fit_transform(raw_check_bads)  
            
            # """
            
            # # plot power spectral density
            # raw.plot_psd()
            
            
            # # describe raw data
            # raw.describe()
            # print(type(raw._data))
            # print(raw._data.shape)
            # raw_data = raw.get_data()
            # print(raw_data.shape)
            
            
            ##############################################################################
            ######## CLEANING the data which had problems during data collection #########
            ##############################################################################
            
            # gives you the last second 
            last_sec = raw.times[-1]
            
            if filename == "BTSCZ002_V3_eyes-closed.vhdr":
                print("delete first 5secs")
                raw.crop(tmin=5)
            elif filename == "BTSCZ006_V3_eyes-closed.vhdr":
                print("delete the last 3-4 seconds ")
                tmax = last_sec - 4
                raw.crop(tmax=tmax)
            elif filename == "BTSCZ007_V3_eyes-closed.vhdr":
                print("delete the last 10 seconds")   
                tmax = last_sec - 10
                raw.crop(tmax=tmax)
            elif filename == "BTSCZ009_V1_eyes-closed.vhdr":
                print("delete the first 20 seconds")
                raw.crop(tmin=20)
            elif filename == "BTSCZ017_V1_eyes-closed.vhdr":
                print("delete the first 12 seconds")
                raw.crop(tmin=12)
            elif filename == "BTSCZ026_V1_eyes-open.vhdr":
                print("delete the first 41 seconds")
                raw.crop(tmin=41)
                
                

            
            
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
            
            # Check if "RESP" channel is present
            if 'RESP' in raw.ch_names:
            # Define the channel type mapping
                channel_type_mapping = {
                    'ECG': 'ecg',
                    'RESP': 'resp'
                }
                
                # Set the channel types for "ECG" and "RESP" channels
                raw.set_channel_types(channel_type_mapping)
            else:
                # Define the channel type mapping for only "ECG" channel
                channel_type_mapping = {
                    'ECG': 'ecg'
                }
    
                # Set the channel type for the "ECG" channel
                raw.set_channel_types(channel_type_mapping)

            # Load the easycap-M1 montage (check if its truw by checking the cap)
            montage = mne.channels.make_standard_montage('easycap-M1')
            
            # Apply the montage to your raw data
            raw.set_montage(montage)
            
            # plot montage
            # raw.plot_sensors(show_names=True)
            
            
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
            
            """
            # Filter out irrelevant annotations (you need to define which ones are irrelevant)
            irrelevant_annotations = ['Comment/actiCAP Data On', 'New Segment/']  # Adjust this list as needed

            # find indices of irrelevant annots
            irrelevant_indices = np.where(np.isin(prep_ica_data_v1.annotations.description, irrelevant_annotations))[0]

            # delete irrelevant ones
            prep_ica_data_v1.annotations.delete(irrelevant_indices)

            """

            # Define existing annotations
            existing_annotations = raw.annotations.copy()
            
            # Combine initially existing and R-peaks annotations
            combined_annotations = mne.Annotations(
                onset=np.concatenate((existing_annotations.onset, R_peak_annotations.onset)),
                duration=np.concatenate((existing_annotations.duration, R_peak_annotations.duration)),
                description=np.concatenate((existing_annotations.description, R_peak_annotations.description))
            )
            
            # Apply annotations to raw data
            raw_resampled.set_annotations(combined_annotations)
            
            # inspect annotations
            raw_resampled.annotations.description
            np.unique(raw_resampled.annotations.description, return_counts=True)

            # Filter out irrelevant annotations (you need to define which ones are irrelevant)
            irrelevant_annotations = ['Comment/actiCAP Data On', 'New Segment/', 'Comment/actiCAP USB Power On', 'ControlBox is not connected via USB', 'actiCAP USB Power On', 'Comment/ControlBox is not connected via USB']  # Adjust this list as needed

            # find indices of irrelevant annots
            irrelevant_indices = np.where(np.isin(raw_resampled.annotations.description, irrelevant_annotations))[0]

            # delete irrelevant ones
            raw_resampled.annotations.delete(irrelevant_indices)

            # check whether it worked
            raw_resampled.annotations.description
            print("Remaining annotations:", np.unique(raw_resampled.annotations.description, return_counts=True))
        
            
            
            # Plot ECG with Annotations
            #raw_resampled.plot(n_channels=1, scalings={'ECG': 1e-3})
            #raw_resampled.plot(n_channels=1)
            
                            
            # YAY WORKS WELL UNTIL HERE!
            
            # ecg_events.plot_image(combine="mean")
            
            
                ##############################################################################
                ######## Cut out the breaks in HCT !!!!!!!! #########
                ##############################################################################
               
            # e.g. if filename includes "hct" then do the cut out...

            if task == 'hct':

                # Get event ids of HCT triggers
                events, event_id = mne.events_from_annotations(raw_resampled)
                R_peak_id = event_id['R-peak']
                hct_events = events[events[:, 2] != R_peak_id]

                # Create an empty list to store the EEG data segments
                eeg_segments = []

                for i in range(0, len(hct_events), 2):
                    raw_copy = raw_resampled.copy()
                    raw_part = raw_copy.crop(tmin=hct_events[i][0] / raw_resampled.info['sfreq'], tmax=hct_events[i+1][0] / raw_resampled.info['sfreq']) # get tmin, tmax in seconds
                    print(raw_part)
                    eeg_segments.append(raw_part)# Combine raw_parts into a single raw object
                mne.concatenate_raws(eeg_segments)
                raw_resampled = eeg_segments[0]

            
           

            
            # %% 6. Separate the EEG from ECG & RESP data
            
            # Create a copy of the original RawBrainVision object
            raw_eeg = raw_resampled.copy()
            
            # # Remove non-EEG channels
            # if 'RESP' in raw_eeg.ch_names:
            #     raw_eeg.drop_channels(['ECG', 'RESP'])
            # else:
            #     raw_eeg.drop_channels(['ECG'])
                
            # extract total nr of eeg channels 
            # total_electrode_nr = len(raw_eeg.ch_names)
            eeg_chans = raw_eeg.copy().pick(picks=["eeg"])
            total_electrode_nr = len(eeg_chans.ch_names)

            
            # %% 7. Preprocess the EEG data: DO NOT INCLUDE the ECG ND RESP channel here !!
            
            # ### A. Remove line noise
            # Apply notch filter to remove line noise (e.g., 50 Hz from Antonin's manuscript)
            line_freq = 50  # Set the line frequency to 50, as Antonin did: 
            raw_resampled_line= raw_eeg.copy()
            raw_resampled_line.notch_filter(freqs=line_freq, picks=["eeg"])  # Apply notch filter to EEG channels only ?? OR: 49.5 to 50.5 in a method ??
            # Plot the data to visualize the effect of the notch filter
            # raw_resampled_line.plot_psd()
            
            
            # ### B.Robust average rereferencing
            raw_resampled_line_reref = raw_resampled_line.copy()
            raw_resampled_line_reref.set_eeg_reference(ref_channels='average')
            # raw_resampled_line_reref.plot()
            
            
            # ### C. Detect & interpolate noisy channels
            raw_resampled_line_reref_interp = raw_resampled_line_reref.copy()
            # Assign the mne object to the NoisyChannels class. The resulting object will be the place where all following methods are performed.
            noisy_data = NoisyChannels(raw_resampled_line_reref_interp, random_state=1337) # for NoisyChannels this Version: only EEG channels are supported and any non-EEG channels in the provided data will be ignored.
            
            ## 2 options here, comment one out and make sure to adapt the output! you can also try without ransac ....
            
            ## 1
            # find bad by corr
            noisy_data.find_bad_by_correlation()
            print(noisy_data.bad_by_correlation)
            # find bad by deviation
            noisy_data.find_bad_by_deviation()
            print(noisy_data.bad_by_deviation)
            #find bad by ransac: finds nothing, do I first have to mark bads from the methods before? UPDATE: It did find stuff, I am now trying to do without to be more liberal and see how many session/person get excluded
            #noisy_data.find_bad_by_ransac(channel_wise=True, max_chunk_size=1) 
            #print(noisy_data.bad_by_ransac)
        
            ## 2
            # find all bads
            # noisy_data.find_all_bads(ransac=True, channel_wise=False, max_chunk_size=None)
            
            # get channel names marked as bad and assign them into bads of the data from the step before
            raw_resampled_line_reref_interp.info["bads"] = noisy_data.get_bads()
            bads = noisy_data.get_bads() 
            rejected_electrode_nr = len(bads)
            # all bad in a string to record them in csv 
            interpolated_bads_str = ', '.join(bads)
            # calculate % bad electrodes
            percent_rejected_electrode = (rejected_electrode_nr / total_electrode_nr) * 100

            # Interpolate noisy Channels
            raw_resampled_line_reref_interp.interpolate_bads() 
            
            
            # ### D. Bandpass filter [0.3  45]: Do this before all other steps?
            raw_resampled_line_reref_interp_filt = raw_resampled_line_reref_interp.copy()
            # Define the bandpass filter frequency range
            low_freq = 0.3  # Lower cutoff frequency (in Hz)
            #low_freq = 1
            high_freq = 45.0  # Upper cutoff frequency (in Hz)
            bandpass_filter = (low_freq, high_freq)
            # Apply the bandpass filter
            raw_resampled_line_reref_interp_filt.filter(l_freq=low_freq, h_freq=high_freq, method='fir', phase='zero', picks=["eeg"]) # check method and phase: they are defaults
            
            
            ######  CSV
            
            # participant id should be BTSCZ...
            participant_id = f"BTSCZ{participant_no}"
            
            # record end_time 
            end_time = datetime.datetime.now()
            
            # Calculate the duration of the analysis
            duration = end_time - start_time
           
            # Create a dictionary representing the new row
            new_row = pd.Series({'subject_id': participant_id, 
                                 'session': session, 
                                 'task': task,
                                 'montage': "make_standard_montage('easycap-M1')",
                                 'new_sampling_rate': new_sampling_rate,
                                 'removed_line_noise_freq': line_freq,
                                 'rereferencing': 'robust average rereferencing',
                                 'total_electrode_nr': total_electrode_nr,
                                 'interpolated_electrode_nr': rejected_electrode_nr,
                                 'percent_interpolated_electrode': percent_rejected_electrode,
                                 'interpolated_chans': interpolated_bads_str,
                                 'bad_chan_detect_method': 'corr, dev ran sequentially, interpolation at last', # or.. 'find_all_bads(), includes corr, dev, ransac..' OR 'corr, dev, ransac ran sequentially, interpolation at last',
                                 'bandpass_filtering': bandpass_filter,
                                 'bandpass_filter_method': 'fir',
                                 'start_time_of_analysis': start_time,
                                 'analysis_duration': duration
                                 })
           
            # convert row to df
            new_row =  new_row.to_frame().T

            # add to existing df the current data outputs
            prep_outputs = pd.concat([prep_outputs, new_row], ignore_index=True)

            # Print the DataFrame
            print(prep_outputs)
            
            
            # E. SAVE the Preprocessed Data & Prep Ouputs in a CSV
            
            ### Data 
            # Specify the folder name
            folder_name = 'Preprocessed_until_ICA' 
            # Construct file name based on the initial raw file's name 
            filename = os.path.splitext(os.path.basename(raw.filenames[0]))[0]   # Specify the desired file name (without extension)
            # Construct the full file path
            file_path = os.path.join(os.getcwd(), folder_name, filename + '_prep_until_ICA.fif')
            # Save the preprocessed data
            raw_resampled_line_reref_interp_filt.save(file_path, overwrite=True)
            
            ### CSV 
            # Construct the csv path
            csv_filename = 'prep_interp_output_bads_corr_dev.csv'    # or  'prep_interp_output'
            csv_path = os.path.join(os.getcwd(), folder_name,csv_filename)
            prep_outputs.to_csv(csv_path, mode='w', sep=',', index=False)
            
            
            ### PSD Plot
            psd_plot = raw_resampled_line_reref_interp_filt.plot_psd()
            psd_folder_name = 'PSDs' 
            psd_file_path = os.path.join(os.getcwd(), folder_name, psd_folder_name, filename + '_prep_until_ICA_psd_no_ransac.jpg')
            psd_plot.savefig(psd_file_path, format='jpg')
            

        
            # Decided instead to have a csv, if u undecide, put it back in the loop...
            # # Also save interpolated bads in a text file, can also try to append the .txt to make a csv including all bads of all subjects
            # # interpolated_bads_str = ', '.join(bads)
            # bads_path = os.path.join(os.getcwd(), folder_name, filename + '_interpolated_bads.txt')
            # with open(bads_path, 'w') as file:
            #     file.write(interpolated_bads_str)
            




# From here on, transfer to script 2
