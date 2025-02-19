"""
Script: extract_parameters.py

Description:
    This script defines the `extract_swimming_parameters` and 
    `extract_joint_angles_only` functions, which take the output of
    'swimming_algorithm' to compute relevant parameters and store 
    them in CSV fildes for swimming analysis. 
    
    `extract_swimming_parameters` extracts the following parameters:
    - Number of strokes
    - Time per 5m of swimming
    - Stroke rate
    - Stroke duration (mean, std)
    - Distance per stroke
    - Ankle, knee, and hip joint angles (min, max, range)
    - Range of motion (RoM)
    - Ankle displacement (horizontal, vertical, sidewards)
    - Angular component of the coefficient of correspondence or ACC (E. Field-Fote et al., 2002)
    - Sum of squared distances or SSD (Awai et al., 2014)
    - Asymmetry SSD
    - Percentage asymmetry
    - Phase shift (mean, std)

Dependencies:
    - Requires 'swimming_algorithm', and 'function' modules.

Usage:
    - This script is not meant to be run directly - the functions are imported 
      into the main processing pipeline.

Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import csv
import pandas as pd
import numpy as np
from src.functions import *
from src.swimming_algorithm import *


def extract_swimming_parameters(swimming_data, results_dir, file, participant_folder, ref_data, create_files):
    """
    Extracts swimming parameters from the output of swimming_algorithm and saves them to CSV files.
    
    Parameters:
    swimming_data (dict): Output of swimming_algorithm containing relevant data.
    results_dir (str): Path to the directory where results should be saved.
    file (str): Filename identifier for the participant.
    participant_folder (str): Foldername identifier for the participant.
    ref_data: Healthy reference data for hip-knee and ankle-knee angles.
    create_files (boolian): Saves data to CSV files if True.
    
    """

    # Unpack relevant data from swimming_data
    reference_hip_knee, reference_ankle_knee = ref_data[0], ref_data[1]

    freq = swimming_data['freq']
    trial_nr = swimming_data['trial_nr']
    acc_per_trial = swimming_data['acc_per_trial']
    x_starts = swimming_data['x_starts']
    ankle_nr_L = swimming_data['ankle_nr_L']
    ankle_nr_R = swimming_data['ankle_nr_R']
    time_list = swimming_data['time_list']

    flexion_extension_hip_L_per_trial = swimming_data['flexion_extension_hip_L_per_trial']
    flexion_extension_hip_R_per_trial = swimming_data['flexion_extension_hip_R_per_trial']
    adduction_abduction_hip_L_per_trial = swimming_data['adduction_abduction_hip_L_per_trial']
    adduction_abduction_hip_R_per_trial = swimming_data['adduction_abduction_hip_R_per_trial']
    flexion_extension_knee_L_per_trial = swimming_data['flexion_extension_knee_L_per_trial']
    flexion_extension_knee_R_per_trial = swimming_data['flexion_extension_knee_R_per_trial']
    flexion_extension_ankle_L_per_trial = swimming_data['flexion_extension_ankle_L_per_trial']
    flexion_extension_ankle_R_per_trial = swimming_data['flexion_extension_ankle_R_per_trial']

    # Extract absolute joint angles
    angle_ankle_L_per_trial = swimming_data['angle_ankle_L_per_trial']
    angle_ankle_R_per_trial = swimming_data['angle_ankle_R_per_trial']
    angle_knee_L_per_trial = swimming_data['angle_knee_L_per_trial']
    angle_knee_R_per_trial = swimming_data['angle_knee_R_per_trial']
    angle_hip_L_per_trial = swimming_data['angle_hip_L_per_trial']
    angle_hip_R_per_trial = swimming_data['angle_hip_R_per_trial']
    
    # Set minimum time between strokes (1 frame)
    min_distance = 1
    
    # Set cutoff for Butterworth filter
    cut_off = 0.01
    
    # Initialize all necessary lists
    Strokes_per_trial = []
    Time_per_trial = []
    Stroke_rate_per_trial = []
    Distance_per_stroke_per_trial = []
    Variability_stroke_duration_per_trial_L = []
    Variability_stroke_duration_per_trial_R = []
    Stroke_duration_per_trial_L = []
    Stroke_duration_per_trial_R = []

    Horizontal_displacement_L_per_trial = []
    Vertical_displacement_L_per_trial = []
    Sidewards_displacement_L_per_trial = []
    Horizontal_displacement_R_per_trial = []
    Vertical_displacement_R_per_trial = []
    Sidewards_displacement_R_per_trial = []

    RoM_ankle_L_per_trial = []
    RoM_ankle_R_per_trial = []
    RoM_knee_L_per_trial = []
    RoM_knee_R_per_trial = []
    RoM_hip_L_per_trial = []
    RoM_hip_R_per_trial = []
    RoM_hip_L_adduction_abduction_per_trial = []
    RoM_hip_R_adduction_abduction_per_trial = []

    Min_ankle_L_per_trial = []
    Max_ankle_L_per_trial = []
    Min_ankle_R_per_trial = []
    Max_ankle_R_per_trial = []
    Min_knee_L_per_trial = []
    Max_knee_L_per_trial = []
    Min_knee_R_per_trial = []
    Max_knee_R_per_trial = []
    Min_hip_L_per_trial = []
    Max_hip_L_per_trial = []
    Min_hip_R_per_trial = []
    Max_hip_R_per_trial = []
    Min_coronal_hip_L_per_trial = []
    Max_coronal_hip_L_per_trial = []
    Min_coronal_hip_R_per_trial = []
    Max_coronal_hip_R_per_trial = []

    ACC_ankle_knee_per_trial_L = []
    ACC_ankle_knee_per_trial_R = []
    ACC_ankle_knee_std_per_trial_L = []
    ACC_ankle_knee_std_per_trial_R = []
    ACC_hip_knee_per_trial_L = []
    ACC_hip_knee_per_trial_R = []
    ACC_hip_knee_std_per_trial_L = []
    ACC_hip_knee_std_per_trial_R = []

    SSD_hip_knee_per_trial_L = []
    SSD_hip_knee_per_trial_R = []
    SSD_ankle_knee_per_trial_L = []
    SSD_ankle_knee_per_trial_R = []

    Phase_shift_per_trial = []
    Phase_shift_std_per_trial = []

    Asymmetry_SSD_hip_knee_per_trial = []
    Asymmetry_SSD_ankle_knee_per_trial = []
    Asymmetry_SSD_hip_ankle_per_trail = []
    Asymmetry_hip_per_trial = []
    Asymmetry_knee_per_trial = []
    Asymmetry_ankle_per_trial = []

    mean_ankle_per_trial = []
    mean_knee_per_trial = []
    mean_hip_per_trial = []
    mean_hip_add_abd_per_trial = []

    # Detect strokes and compute parameters
    for trial in range(trial_nr):

        # Sides to consider
        sides = ['Left', 'Right']

        # Felxion/extension
        angle_hip_per_trial = [flexion_extension_hip_L_per_trial, flexion_extension_hip_R_per_trial]
        angle_hip_add_abd_per_trial = [adduction_abduction_hip_L_per_trial, adduction_abduction_hip_R_per_trial]
        angle_knee_per_trial = [flexion_extension_knee_L_per_trial, flexion_extension_knee_R_per_trial]
        angle_ankle_per_trial = [flexion_extension_ankle_L_per_trial, flexion_extension_ankle_R_per_trial]

        # Define variables per side
        stroke_nr_per_side = []
        average_stroke_duration_per_side = []
        variability_stroke_duration_per_side = []
        stroke_rate_per_side = []

        for side in range(len(sides)):
            peaks, peak_values = find_stroke_peaks(angle_knee_per_trial[side][trial], 
                                                   cut_off, 
                                                   min_distance, 
                                                   plot=False)
            strokes = detect_swimming_strokes(angle_knee_per_trial[side][trial], 
                                              peaks, 
                                              peak_values, 
                                              plot=False)

        # Initialize variables
        angles_filtered_ankle_L = []
        angles_filtered_ankle_R = []
        angles_filtered_knee_L = []
        angles_filtered_knee_R = []
        angles_filtered_hip_L = []
        angles_filtered_hip_R = []
        angles_filtered_hip_L_coronal = []
        angles_filtered_hip_R_coronal = []

        i = 0
        # Filter angles to only include valid strokes
        for stroke in strokes:
            # Ankle
            angles_filtered_ankle_L.extend(flexion_extension_ankle_L_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_ankle_R.extend(flexion_extension_ankle_R_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            # Knee
            angles_filtered_knee_L.extend(flexion_extension_knee_L_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_knee_R.extend(flexion_extension_knee_R_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            # Hip
            angles_filtered_hip_L.extend(flexion_extension_hip_L_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_hip_R.extend(flexion_extension_hip_R_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_hip_L_coronal.extend(adduction_abduction_hip_L_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_hip_R_coronal.extend(adduction_abduction_hip_R_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            i=+1

        '--------------------------- Hip ---------------------------'
        max_angle_hip_L = np.percentile(angles_filtered_hip_L, 95)
        min_angle_hip_L = np.percentile(angles_filtered_hip_L, 5)
        max_angle_hip_R = np.percentile(angles_filtered_hip_R, 95)
        min_angle_hip_R = np.percentile(angles_filtered_hip_R, 5)

        RoM_hip_L = max_angle_hip_L - min_angle_hip_L
        RoM_hip_R = max_angle_hip_R - min_angle_hip_R

        max_angle_coronal_hip_L = np.percentile(angles_filtered_hip_L_coronal, 95)
        min_angle_coronal_hip_L = np.percentile(angles_filtered_hip_L_coronal, 5)
        max_angle_coronal_hip_R = np.percentile(angles_filtered_hip_R_coronal, 95)
        min_angle_coronal_hip_R = np.percentile(angles_filtered_hip_R_coronal, 5)

        RoM_hip_L_adduction_abduction = max_angle_coronal_hip_L - min_angle_coronal_hip_L
        RoM_hip_R_adduction_abduction = max_angle_coronal_hip_R - min_angle_coronal_hip_R

        '--------------------------- Knee ---------------------------'

        max_angle_knee_L = np.percentile(angles_filtered_knee_L, 95)
        min_angle_knee_L = np.percentile(angles_filtered_knee_L, 5)
        max_angle_knee_R = np.percentile(angles_filtered_knee_R, 95)
        min_angle_knee_R = np.percentile(angles_filtered_knee_R, 5)

        RoM_knee_L = max_angle_knee_L - min_angle_knee_L
        RoM_knee_R = max_angle_knee_R - min_angle_knee_R
        
        '--------------------------- Ankle ---------------------------'

        max_angle_ankle_L = np.percentile(angles_filtered_ankle_L, 95)
        min_angle_ankle_L = np.percentile(angles_filtered_ankle_L, 5)
        max_angle_ankle_R = np.percentile(angles_filtered_ankle_R, 95)
        min_angle_ankle_R = np.percentile(angles_filtered_ankle_R, 5)

        RoM_ankle_L = max_angle_ankle_L - min_angle_ankle_L
        RoM_ankle_R = max_angle_ankle_R - min_angle_ankle_R

        RoM_ankle_L_per_trial.append(RoM_ankle_L)
        RoM_ankle_R_per_trial.append(RoM_ankle_R)
        RoM_knee_L_per_trial.append(RoM_knee_L)
        RoM_knee_R_per_trial.append(RoM_knee_R)
        RoM_hip_L_per_trial.append(RoM_hip_L)
        RoM_hip_R_per_trial.append(RoM_hip_R)
        RoM_hip_L_adduction_abduction_per_trial.append(RoM_hip_L_adduction_abduction)
        RoM_hip_R_adduction_abduction_per_trial.append(RoM_hip_R_adduction_abduction)

        Min_ankle_L_per_trial.append(min_angle_ankle_L)
        Max_ankle_L_per_trial.append(max_angle_ankle_L)
        Min_ankle_R_per_trial.append(min_angle_ankle_R)
        Max_ankle_R_per_trial.append(max_angle_ankle_R)
        Min_knee_L_per_trial.append(min_angle_knee_L)
        Max_knee_L_per_trial.append(max_angle_knee_L)
        Min_knee_R_per_trial.append(min_angle_knee_R)
        Max_knee_R_per_trial.append(max_angle_knee_R)
        Min_hip_L_per_trial.append(min_angle_hip_L)
        Max_hip_L_per_trial.append(max_angle_hip_L)
        Min_hip_R_per_trial.append(min_angle_hip_R)
        Max_hip_R_per_trial.append(max_angle_hip_R)
        Min_coronal_hip_L_per_trial.append(min_angle_coronal_hip_L)
        Max_coronal_hip_L_per_trial.append(max_angle_coronal_hip_L)
        Min_coronal_hip_R_per_trial.append(min_angle_coronal_hip_R)
        Max_coronal_hip_R_per_trial.append(max_angle_coronal_hip_R)



        ########################################### EXTRACT DATA PER STROKE ###########################################


        ankles = [ankle_nr_L, ankle_nr_R]

        # Define variables per stroke
        displacement_per_stroke_sidewards = []
        displacement_per_stroke_horizontal = []
        displacement_per_stroke_vertical = []

        strokes_filtered_per_side = []

        ACC_ankle_knee_per_side = []
        ACC_ankle_knee_std_per_side = []
        ACC_ankle_hip_per_side = []
        ACC_ankle_hip_std_per_side = []
        ACC_hip_knee_per_side = []
        ACC_hip_knee_std_per_side = []

        SSD_ankle_knee_per_side = []
        SSD_hip_knee_per_side = []

        patient_hip_knee_per_side = []
        patient_ankle_knee_per_side = []
        patient_ankle_hip_per_side = []

        mean_ankle_per_side = []
        mean_knee_per_side = []
        mean_hip_per_side = []
        mean_hip_add_abd_per_side = []

        # Filter angles to remove noise
        b, a = butter(3, cut_off, 'lowpass')
        angle_knee_per_trial[0][trial] = filtfilt(b, a, angle_knee_per_trial[0][trial])
        angle_knee_per_trial[1][trial] = filtfilt(b, a, angle_knee_per_trial[1][trial])
        angle_hip_per_trial[0][trial] = filtfilt(b, a, angle_hip_per_trial[0][trial])
        angle_hip_per_trial[1][trial] = filtfilt(b, a, angle_hip_per_trial[1][trial])

        for side in range(len(sides)):
            peaks, peak_values = find_stroke_peaks(angle_knee_per_trial[side][trial], 
                                                   cut_off, 
                                                   min_distance, 
                                                   plot=False)
            strokes = detect_swimming_strokes(angle_knee_per_trial[side][trial], 
                                              peaks, 
                                              peak_values, 
                                              plot=False)    

            # Define trail variables per stroke
            velocity_per_stroke = []
            displacement_per_stroke = []
            angle_ankle_per_stroke = []
            angle_knee_per_stroke = []
            angle_hip_per_stroke = []
            angle_hip_add_abd_per_stroke = []

            # Extract gyroscope and accelerometer data per stroke
            for stroke in strokes:

                # Start and stop times of stroke within the trial data
                [stroke_start, stroke_stop] = stroke
                acc_per_stroke = acc_per_trial[trial][ankles[side]][stroke_start:stroke_stop]

                # Start and stop times of stroke within the whole data
                stroke_start_total = stroke_start + int(x_starts[trial] * 60 * 60 * freq)
                stroke_stop_total = stroke_stop + int(x_starts[trial] * 60 * 60 * freq)

                # Frames during which the stroke event occurs
                time_array = np.array(time_list[stroke_start_total:stroke_stop_total])

                # Integrate twice to get displacement
                velocity = integrate_acceleration(acc_per_stroke, time_array)
                displacement = integrate_velocity(velocity, time_array)

                velocity_per_stroke.append(velocity)
                displacement_per_stroke.append(displacement)

                angle_ankle_per_stroke.append(angle_ankle_per_trial[side][trial][stroke_start:stroke_stop])
                angle_knee_per_stroke.append(angle_knee_per_trial[side][trial][stroke_start:stroke_stop])
                angle_hip_per_stroke.append(angle_hip_per_trial[side][trial][stroke_start:stroke_stop])
                angle_hip_add_abd_per_stroke.append(angle_hip_add_abd_per_trial[side][trial][stroke_start:stroke_stop])

            # Remove outliers
            strokes_filtered, displacement_per_stroke_filtered = remove_outliers_cyclogram(strokes,
                                                                                           displacement_per_stroke,
                                                                                           side)
            strokes_filtered_per_side.append(strokes_filtered)

            # Remove stroke outliers from angles
            strokes_filtered_indices = [index for index, stroke in enumerate(strokes) if stroke in strokes_filtered]
            angle_ankle_per_stroke_filtered = [angle_ankle_per_stroke[index] for index in strokes_filtered_indices]
            angle_knee_per_stroke_filtered = [angle_knee_per_stroke[index] for index in strokes_filtered_indices]
            angle_hip_per_stroke_filtered = [angle_hip_per_stroke[index] for index in strokes_filtered_indices]
            angle_hip_add_abd_per_stroke_filtered = [angle_hip_add_abd_per_stroke[index] for index in strokes_filtered_indices]
            
            # Hip-knee
            mean_hip_centered, mean_knee_centered, mean_hip, mean_knee = center_angles(angle_hip_per_stroke_filtered, 
                                                                                       angle_knee_per_stroke_filtered)
                                
            patient_hip_knee_centered = [mean_hip_centered, mean_knee_centered]
            
            # Ankle-knee
            mean_ankle_centered, mean_knee_centered, mean_ankle, _ = center_angles(angle_ankle_per_stroke_filtered,
                                                                                   angle_knee_per_stroke_filtered)  
                                                
            patient_ankle_knee_centered = [mean_ankle_centered, mean_knee_centered]

            # Ankle-hip
            mean_ankle_centered, mean_hip_centered, _, _ = center_angles(angle_ankle_per_stroke_filtered,
                                                                         angle_hip_per_stroke_filtered)  
            
            patient_ankle_hip_centered = [mean_ankle_centered, mean_hip_centered]

            # Get mean hip adduction / abduction
            _, _, mean_hip_add_abd, _ = center_angles(angle_hip_add_abd_per_stroke_filtered,
                                                      angle_hip_per_stroke_filtered)

            patient_hip_knee_per_side.append(patient_hip_knee_centered)
            patient_ankle_knee_per_side.append(patient_ankle_knee_centered)
            patient_ankle_hip_per_side.append(patient_ankle_hip_centered)                             
                
            mean_ankle_per_side.append(mean_ankle)
            mean_knee_per_side.append(mean_knee)
            mean_hip_per_side.append(mean_hip)
            mean_hip_add_abd_per_side.append(mean_hip_add_abd)

            # Save angles to csv file (for SSD and angle plots)
            first_row = ['Participant_' +  file, 'Trial' + str(trial+1), sides[side], 'Knee'] + mean_knee
            second_row = ['Participant_' +  file, 'Trial' + str(trial+1), sides[side], 'Hip'] + mean_hip
            third_row = ['Participant_' +  file, 'Trial' + str(trial+1), sides[side], 'Ankle'] + mean_ankle
            forth_row = ['Participant_' +  file, 'Trial' + str(trial+1), sides[side], 'Hip_add_abd'] + mean_hip_add_abd

            if create_files:
                csv_file_path = os.path.join(results_dir, f'Joint_Angles_{participant_folder}.csv') 
                # Read existing data if the file exists
                existing_data = []
                if os.path.exists(csv_file_path):
                    with open(csv_file_path, 'r', newline='') as data_file:
                        reader = csv.reader(data_file)
                        existing_data = list(reader)

                # Extract existing participant and trial entries
                existing_participant_trials = [
                    (row[0], row[1], row[2]) for row in existing_data if len(row) >= 2
                ]

                # Define the new participant and trial key
                new_participant = 'Participant_' + file
                new_trial = 'Trial' + str(trial+1)
                new_side = sides[side]

                # Check if the participant and trial are already in the file
                if any([
                    existing_participant_trials[entry][0] == new_participant and
                    existing_participant_trials[entry][1] == new_trial and
                    existing_participant_trials[entry][2] == new_side
                    for entry in range(len(existing_participant_trials))
                ]):
                    pass
                else:  
                    with open(csv_file_path, 'a', newline='') as data_file:
                        writer = csv.writer(data_file)
                        writer.writerow(first_row)
                        writer.writerow(second_row)
                        writer.writerow(third_row)
                        writer.writerow(forth_row)
            
            # Calculate ACC
            ACC_ankle_hip, ACC_ankle_hip_std = calculate_ACC(angle_ankle_per_stroke_filtered, 
                                                            angle_hip_per_stroke_filtered, 
                                                            strokes_filtered)
            ACC_hip_knee, ACC_hip_knee_std = calculate_ACC(angle_hip_per_stroke_filtered,
                                                        angle_knee_per_stroke_filtered,  
                                                        strokes_filtered)
            ACC_ankle_knee, ACC_ankle_knee_std = calculate_ACC(angle_ankle_per_stroke_filtered, 
                                                            angle_knee_per_stroke_filtered,
                                                            strokes_filtered)

            ACC_ankle_knee_per_side.append(ACC_ankle_knee)
            ACC_ankle_knee_std_per_side.append(ACC_ankle_knee_std)
            ACC_ankle_hip_per_side.append(ACC_ankle_hip)
            ACC_ankle_hip_std_per_side.append(ACC_ankle_hip_std)
            ACC_hip_knee_per_side.append(ACC_hip_knee)
            ACC_hip_knee_std_per_side.append(ACC_hip_knee_std)

            # Calculate SSD
            SSD_hip_knee = calculate_SSD(patient_hip_knee_centered[0], patient_hip_knee_centered[1], 
                                         reference_hip_knee[0], reference_hip_knee[1])
            SSD_ankle_knee = calculate_SSD(patient_ankle_knee_centered[0], patient_ankle_knee_centered[1],
                                           reference_ankle_knee[0], reference_ankle_knee[1])
            SSD_hip_knee_per_side.append(SSD_hip_knee)
            SSD_ankle_knee_per_side.append(SSD_ankle_knee)

            # Temporary variables
            sidewards_temp = []
            horizontal_temp = []
            vertical_temp = []

            # Maximum displacement for filtered strokes
            for stroke in range(len(strokes_filtered)):
                sidewards_temp.append(np.abs(displacement_per_stroke_filtered[stroke][:, 0]).max(0))
                horizontal_temp.append(np.abs(displacement_per_stroke_filtered[stroke][:, 1]).max(0))
                vertical_temp.append(np.abs(displacement_per_stroke_filtered[stroke][:, 2]).max(0))
            
            displacement_per_stroke_sidewards.append(sidewards_temp)
            displacement_per_stroke_horizontal.append(horizontal_temp)
            displacement_per_stroke_vertical.append(vertical_temp)

            # Trial time filtered
            durations_per_side = []
            for strokes in strokes_filtered_per_side:
                first_frame = strokes[0][0]
                last_frame = strokes[-1][-1]
                durations_per_side.append((last_frame - first_frame)/freq)
            filtered_duration = min(durations_per_side)

            # Finde number of strokes
            stroke_nr = len(strokes_filtered)
            stroke_nr_per_side.append(stroke_nr)

            # Find time per stroke (200 timesteps = 1 second) --> STROKE DURATION
            stroke_durations = []

            # Using filtered strokes
            for stroke in strokes_filtered:
                stroke_duration = (stroke[1] - stroke[0]) / 200  # [sec]
                stroke_durations.append(stroke_duration)

            average_stroke_duration = np.mean(stroke_durations)
            average_stroke_duration_per_side.append(average_stroke_duration)

            variability_stroke_duration = np.std(stroke_durations)
            variability_stroke_duration_per_side.append(variability_stroke_duration)

            # Find number of strokes per minute --> CADENCE or STROKE RATE
            stroke_rate = stroke_nr / filtered_duration * 60  # [strokes/min]
            stroke_rate_per_side.append(stroke_rate)

        mean_ankle_per_trial.append(mean_ankle_per_side)
        mean_knee_per_trial.append(mean_knee_per_side)
        mean_hip_per_trial.append(mean_hip_per_side)
        mean_hip_add_abd_per_trial.append(mean_hip_add_abd_per_side)
        
        # Separate left and right
        displacement_per_stroke_sidewards_L = displacement_per_stroke_sidewards[0]
        displacement_per_stroke_sidewards_R = displacement_per_stroke_sidewards[1]
        displacement_per_stroke_horizontal_L = displacement_per_stroke_horizontal[0]
        displacement_per_stroke_horizontal_R = displacement_per_stroke_horizontal[1]
        displacement_per_stroke_vertical_L = displacement_per_stroke_vertical[0]
        displacement_per_stroke_vertical_R = displacement_per_stroke_vertical[1]

        # Parameters for CSV file (average of each parameter over left and right)
        Strokes_per_trial.append(np.round(np.sum(stroke_nr_per_side)/2, decimals=0))
        Time_per_trial.append(filtered_duration)
        Stroke_rate_per_trial.append(np.sum(stroke_rate_per_side)/2)
        Distance_per_stroke_per_trial.append(5/Strokes_per_trial[trial])
        Variability_stroke_duration_per_trial_L.append(variability_stroke_duration_per_side[0])
        Variability_stroke_duration_per_trial_R.append(variability_stroke_duration_per_side[1])
        Stroke_duration_per_trial_L.append(average_stroke_duration_per_side[0])
        Stroke_duration_per_trial_R.append(average_stroke_duration_per_side[1])

        Horizontal_displacement_L_per_trial.append(np.mean(displacement_per_stroke_horizontal_L))
        Vertical_displacement_L_per_trial.append(np.mean(displacement_per_stroke_vertical_L))
        Sidewards_displacement_L_per_trial.append(np.mean(displacement_per_stroke_sidewards_L))
        Horizontal_displacement_R_per_trial.append(np.mean(displacement_per_stroke_horizontal_R))
        Vertical_displacement_R_per_trial.append(np.mean(displacement_per_stroke_vertical_R))
        Sidewards_displacement_R_per_trial.append(np.mean(displacement_per_stroke_sidewards_R))

        # Sparate ACC into L and R
        ACC_ankle_knee_per_trial_L.append(ACC_ankle_knee_per_side[0])
        ACC_ankle_knee_per_trial_R.append(ACC_ankle_knee_per_side[1])
        ACC_ankle_knee_std_per_trial_L.append(ACC_ankle_knee_std_per_side[0])
        ACC_ankle_knee_std_per_trial_R.append(ACC_ankle_knee_std_per_side[1])
        ACC_hip_knee_per_trial_L.append(ACC_hip_knee_per_side[0])
        ACC_hip_knee_per_trial_R.append(ACC_hip_knee_per_side[1])
        ACC_hip_knee_std_per_trial_L.append(ACC_hip_knee_std_per_side[0])
        ACC_hip_knee_std_per_trial_R.append(ACC_hip_knee_std_per_side[1])

        # Sparate SSD into L and R
        SSD_hip_knee_per_trial_L.append(SSD_hip_knee_per_side[0])
        SSD_hip_knee_per_trial_R.append(SSD_hip_knee_per_side[1])
        SSD_ankle_knee_per_trial_L.append(SSD_ankle_knee_per_side[0])
        SSD_ankle_knee_per_trial_R.append(SSD_ankle_knee_per_side[1])

        # Separate L and R
        patient_hip_knee_L = patient_hip_knee_per_side[0]
        patient_hip_knee_R = patient_hip_knee_per_side[1]
        patient_ankle_knee_L = patient_ankle_knee_per_side[0]
        patient_ankle_knee_R = patient_ankle_knee_per_side[1]
        patient_ankle_hip_L = patient_ankle_hip_per_side[0]
        patient_ankle_hip_R = patient_ankle_hip_per_side[1]
        
        # Asymmetry SSD
        Asymmetry_SSD_hip_knee_per_trial.append(calculate_SSD(patient_hip_knee_L[0], patient_hip_knee_L[1], 
                                                            patient_hip_knee_R[0], patient_hip_knee_R[1]))
        Asymmetry_SSD_ankle_knee_per_trial.append(calculate_SSD(patient_ankle_knee_L[0], patient_ankle_knee_L[1], 
                                                                patient_ankle_knee_R[0], patient_ankle_knee_R[1]))
        Asymmetry_SSD_hip_ankle_per_trail.append(calculate_SSD(patient_ankle_hip_L[0], patient_ankle_hip_L[1], 
                                                            patient_ankle_hip_R[0], patient_ankle_hip_R[1]))
        
        Asymmetry_hip_per_trial.append(np.mean(np.abs((np.array(angle_hip_L_per_trial[trial]) 
                                                    - np.array(angle_hip_R_per_trial[trial]))
                                                    / np.array(angle_hip_L_per_trial[trial]) * 100)))
        Asymmetry_knee_per_trial.append(np.mean(np.abs((np.array(angle_knee_L_per_trial[trial]) 
                                                        - np.array(angle_knee_R_per_trial[trial]))
                                                        / np.array(angle_knee_L_per_trial[trial]) * 100)))
        Asymmetry_ankle_per_trial.append(np.mean(np.abs((np.array(angle_ankle_L_per_trial[trial]) 
                                                        - np.array(angle_ankle_R_per_trial[trial]))
                                                        / np.array(angle_ankle_L_per_trial[trial]) * 100)))
        
        # Separate L and R 
        strokes_filtered_L = strokes_filtered_per_side[0]
        strokes_filtered_R = strokes_filtered_per_side[1]

        # Phase shift between left and right
        phase_shift_per_stroke = []
        for stroke_L in strokes_filtered_L:
            # Find corresponding right stroke (if any) for each left stroke
            stroke_R = select_corresponding_stroke(stroke_L, strokes_filtered_R)

            if stroke_R:
                # Calculate percentage difference in stroke start
                duration_L = stroke_L[1] - stroke_L[0]
                diff = abs(stroke_L[0] - stroke_R[0])
                percentage_diff = (diff / duration_L) * 100

                phase_shift_per_stroke.append(percentage_diff)

        Phase_shift_per_trial.append(np.mean(phase_shift_per_stroke))
        Phase_shift_std_per_trial.append(np.std(phase_shift_per_stroke))

    RoM_hip_L = np.mean(RoM_hip_L_per_trial)
    RoM_hip_R = np.mean(RoM_hip_R_per_trial)
    RoM_hip_L_adduction_abduction = np.mean(RoM_hip_L_adduction_abduction_per_trial)
    RoM_hip_R_adduction_abduction = np.mean(RoM_hip_R_adduction_abduction_per_trial)
    RoM_knee_L = np.mean(RoM_knee_L_per_trial)
    RoM_knee_R = np.mean(RoM_knee_R_per_trial)
    RoM_ankle_L = np.mean(RoM_ankle_L_per_trial)
    RoM_ankle_R = np.mean(RoM_ankle_R_per_trial)

    # Save parameters in CSV file
    if create_files:
        csv_file_path = os.path.join(results_dir, f'Swimming_Parameters_{participant_folder}.csv')
        outcome_parameters = pd.read_csv(csv_file_path, header=0)

        # Add empty column to insure rows are appended below header
        if len(outcome_parameters['Participant']) == 0:
            csv_file = open(csv_file_path, 'a', newline='')
            writer = csv.writer(csv_file)
        if any([outcome_parameters['Participant'][participant] == 'Participant_' +  file 
                for participant in range(len(outcome_parameters['Participant']))]):
            print('Data already saved to file!')
        else:
            csv_file = open(csv_file_path, 'a', newline='')
            writer = csv.writer(csv_file)

            # Parameters for each trial
            for trial in range(trial_nr):
                writer.writerow(['Participant_' +  file, trial+1, Strokes_per_trial[trial], Time_per_trial[trial], 
                                Stroke_rate_per_trial[trial], Stroke_duration_per_trial_L[trial], Stroke_duration_per_trial_R[trial],
                                Variability_stroke_duration_per_trial_L[trial], Variability_stroke_duration_per_trial_R[trial],
                                Distance_per_stroke_per_trial[trial], Min_ankle_L_per_trial[trial],
                                Max_ankle_L_per_trial[trial], Min_ankle_R_per_trial[trial], Max_ankle_R_per_trial[trial],
                                Min_knee_L_per_trial[trial], Max_knee_L_per_trial[trial], Min_knee_R_per_trial[trial],
                                Max_knee_R_per_trial[trial], Min_hip_L_per_trial[trial], Max_hip_L_per_trial[trial],
                                Min_hip_R_per_trial[trial], Max_hip_R_per_trial[trial], Min_coronal_hip_L_per_trial[trial],
                                Max_coronal_hip_L_per_trial[trial], Min_coronal_hip_R_per_trial[trial],
                                Max_coronal_hip_R_per_trial[trial], RoM_hip_L_per_trial[trial], 
                                RoM_hip_R_per_trial[trial], RoM_hip_L_adduction_abduction_per_trial[trial], 
                                RoM_hip_R_adduction_abduction_per_trial[trial], RoM_knee_L_per_trial[trial], 
                                RoM_knee_R_per_trial[trial], RoM_ankle_L_per_trial[trial], RoM_ankle_R_per_trial[trial],
                                Horizontal_displacement_L_per_trial[trial], Horizontal_displacement_R_per_trial[trial],
                                Vertical_displacement_L_per_trial[trial], Vertical_displacement_R_per_trial[trial], 
                                Sidewards_displacement_L_per_trial[trial], Sidewards_displacement_R_per_trial[trial],
                                ACC_hip_knee_per_trial_L[trial], ACC_hip_knee_per_trial_R[trial],
                                ACC_hip_knee_std_per_trial_L[trial], ACC_hip_knee_std_per_trial_R[trial], 
                                ACC_ankle_knee_per_trial_L[trial], ACC_ankle_knee_per_trial_R[trial], 
                                ACC_ankle_knee_std_per_trial_L[trial], ACC_ankle_knee_std_per_trial_R[trial], 
                                SSD_hip_knee_per_trial_L[trial], SSD_hip_knee_per_trial_R[trial], 
                                SSD_ankle_knee_per_trial_L[trial], SSD_ankle_knee_per_trial_R[trial],
                                Asymmetry_SSD_hip_knee_per_trial[trial], Asymmetry_SSD_ankle_knee_per_trial[trial],
                                Asymmetry_hip_per_trial[trial], Asymmetry_knee_per_trial[trial], 
                                Asymmetry_ankle_per_trial[trial], Phase_shift_per_trial[trial], 
                                Phase_shift_std_per_trial[trial]])
            csv_file.close()

    return {"RoM_ankle_L_per_trial": RoM_ankle_L_per_trial,
            "RoM_ankle_R_per_trial": RoM_ankle_R_per_trial,
            "RoM_knee_L_per_trial": RoM_knee_L_per_trial,
            "RoM_knee_R_per_trial": RoM_knee_R_per_trial,
            "RoM_hip_L_per_trial": RoM_hip_L_per_trial,
            "RoM_hip_R_per_trial": RoM_hip_R_per_trial,
            "RoM_hip_L_adduction_abduction_per_trial": RoM_hip_L_adduction_abduction_per_trial,
            "RoM_hip_R_adduction_abduction_per_trial": RoM_hip_R_adduction_abduction_per_trial,
            "mean_ankle_per_trial": mean_ankle_per_trial,
            "mean_knee_per_trial": mean_knee_per_trial,
            "mean_hip_per_trial": mean_hip_per_trial,
            "mean_hip_add_abd_per_trial": mean_hip_add_abd_per_trial}


def extract_joint_angles_only(swimming_data, data_dir, file):
    """
    Extracts joint angles from the output of swimming_algorithm and saves them to CSV files.
    
    Parameters:
    swimming_data (dict): Output of swimming_algorithm containing relevant data.
    data_dir (str): Path to the directory where data should be saved.
    file (str): Filename identifier for the participant.
    """
    # Unpack relevant data from swimming_data
    freq = swimming_data['freq']
    trial_nr = swimming_data['trial_nr']
    acc_per_trial = swimming_data['acc_per_trial']
    x_starts = swimming_data['x_starts']
    ankle_nr_L = swimming_data['ankle_nr_L']
    ankle_nr_R = swimming_data['ankle_nr_R']
    time_list = swimming_data['time_list']

    flexion_extension_hip_L_per_trial = swimming_data['flexion_extension_hip_L_per_trial']
    flexion_extension_hip_R_per_trial = swimming_data['flexion_extension_hip_R_per_trial']
    adduction_abduction_hip_L_per_trial = swimming_data['adduction_abduction_hip_L_per_trial']
    adduction_abduction_hip_R_per_trial = swimming_data['adduction_abduction_hip_R_per_trial']
    flexion_extension_knee_L_per_trial = swimming_data['flexion_extension_knee_L_per_trial']
    flexion_extension_knee_R_per_trial = swimming_data['flexion_extension_knee_R_per_trial']
    flexion_extension_ankle_L_per_trial = swimming_data['flexion_extension_ankle_L_per_trial']
    flexion_extension_ankle_R_per_trial = swimming_data['flexion_extension_ankle_R_per_trial']
    
    # Set minimum time between strokes (1 frame)
    min_distance = 1
    
    # Set cutoff for Butterworth filter
    cut_off = 0.01
    
    # Initialize all necessary lists
    RoM_ankle_L_per_trial = []
    RoM_ankle_R_per_trial = []
    RoM_knee_L_per_trial = []
    RoM_knee_R_per_trial = []
    RoM_hip_L_per_trial = []
    RoM_hip_R_per_trial = []
    RoM_hip_L_adduction_abduction_per_trial = []
    RoM_hip_R_adduction_abduction_per_trial = []

    Min_ankle_L_per_trial = []
    Max_ankle_L_per_trial = []
    Min_ankle_R_per_trial = []
    Max_ankle_R_per_trial = []
    Min_knee_L_per_trial = []
    Max_knee_L_per_trial = []
    Min_knee_R_per_trial = []
    Max_knee_R_per_trial = []
    Min_hip_L_per_trial = []
    Max_hip_L_per_trial = []
    Min_hip_R_per_trial = []
    Max_hip_R_per_trial = []
    Min_coronal_hip_L_per_trial = []
    Max_coronal_hip_L_per_trial = []
    Min_coronal_hip_R_per_trial = []
    Max_coronal_hip_R_per_trial = []

    # Detect strokes and compute parameters
    for trial in range(trial_nr):

        # Sides to consider
        sides = ['Left', 'Right']

        # Felxion/extension
        angle_hip_per_trial = [flexion_extension_hip_L_per_trial, flexion_extension_hip_R_per_trial]
        angle_hip_add_abd_per_trial = [adduction_abduction_hip_L_per_trial, adduction_abduction_hip_R_per_trial]
        angle_knee_per_trial = [flexion_extension_knee_L_per_trial, flexion_extension_knee_R_per_trial]
        angle_ankle_per_trial = [flexion_extension_ankle_L_per_trial, flexion_extension_ankle_R_per_trial]

        for side in range(len(sides)):
            peaks, peak_values = find_stroke_peaks(angle_knee_per_trial[side][trial], 
                                                   cut_off, 
                                                   min_distance, 
                                                   plot=False)
            strokes = detect_swimming_strokes(angle_knee_per_trial[side][trial], 
                                              peaks, 
                                              peak_values, 
                                              plot=False)

        # Initialize variables
        angles_filtered_ankle_L = []
        angles_filtered_ankle_R = []
        angles_filtered_knee_L = []
        angles_filtered_knee_R = []
        angles_filtered_hip_L = []
        angles_filtered_hip_R = []
        angles_filtered_hip_L_coronal = []
        angles_filtered_hip_R_coronal = []

        i = 0
        # Filter angles to only include valid strokes
        for stroke in strokes:
            # Ankle
            angles_filtered_ankle_L.extend(flexion_extension_ankle_L_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_ankle_R.extend(flexion_extension_ankle_R_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            # Knee
            angles_filtered_knee_L.extend(flexion_extension_knee_L_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_knee_R.extend(flexion_extension_knee_R_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            # Hip
            angles_filtered_hip_L.extend(flexion_extension_hip_L_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_hip_R.extend(flexion_extension_hip_R_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_hip_L_coronal.extend(adduction_abduction_hip_L_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            angles_filtered_hip_R_coronal.extend(adduction_abduction_hip_R_per_trial[trial][strokes[i][0]:strokes[i][-1]])
            i=+1

        '--------------------------- Hip ---------------------------'
        max_angle_hip_L = np.percentile(angles_filtered_hip_L, 95)
        min_angle_hip_L = np.percentile(angles_filtered_hip_L, 5)
        max_angle_hip_R = np.percentile(angles_filtered_hip_R, 95)
        min_angle_hip_R = np.percentile(angles_filtered_hip_R, 5)

        RoM_hip_L = max_angle_hip_L - min_angle_hip_L
        RoM_hip_R = max_angle_hip_R - min_angle_hip_R

        max_angle_coronal_hip_L = np.percentile(angles_filtered_hip_L_coronal, 95)
        min_angle_coronal_hip_L = np.percentile(angles_filtered_hip_L_coronal, 5)
        max_angle_coronal_hip_R = np.percentile(angles_filtered_hip_R_coronal, 95)
        min_angle_coronal_hip_R = np.percentile(angles_filtered_hip_R_coronal, 5)

        RoM_hip_L_adduction_abduction = max_angle_coronal_hip_L - min_angle_coronal_hip_L
        RoM_hip_R_adduction_abduction = max_angle_coronal_hip_R - min_angle_coronal_hip_R

        '--------------------------- Knee ---------------------------'

        max_angle_knee_L = np.percentile(angles_filtered_knee_L, 95)
        min_angle_knee_L = np.percentile(angles_filtered_knee_L, 5)
        max_angle_knee_R = np.percentile(angles_filtered_knee_R, 95)
        min_angle_knee_R = np.percentile(angles_filtered_knee_R, 5)

        RoM_knee_L = max_angle_knee_L - min_angle_knee_L
        RoM_knee_R = max_angle_knee_R - min_angle_knee_R
        
        '--------------------------- Ankle ---------------------------'

        max_angle_ankle_L = np.percentile(angles_filtered_ankle_L, 95)
        min_angle_ankle_L = np.percentile(angles_filtered_ankle_L, 5)
        max_angle_ankle_R = np.percentile(angles_filtered_ankle_R, 95)
        min_angle_ankle_R = np.percentile(angles_filtered_ankle_R, 5)

        RoM_ankle_L = max_angle_ankle_L - min_angle_ankle_L
        RoM_ankle_R = max_angle_ankle_R - min_angle_ankle_R

        RoM_ankle_L_per_trial.append(RoM_ankle_L)
        RoM_ankle_R_per_trial.append(RoM_ankle_R)
        RoM_knee_L_per_trial.append(RoM_knee_L)
        RoM_knee_R_per_trial.append(RoM_knee_R)
        RoM_hip_L_per_trial.append(RoM_hip_L)
        RoM_hip_R_per_trial.append(RoM_hip_R)
        RoM_hip_L_adduction_abduction_per_trial.append(RoM_hip_L_adduction_abduction)
        RoM_hip_R_adduction_abduction_per_trial.append(RoM_hip_R_adduction_abduction)

        Min_ankle_L_per_trial.append(min_angle_ankle_L)
        Max_ankle_L_per_trial.append(max_angle_ankle_L)
        Min_ankle_R_per_trial.append(min_angle_ankle_R)
        Max_ankle_R_per_trial.append(max_angle_ankle_R)
        Min_knee_L_per_trial.append(min_angle_knee_L)
        Max_knee_L_per_trial.append(max_angle_knee_L)
        Min_knee_R_per_trial.append(min_angle_knee_R)
        Max_knee_R_per_trial.append(max_angle_knee_R)
        Min_hip_L_per_trial.append(min_angle_hip_L)
        Max_hip_L_per_trial.append(max_angle_hip_L)
        Min_hip_R_per_trial.append(min_angle_hip_R)
        Max_hip_R_per_trial.append(max_angle_hip_R)
        Min_coronal_hip_L_per_trial.append(min_angle_coronal_hip_L)
        Max_coronal_hip_L_per_trial.append(max_angle_coronal_hip_L)
        Min_coronal_hip_R_per_trial.append(min_angle_coronal_hip_R)
        Max_coronal_hip_R_per_trial.append(max_angle_coronal_hip_R)



        ########################################### EXTRACT DATA PER STROKE ###########################################


        ankles = [ankle_nr_L, ankle_nr_R]

        # Define variables per side
        strokes_filtered_per_side = []

        patient_hip_knee_per_side = []
        patient_ankle_knee_per_side = []
        patient_ankle_hip_per_side = []

        mean_ankle_per_side = []
        mean_knee_per_side = []
        mean_hip_per_side = []
        mean_hip_add_abd_per_side = []

        # Filter angles to remove noise
        b, a = butter(3, cut_off, 'lowpass')
        angle_knee_per_trial[0][trial] = filtfilt(b, a, angle_knee_per_trial[0][trial])
        angle_knee_per_trial[1][trial] = filtfilt(b, a, angle_knee_per_trial[1][trial])
        angle_hip_per_trial[0][trial] = filtfilt(b, a, angle_hip_per_trial[0][trial])
        angle_hip_per_trial[1][trial] = filtfilt(b, a, angle_hip_per_trial[1][trial])

        for side in range(len(sides)):
            peaks, peak_values = find_stroke_peaks(angle_knee_per_trial[side][trial], 
                                                   cut_off, 
                                                   min_distance, 
                                                   plot=False)
            strokes = detect_swimming_strokes(angle_knee_per_trial[side][trial], 
                                              peaks, 
                                              peak_values, 
                                              plot=False)    

            # Define trail variables per stroke
            velocity_per_stroke = []
            displacement_per_stroke = []
            angle_ankle_per_stroke = []
            angle_knee_per_stroke = []
            angle_hip_per_stroke = []
            angle_hip_add_abd_per_stroke = []

            # Extract gyroscope and accelerometer data per stroke
            for stroke in strokes:

                # Start and stop times of stroke within the trial data
                [stroke_start, stroke_stop] = stroke
                acc_per_stroke = acc_per_trial[trial][ankles[side]][stroke_start:stroke_stop]

                # Start and stop times of stroke within the whole data
                stroke_start_total = stroke_start + int(x_starts[trial] * 60 * 60 * freq)
                stroke_stop_total = stroke_stop + int(x_starts[trial] * 60 * 60 * freq)

                # Frames during which the stroke event occurs
                time_array = np.array(time_list[stroke_start_total:stroke_stop_total])

                # Integrate twice to get displacement
                velocity = integrate_acceleration(acc_per_stroke, time_array)
                displacement = integrate_velocity(velocity, time_array)

                velocity_per_stroke.append(velocity)
                displacement_per_stroke.append(displacement)

                angle_ankle_per_stroke.append(angle_ankle_per_trial[side][trial][stroke_start:stroke_stop])
                angle_knee_per_stroke.append(angle_knee_per_trial[side][trial][stroke_start:stroke_stop])
                angle_hip_per_stroke.append(angle_hip_per_trial[side][trial][stroke_start:stroke_stop])
                angle_hip_add_abd_per_stroke.append(angle_hip_add_abd_per_trial[side][trial][stroke_start:stroke_stop])

            # Remove outliers
            strokes_filtered, _ = remove_outliers_cyclogram(strokes, displacement_per_stroke, side)
            strokes_filtered_per_side.append(strokes_filtered)

            # Remove stroke outliers from angles
            strokes_filtered_indices = [index for index, stroke in enumerate(strokes) if stroke in strokes_filtered]
            angle_ankle_per_stroke_filtered = [angle_ankle_per_stroke[index] for index in strokes_filtered_indices]
            angle_knee_per_stroke_filtered = [angle_knee_per_stroke[index] for index in strokes_filtered_indices]
            angle_hip_per_stroke_filtered = [angle_hip_per_stroke[index] for index in strokes_filtered_indices]
            angle_hip_add_abd_per_stroke_filtered = [angle_hip_add_abd_per_stroke[index] for index in strokes_filtered_indices]
            
            # Hip-knee
            mean_hip_centered, mean_knee_centered, mean_hip, mean_knee = center_angles(angle_hip_per_stroke_filtered, 
                                                                                       angle_knee_per_stroke_filtered)
                                
            patient_hip_knee_centered = [mean_hip_centered, mean_knee_centered]
            
            # Ankle-knee
            mean_ankle_centered, mean_knee_centered, mean_ankle, _ = center_angles(angle_ankle_per_stroke_filtered,
                                                                                   angle_knee_per_stroke_filtered)  
                                                
            patient_ankle_knee_centered = [mean_ankle_centered, mean_knee_centered]

            # Ankle-hip
            mean_ankle_centered, mean_hip_centered, _, _ = center_angles(angle_ankle_per_stroke_filtered,
                                                                         angle_hip_per_stroke_filtered)  
            
            patient_ankle_hip_centered = [mean_ankle_centered, mean_hip_centered]

            # Get mean hip adduction / abduction
            _, _, mean_hip_add_abd, _ = center_angles(angle_hip_add_abd_per_stroke_filtered,
                                                      angle_hip_per_stroke_filtered)

            patient_hip_knee_per_side.append(patient_hip_knee_centered)
            patient_ankle_knee_per_side.append(patient_ankle_knee_centered)
            patient_ankle_hip_per_side.append(patient_ankle_hip_centered)                             
                
            mean_ankle_per_side.append(mean_ankle)
            mean_knee_per_side.append(mean_knee)
            mean_hip_per_side.append(mean_hip)
            mean_hip_add_abd_per_side.append(mean_hip_add_abd)

            # Save angles to csv file (for SSD and angle plots)
            first_row = ['Participant_' +  file, 'Trial' + str(trial+1), sides[side], 'Knee'] + mean_knee
            second_row = ['Participant_' +  file, 'Trial' + str(trial+1), sides[side], 'Hip'] + mean_hip
            third_row = ['Participant_' +  file, 'Trial' + str(trial+1), sides[side], 'Ankle'] + mean_ankle
            forth_row = ['Participant_' +  file, 'Trial' + str(trial+1), sides[side], 'Hip_add_abd'] + mean_hip_add_abd

            csv_file_path = os.path.join(data_dir, f'Healthy_Reference_Data/Joint_Angles_Healthy_Controls.csv')
            # Read existing data if the file exists
            existing_data = []
            if os.path.exists(csv_file_path):
                with open(csv_file_path, 'r', newline='') as data_file:
                    reader = csv.reader(data_file)
                    existing_data = list(reader)

            # Extract existing participant and trial entries
            existing_participant_trials = [
                (row[0], row[1], row[2]) for row in existing_data if len(row) >= 2
            ]

            # Define the new participant and trial key
            new_participant = 'Participant_' + file
            new_trial = 'Trial' + str(trial+1)
            new_side = sides[side]

            # Check if the participant and trial are already in the file
            if any([
                existing_participant_trials[entry][0] == new_participant and
                existing_participant_trials[entry][1] == new_trial and
                existing_participant_trials[entry][2] == new_side
                for entry in range(len(existing_participant_trials))
            ]):
                pass
            else:  
                with open(csv_file_path, 'a', newline='') as data_file:
                    writer = csv.writer(data_file)
                    writer.writerow(first_row)
                    writer.writerow(second_row)
                    writer.writerow(third_row)
                    writer.writerow(forth_row)
