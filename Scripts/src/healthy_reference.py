"""
Script: healthy_reference.py

Description:
    This script processes healthy reference data for joint angles during swimming.
    It computes statistical summaries and mean shapes for ankle, knee, and hip joints.
    The script performs the following steps:

    1. Load healthy reference data.
    2. Compute joint angle averages (with shapes centered around the origin, 
       see calculate_mean_shape_centered).
       - Separates data by joint (hip, knee, ankle).
    3. Outputs reference data as a baseline for comparing patient data.

Dependencies:
    - Joint angles should be stored in 'Data/Healthy_Reference_Data/Joint_Angles_Healthy_Controls.csv'.

Usage:
    - This script is not meant to be run directly - the computed reference data 
      are imported for comparison with patient swimming trials.
    
Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from src.functions import calculate_mean_shape_centered
from config import data_path

# Load healthy reference data
csv_file = 'Healthy_Reference_Data/Joint_Angles_Healthy_Controls.csv'
csv_file_path = os.path.join(data_path, csv_file)
healthy_referenece = pd.read_csv(csv_file_path)

# List of lists with angles of each trial and paricipant
all_ankle_angles = []
all_knee_angles = []
all_hip_angles = []

for participant in healthy_referenece['Participant'].unique():
    participant_data = healthy_referenece[healthy_referenece['Participant'] == participant]

    for trial in participant_data['Trial'].unique():
        trial_data = participant_data[participant_data['Trial'] == trial]
        
        for side in ['Left', 'Right']:
            side_data = trial_data[trial_data['Side'] == side]
            ankle_angles = side_data[side_data['Joint'] == 'Ankle'].iloc[0, -100:].to_frame().T.iloc[0, -100:].values.tolist()
            knee_angles = side_data[side_data['Joint'] == 'Knee'].iloc[0, -100:].to_frame().T.iloc[0, -100:].values.tolist()
            hip_angles = side_data[side_data['Joint'] == 'Hip'].iloc[0, -100:].to_frame().T.iloc[0, -100:].values.tolist()
            
            # Also averaged over left and right
            all_ankle_angles.append(ankle_angles)
            all_knee_angles.append(knee_angles)
            all_hip_angles.append(hip_angles)
            
# Ankle-knee
mean_ankle_angle, mean_knee_angle = calculate_mean_shape_centered(all_ankle_angles, all_knee_angles)
reference_ankle_knee = [mean_ankle_angle, mean_knee_angle]

# Hip-knee
mean_hip_angle, mean_knee_angle = calculate_mean_shape_centered(all_hip_angles, all_knee_angles)
reference_hip_knee = [mean_hip_angle, mean_knee_angle]

# Ankle-hip
mean_ankle_angle, mean_hip_angle = calculate_mean_shape_centered(all_ankle_angles, all_hip_angles)
reference_ankle_hip = [mean_ankle_angle, mean_hip_angle]

reference_data = [reference_hip_knee, reference_ankle_knee]
