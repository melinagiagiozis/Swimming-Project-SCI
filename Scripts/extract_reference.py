"""
Script: extract_reference.py

Description:
    This script extracts joint angles during swimming from healthy participants
    for further analysis. It performs the following steps:
    
    1. Iterates through each participant's folder containing sensor data.
    2. Creates CSV files to store extracted joint angles.
    3. Loads sensor data (.mat files) and trial times (.csv files).
    4. Runs a swimming algorithm to detect strokes.
    5. Extracts joint angles and saves them to CSV files.
    6. Measures and prints the total execution time of the processing.

Dependencies:
    - Requires 'data_setup', 'swimming_algorithm', 'extract_parameters' modules.
    - Sensor data should be stored in 'Data/Participant_Data/Healthy_Contorls'.
    - Output CSV files are saved in predefined results directory.

Usage:
    Run the script in a Python environment where all dependencies are 
    installed (see requirements.txt). Ensure the 'Data/Participant_Data/Healthy_Contorls' 
    directory contains participant sensor data and trial times.

Output:
    - Extracted joint angles are saved to CSV files at `Results/Joint_Angles_Healthy_Controls.csv`.

Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import time
from config import *
from src.data_setup import *
from src.swimming_algorithm import *
from src.extract_parameters import *

# Get the absolute path of the current working directory
data_dir = os.path.join(data_path, "Participant_Data")

# Start the timer
start_time = time.time()

folder = "Healthy_Controls"

# Create Healthy_Reference_Data folder if it does not yet exist
reference_dir = os.path.join(data_path, "Healthy_Reference_Data")
os.makedirs(os.path.dirname(reference_dir), exist_ok=True)

if os.path.isdir(os.path.join(data_dir, folder)):
    
    # Create CSV files to save swimming parameters
    create_reference_file(data_path)

    # Define the path to the participant's sensor data directory
    participant_dir = os.path.join(data_dir, folder)

    for participant in sorted(os.listdir(participant_dir)):
        if not participant.startswith('.') and os.path.isdir(os.path.join(participant_dir, participant)):

            print('----------------------------')
            
            # -------- .mat files --------

            # Define the path to the sensor data
            participant_data_dir = os.path.join(participant_dir, participant)

            # Load sensor data for the participant
            sensor_data_dict = load_sensor_data(participant_data_dir)

            # -------- .csv files --------

            # Load trial times for the participant
            trial_times_dict = extract_trial_times(participant_data_dir)
            
            # Run swimming algorithm to extract strokes
            print('... Running swimming algorithm')
            swimming_data = swimming_algorithm(sensor_data_dict, trial_times_dict)

            # Extract swimming parameters and save them to csv files
            print('... Extracting joint angles')
            extract_joint_angles_only(swimming_data, data_path, participant)

    # Stop the timer
    end_time = time.time()

    # Print execution time
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time/60:.2f} minutes")