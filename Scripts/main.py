"""
Script: main.py

Description:
    This script processes swimming data from multiple participants. 
    It performs the following steps:
    
    1. Iterates through each participant's folder containing sensor data.
    2. Creates CSV files to store extracted swimming parameters and joint angles.
    3. Loads sensor data (.mat files) and trial times (.csv files).
    4. Runs a swimming algorithm to detect strokes.
    5. Extracts relevant swimming parameters and saves them to CSV files.
    6. Measures and prints the total execution time of the processing.

Dependencies:
    - Requires 'data_setup', 'swimming_algorithm', 'extract_parameters' modules.
    - Sensor data should be stored in 'Data/Participant_Data'.
    - Output CSV files are saved in predefined results directory.

Usage:
    Run the script in a Python environment where all dependencies 
    are installed (see requirements.txt). Ensure the 'Data' directory 
    contains participant sensor data and trial times.

Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import time
from src.data_setup import *
from src.swimming_algorithm import *
from src.extract_parameters import *
from config import data_path, results_path
from src.healthy_reference import reference_data

def main(data_path, results_path):
    participant_data_dir = os.path.join(data_path, "Participant_Data")

    # Start the timer
    start_time = time.time()

    for participant_folder in os.listdir(participant_data_dir):
        if not participant_folder.startswith('.') and os.path.isdir(os.path.join(participant_data_dir, participant_folder)):
            
            # Create CSV files to save swimming parameters
            create_parameter_files(data_path, results_path, participant_folder, "swimming")

            # Create CSV files to save joint angles
            create_parameter_files(data_path, results_path, participant_folder, "angles")

            # Define the path to the participant's sensor data directory
            participant_dir = os.path.join(participant_data_dir, participant_folder)

            for participant in sorted(os.listdir(participant_dir)):
                if not participant.startswith('.') and os.path.isdir(os.path.join(participant_dir, participant)):

                    print('----------------------------')
                    
                    # -------- .mat files --------

                    # Define the path to the sensor data
                    participant_dir_path = os.path.join(participant_dir, participant)

                    # Load sensor data for the participant
                    sensor_data_dict = load_sensor_data(participant_dir_path)

                    # -------- .csv files --------

                    # Load trial times for the participant
                    trial_times_dict = extract_trial_times(participant_dir_path)
                    
                    # Run swimming algorithm to extract strokes
                    print('... Running swimming algorithm')
                    swimming_data = swimming_algorithm(sensor_data_dict, trial_times_dict)

                    # Extract swimming parameters and save them to csv files
                    print('... Extracting swimming parameters')
                    extract_swimming_parameters(swimming_data, results_path, participant, 
                                                participant_folder, reference_data, create_files=True)

            # Stop the timer
            end_time = time.time()

            # Print execution time
            execution_time = end_time - start_time
            print(f"\nTotal execution time: {execution_time/60:.2f} minutes")

if __name__ == "__main__":
    main(data_path, results_path)
