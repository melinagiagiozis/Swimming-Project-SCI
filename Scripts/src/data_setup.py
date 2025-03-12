"""
Script: data_setup.py

Description:
    This script provides functions for setting up and processing swimming sensor data.
    It includes functions for:
    
    1. Creating parameter files to store joint angles and swimming parameters.
    2. Loading sensor data (accelerometer, gyroscope, sensor positions, and timestamps) 
       from `.mat` files.
    3. Extracting trial times from `.csv` files to obtain start and stop times.

Dependencies:
    - `.mat` files with sensor data should be stored in `Data/Participant_Data/`.
    - CSV templates for swimming parameters and joint angles should be in `Data/Templates/`.

Usage:
    - This script is not meant to be run directly - the functions are imported 
      into the main processing pipeline.
    
Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import mat73
import numpy as np
import pandas as pd


def create_parameter_files(data_dir, results_dir, participant_folder, parameter_type):
    """
    Creates CSV files in Results folder based on the selected template 
    ('Swimming_Parameters_Template.csv' or 'Joint_Angles_Template.csv').

    Args:
        data_dir (str): The absolute path to the data directory.
        results_dir (str): The absolute path to the results directory.
        participant_folder (str): The current participant folder, e.g. SCI_Baseline.
        parameter_type (str): The type of parameters to generate files for. 
                              Options: "swimming" or "angles".
    
    Returns:
        None
    """
    # Define file naming based on parameter type
    templates = {
        "swimming": "Swimming_Parameters_Template.csv",
        "angles": "Joint_Angles_Template.csv"
    }
    
    if parameter_type not in templates:
        raise ValueError("Invalid parameter_type. Choose either 'swimming' or 'angles'.")

    # Define absolute paths
    template_path = os.path.join(data_dir, "Templates", templates[parameter_type])

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Load template
    template_df = pd.read_csv(template_path)

    # Define new file name based on parameter type
    if parameter_type == "swimming":
        new_file_name = f"Swimming_Parameters_{participant_folder}.csv"
    else:
        new_file_name = f"Joint_Angles_{participant_folder}.csv"

    # Define full path for saving
    new_file_path = os.path.join(results_dir, new_file_name)

    # Save a copy of the template
    if not os.path.exists(new_file_path):  # Check if the file already exists
        template_df.to_csv(new_file_path, index=False)

        print(f"Created: {new_file_path}")


def load_sensor_data(file_path_data):
    """
    Load and process sensor data from a .mat file in the specified directory.
    Returns extracted accelerometer, gyroscope data, sensor positions, and time data.
    """
    # Find the only .mat file
    mat_files = [f for f in os.listdir(file_path_data) if f.endswith('.mat') and not f.startswith('.')]
    if not mat_files:
        print(f"No .mat file found in {file_path_data}")
        return None  # No data to return

    mat_file = mat_files[0]  # Select the first (and only) .mat file
    complete_file_path = os.path.join(file_path_data, mat_file)
    
    print(f'Loading file: {mat_file}')
    mat_data = mat73.loadmat(complete_file_path)
    print(f'Successfully loaded {mat_file}')

    acc_data, gyro_data, sensor_positions, time_list = [], [], [], []

    for jumpExp in mat_data.values():
        freq = int(jumpExp['sensors'][0]['header']['ADXL_freq'][:3])
        start_time = jumpExp['header']['startStr']
        stop_time = jumpExp['header']['stopStr']
        
        for sensor in jumpExp['sensors']:
            acc_data.append(sensor['acc'])
            gyro_data.append(sensor['gyro'])
            sensor_positions.append(sensor['header']['position'])
        
        for time_steps in jumpExp['time']:
            time_list.append(time_steps)

    # Fix spelling mistake for patient ST01
    sensor_positions = ['Thigh_Left' if pos == 'Tigh_Left' else pos for pos in sensor_positions]

    # Get sensor number of different positions
    position_name = ['Foot_Left', 'Foot_Right', 'Ankle_Left', 'Ankle_Right',
                     'Thigh_Left', 'Thigh_Right', 'Back_Upper', 'Back_Lower', 'Reference']
    
    position_nr = len(sensor_positions)
    position_no = np.zeros(position_nr)

    for j in range(position_nr):
        for k in range(len(position_name)):
            if sensor_positions[j] == position_name[k]:
                position_no[k] = j
        
    sensor_data_dict = {
            'filename': mat_file,
            'acc_data': acc_data,
            'gyro_data': gyro_data,
            'sensor_positions': sensor_positions,
            'time_list': time_list,
            'freq': freq,
            'start_time': start_time,
            'stop_time': stop_time,
            'positions': {
                'foot_nr_L': int(position_no[0]),
                'foot_nr_R': int(position_no[1]),    
                'ankle_nr_L': int(position_no[2]),
                'ankle_nr_R': int(position_no[3]),
                # Swap thigh sensors for SS21
                'thigh_nr_L': int(position_no[4]) if not str(mat_file[-8:-4]) == 'SS21' else int(position_no[5]),
                'thigh_nr_R': int(position_no[5]) if not str(mat_file[-8:-4]) == 'SS21' else int(position_no[4]),    
                'back_nr_U': int(position_no[6]),
                'back_nr_L': int(position_no[7]),
                'ref_nr': int(position_no[8])
            }
        }
    
    return sensor_data_dict


def extract_trial_times(file_path_data):
    """
    Extracts specific trial times from a participant's CSV file.

    Args:
        file_path_data (str): The directory containing the CSV file.

    Returns:
        dict: A dictionary containing start and stop times for predefined trials and remaining trial times.
    """
    # Find the only .csv file
    csv_files = [f for f in os.listdir(file_path_data) if f.endswith('.csv') and not f.startswith('.')]
    if not csv_files:
        print(f"No CSV file found in {file_path_data}")
        return None  # No data to return

    csv_file = csv_files[0]  # Select the first (and only) CSV file
    complete_file_path = os.path.join(file_path_data, csv_file)
    
    # Load data
    times_table = pd.read_csv(complete_file_path)

    # Initialize dictionary to store times
    trial_times_dict = {}

    # Define trial names
    trial_names = {
        'Calibration 15s:': 'calib',
        '10x Hip Knee Flexion:': 'flex',
        '60s Static Swimming:': 'static',
        'Calibration in Water 15s:': 'calib_water'
    }

    # Iterate through the rows to extract relevant times
    for i in range(len(times_table)):
        trial_name = times_table.iloc[i, 0]
        if trial_name in trial_names:
            trial_key = trial_names[trial_name]
            trial_times_dict[f"x_start_{trial_key}"] = times_table.iloc[i].loc['Start']
            trial_times_dict[f"x_stop_{trial_key}"] = times_table.iloc[i].loc['Stop']
        else:
            # Store the remaining trial times
            trial_times_dict["trial_times"] = times_table.iloc[i:].reset_index(drop=True)
            break

    return trial_times_dict


def create_reference_file(data_directory):
    """
    Creates CSV files in Data/Healthy_Reference_Data folder.

    Args:
        data_directory (str): The absolute path to the data directory.
    
    Returns:
        None
    """

    # Define absolute paths
    template_path = os.path.join(data_directory, "Templates", "Joint_Angles_Template.csv")
    reference_dir = os.path.join(data_directory, "Healthy_Reference_Data")

    # Ensure results directory exists
    os.makedirs(reference_dir, exist_ok=True)

    # Load template
    template_df = pd.read_csv(template_path)

    # Define new file name based on parameter type
    new_file_name = f"Joint_Angles_Healthy_Controls.csv"

    # Define full path for saving
    new_file_path = os.path.join(reference_dir, new_file_name)

    # Save a copy of the template
    if not os.path.exists(new_file_path):  # Check if the file already exists
        template_df.to_csv(new_file_path, index=False)

        print(f"Created: {new_file_path}")
