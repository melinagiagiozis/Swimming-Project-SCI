"""
Script: swimming_algorithm.py

Description:
    This script defines the `swimming_algorithm` function, which processes IMU data
    to compute joint angles and extract relevant motion parameters for swimming analysis. 
    The algorithm performs the following steps:

    1. Extract sensor data (accelerometer and gyroscope data) and trial times.
    2. Establishes a common reference frame using estimated sensor orientations.
    3. Compute joint angles (flexion/extension and adduction/abduction).
    4. Returns a structured dictionary containing processed joint angles, reference data, 
       and other computed swimming parameters.

Dependencies:
    - Requires 'healthy_reference' and 'functions' modules.

Usage:
    - This script is not meant to be run directly - the function is imported 
      into the main processing pipeline.

Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import numpy as np
import ahrs
from scipy.spatial.transform import Rotation as R
from src.functions import *


def swimming_algorithm(sensor_data_dict, trial_times_dict):
    """
    Processes sensor data and trial times, extracts calibration data, computes joint angles, 
    and applies reference frame transformations.

    Args:
        sensor_data_dict (dict): Dictionary containing sensor data from load_sensor_data().
        trial_times_dict (dict): Dictionary containing trial times from extract_trial_times().
        
    Returns:
        dict: Processed sensor and trial data, including extracted joint angles and range of motion.
    """

    # Extract data from sensor_data_dict
    acc_data = sensor_data_dict['acc_data']
    gyro_data = sensor_data_dict['gyro_data']
    sensor_positions = sensor_data_dict['sensor_positions']
    freq = sensor_data_dict['freq']

    # Extract position numbers
    foot_nr_L = sensor_data_dict['positions']['foot_nr_L']
    foot_nr_R = sensor_data_dict['positions']['foot_nr_R']
    ankle_nr_L = sensor_data_dict['positions']['ankle_nr_L']
    ankle_nr_R = sensor_data_dict['positions']['ankle_nr_R']
    thigh_nr_L = sensor_data_dict['positions']['thigh_nr_L']
    thigh_nr_R = sensor_data_dict['positions']['thigh_nr_R']
    back_nr_L = sensor_data_dict['positions']['back_nr_L']
    back_nr_U = sensor_data_dict['positions']['back_nr_U']
    ref_nr = sensor_data_dict['positions']['ref_nr']

    # Define general variables
    position_nr = len(sensor_positions)
    trial_nr = len(trial_times_dict['trial_times'])
    x_starts = []
    x_stops = []

    # Define trial variables
    acc_per_trial = []
    gyro_per_trial = []
    quat_per_trial = []
    rotation_matrix_per_trial = []

    # Define joint angle lists
    angle_ankle_L_per_trial = []
    angle_ankle_R_per_trial = []
    angle_knee_L_per_trial = []
    angle_knee_R_per_trial = []
    angle_hip_L_per_trial = []
    angle_hip_R_per_trial = []
    adduction_abduction_hip_L_per_trial = []
    adduction_abduction_hip_R_per_trial = []

    # Extract sensor data during calibration
    acc_per_segment_calib = []
    gyro_per_segment_calib = []

    try:
        calibration_timepoints = [trial_times_dict["x_start_calib_water"], 
                                  trial_times_dict["x_stop_calib_water"]]
    except KeyError:
        try:
            # If no calibration in water use dry calibration
            calibration_timepoints = [trial_times_dict["x_start_calib"], 
                                      trial_times_dict["x_stop_calib"]]
        except KeyError:
            # If there is no calibration use first 2 seconds of hip-knee flexions
            calibration_timepoints = [trial_times_dict['x_start_flex'], 
                                      trial_times_dict['x_start_flex'] + 2/60/60]
            
    # Start and end of calibration
    start = int(calibration_timepoints[0].astype(float)*60*60*freq)
    stop = int(calibration_timepoints[1].astype(float)*60*60*freq)

    # Standing at during calibration?
    threshold_mean = -0.8   # Gravity: -1g
    threshold_std = 0.1

    # First 400 timesteps are 2 seconds
    mean = np.mean(acc_data[thigh_nr_L][start:stop][:,0])
    std = np.std(acc_data[thigh_nr_L][start:stop][:,0])

    if mean < threshold_mean and std < threshold_std:
        pass
    else:
        print('WARNING: No standing during calibration!')

    for position in range(position_nr):
        # Could also use calibration on land?
        acc_per_segment_calib.append(acc_data[position]
                                    [int(calibration_timepoints[0].astype(float)*60*60*freq) : 
                                    int(calibration_timepoints[1].astype(float)*60*60*freq), :] 
                                    * 9.81)  # [g] to [m/s^2]
        gyro_per_segment_calib.append(gyro_data[position]
                                    [int(calibration_timepoints[0].astype(float)*60*60*freq) : 
                                    int(calibration_timepoints[1].astype(float)*60*60*freq), :] 
                                    * np.pi/180)  # [deg/s] to [rad/s]
        timesteps_in_calib = len(acc_per_segment_calib[position])

        # Calculate quaternions with Madgwick
        madgwick = ahrs.filters.Madgwick(gyro_per_segment_calib[position], 
                                        acc_per_segment_calib[position], 
                                        frequency=freq)
        
        # Find the average quaternion in the calbration time
        quaternions_in_calib = []
        for timestep in range(timesteps_in_calib):
            quaternions_in_calib.append(madgwick.Q[timestep])
        mean_quaternions_in_calib = np.mean(quaternions_in_calib, axis=0)
        rotation_matrix_in_calib = rotation_matrix_from_quaternion(mean_quaternions_in_calib)

        # Calculate pitch angle (y-axis) of foot sensors
        if position == foot_nr_R:
            yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_in_calib)
            pitch_foot_R = pitch
            roll_foot_R = 180 + roll
        
        if position == foot_nr_L:
            yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_in_calib)
            pitch_foot_L = pitch
            roll_foot_L = 180 - roll

    # Common reference frame (right thigh/ankle)
    rotation_matrices_x = [0 for _ in range(position_nr)]
    rotation_matrices_y = [0 for _ in range(position_nr)]
    rotation_matrices_z = [0 for _ in range(position_nr)]

    # foot L
    rotation_matrices_x[foot_nr_L] = R.from_euler('x', -roll_foot_L, degrees=True)
    rotation_matrices_y[foot_nr_L] = R.from_euler('y', -90-pitch_foot_L, degrees=True)
    rotation_matrices_z[foot_nr_L] = R.from_euler('z', 90, degrees=True)

    # foot R
    rotation_matrices_x[foot_nr_R] = R.from_euler('x', 180+roll_foot_R, degrees=True)
    rotation_matrices_y[foot_nr_R] = R.from_euler('y', 90+pitch_foot_R, degrees=True)
    rotation_matrices_z[foot_nr_R] = R.from_euler('z', 90, degrees=True)

    # ankle L
    rotation_matrices_x[ankle_nr_L] = R.from_euler('x', 180, degrees=True)
    rotation_matrices_y[ankle_nr_L] = R.from_euler('y', 0, degrees=True)
    rotation_matrices_z[ankle_nr_L] = R.from_euler('z', 90, degrees=True)

    # ankle R
    rotation_matrices_x[ankle_nr_R] = R.from_euler('x', 0, degrees=True)
    rotation_matrices_y[ankle_nr_R] = R.from_euler('y', 0, degrees=True)
    rotation_matrices_z[ankle_nr_R] = R.from_euler('z', 90, degrees=True)

    # thigh L
    rotation_matrices_x[thigh_nr_L] = R.from_euler('x', 180, degrees=True)
    rotation_matrices_y[thigh_nr_L] = R.from_euler('y', 0, degrees=True)
    rotation_matrices_z[thigh_nr_L] = R.from_euler('z', 90, degrees=True)

    # thigh R
    rotation_matrices_x[thigh_nr_R] = R.from_euler('x', 0, degrees=True)
    rotation_matrices_y[thigh_nr_R] = R.from_euler('y', 0, degrees=True)
    rotation_matrices_z[thigh_nr_R] = R.from_euler('z', 90, degrees=True)

    # back U
    rotation_matrices_x[back_nr_U] = R.from_euler('x', 90, degrees=True)
    rotation_matrices_y[back_nr_U] = R.from_euler('y', 0, degrees=True)
    rotation_matrices_z[back_nr_U] = R.from_euler('z', 90, degrees=True)

    # back L
    rotation_matrices_x[back_nr_L] = R.from_euler('x', 90, degrees=True)
    rotation_matrices_y[back_nr_L] = R.from_euler('y', 0, degrees=True)
    rotation_matrices_z[back_nr_L] = R.from_euler('z', 90, degrees=True)

    # reference
    rotation_matrices_x[ref_nr] = R.from_euler('x', 0, degrees=True)
    rotation_matrices_y[ref_nr] = R.from_euler('y', 0, degrees=True)
    rotation_matrices_z[ref_nr] = R.from_euler('z', 0, degrees=True)

    # Plot the accelerometer data during trial of right ankle
    for position in range(position_nr):
        acc_data[position] = rotation_matrices_x[position].apply(acc_data[position])
        acc_data[position] = rotation_matrices_y[position].apply(acc_data[position])
        acc_data[position] = rotation_matrices_z[position].apply(acc_data[position])

        gyro_data[position] = rotation_matrices_x[position].apply(gyro_data[position])
        gyro_data[position] = rotation_matrices_y[position].apply(gyro_data[position])
        gyro_data[position] = rotation_matrices_z[position].apply(gyro_data[position])

    # Extract data per trial
    for trial in range(trial_nr):
        
        x_starts.append(trial_times_dict['trial_times'].loc[trial].iloc[1].astype(float))
        x_stops.append(trial_times_dict['trial_times'].loc[trial].iloc[2].astype(float))

        acc_per_segment = []
        gyro_per_segment = []
        quat_per_segment = []
        rotation_matrix_per_segment = []
        
        # Extract sensor data within trial
        for position in range(position_nr):
            acc_per_segment.append(acc_data[position]
                                [int(x_starts[trial]*60*60*freq) : int(x_stops[trial]*60*60*freq), :] 
                                * 9.81)  # [g] to [m/s^2]
            gyro_per_segment.append(gyro_data[position]
                                    [int(x_starts[trial]*60*60*freq) : int(x_stops[trial]*60*60*freq), :] 
                                    * np.pi/180)  # [deg/s] to [rad/s]
            timesteps_in_tiral = len(acc_per_segment[position])

            # Calculate quaternions with Madgwick
            madgwick = ahrs.filters.Madgwick(gyro_per_segment[position], acc_per_segment[position], frequency=freq)
            quat_per_segment.append(madgwick.Q)

            # Calculate rotation matrices for all segments
            rotation_matrix_per_timestep = []
            for timestep in range(timesteps_in_tiral):  # For each row in quaternion
                rotation_matrix = rotation_matrix_from_quaternion(quat_per_segment[position][timestep, :])
                rotation_matrix_per_timestep.append(rotation_matrix)
            rotation_matrix_per_segment.append(rotation_matrix_per_timestep)
            
        acc_per_trial.append(acc_per_segment)
        gyro_per_trial.append(gyro_per_segment)
        quat_per_trial.append(quat_per_segment)
        rotation_matrix_per_trial.append(rotation_matrix_per_segment)

    # Calculate joint angles
    angle_ankle_L_per_trial = []
    angle_ankle_R_per_trial = []
    angle_knee_L_per_trial = [] 
    angle_knee_R_per_trial = []
    angle_hip_L_per_trial = []
    angle_hip_R_per_trial = []

    flexion_extension_ankle_L_per_trial = []
    flexion_extension_ankle_R_per_trial = []
    flexion_extension_knee_L_per_trial = []
    flexion_extension_knee_R_per_trial = []
    flexion_extension_hip_L_per_trial = []
    flexion_extension_hip_R_per_trial = []
    adduction_abduction_hip_L_per_trial = []
    adduction_abduction_hip_R_per_trial = []

    for trial in range(trial_nr):

        # Rotation matrices per position of sensor
        rotation_matrix_foot_L = rotation_matrix_per_trial[trial][foot_nr_L]
        rotation_matrix_foot_R = rotation_matrix_per_trial[trial][foot_nr_R]
        rotation_matrix_ankle_L = rotation_matrix_per_trial[trial][ankle_nr_L]
        rotation_matrix_ankle_R = rotation_matrix_per_trial[trial][ankle_nr_R]
        rotation_matrix_thigh_L = rotation_matrix_per_trial[trial][thigh_nr_L]
        rotation_matrix_thigh_R = rotation_matrix_per_trial[trial][thigh_nr_R]
        rotation_matrix_back_U = rotation_matrix_per_trial[trial][back_nr_U]
        rotation_matrix_back_L = rotation_matrix_per_trial[trial][back_nr_L]
        rotation_matrix_ref = rotation_matrix_per_trial[trial][ref_nr]

        ###################################### Common Reference Frame ######################################

        # Calculate joint angles
        angle_ankle_L = []
        angle_ankle_R = []
        angle_knee_L = []
        angle_knee_R = []
        angle_hip_L = []
        angle_hip_R = []

        flexion_extension_ankle_L = []
        flexion_extension_ankle_R = []
        flexion_extension_knee_L = []
        flexion_extension_knee_R = []
        flexion_extension_hip_L = []
        flexion_extension_hip_R = []
        adduction_abduction_hip_L = []
        adduction_abduction_hip_R = []

        timesteps_in_tiral = len(rotation_matrix_foot_L)

        # Compute the absolute joint angles 
        for timestep in range(timesteps_in_tiral):

            '--------------------------- Left ankle joint angle ---------------------------'
            rotation_matrix_ankle_joint_L = rotation_matrix_foot_L[timestep].T @ rotation_matrix_ankle_L[timestep]
            angle = absolute_angle_of_rotation_matrix(rotation_matrix_ankle_joint_L)
            angle_ankle_L.append(angle)

            # Convert the rotation matrix to Euler angles (pitch, roll, yaw)
            yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_ankle_joint_L)

            # Extract the yaw angle (extension/flexion)
            flexion_extension_ankle_L.append(yaw)

            '--------------------------- Right ankle joint angle ---------------------------'
            rotation_matrix_ankle_joint_R = rotation_matrix_foot_R[timestep].T @ rotation_matrix_ankle_R[timestep]
            angle = absolute_angle_of_rotation_matrix(rotation_matrix_ankle_joint_R)
            angle_ankle_R.append(angle)

            # Convert the rotation matrix to Euler angles (pitch, roll, yaw)
            yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_ankle_joint_R)

            # Extract the yaw angle (extension/flexion)
            flexion_extension_ankle_R.append(yaw)

            '--------------------------- Left knee joint angle ---------------------------'
            rotation_matrix_knee_L = rotation_matrix_thigh_L[timestep].T @ rotation_matrix_ankle_L[timestep]
            angle = absolute_angle_of_rotation_matrix(rotation_matrix_knee_L)
            angle_knee_L.append(angle)

            # Convert the rotation matrix to Euler angles (pitch, roll, yaw)
            yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_knee_L)

            # Extract the yaw angle (extension/flexion)
            flexion_extension_knee_L.append(180-yaw)

            '--------------------------- Right knee joint angle ---------------------------'
            rotation_matrix_knee_R = rotation_matrix_thigh_R[timestep].T @ rotation_matrix_ankle_R[timestep]
            angle = absolute_angle_of_rotation_matrix(rotation_matrix_knee_R)
            angle_knee_R.append(angle)

            # Convert the rotation matrix to Euler angles (pitch, roll, yaw)
            yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_knee_R)

            # Extract the yaw angle (extension/flexion)
            flexion_extension_knee_R.append(180-yaw)

            '--------------------------- Left hip joint angle ---------------------------'
            rotation_matrix_hip_L = rotation_matrix_thigh_L[timestep].T @ rotation_matrix_back_L[timestep]
            angle = absolute_angle_of_rotation_matrix(rotation_matrix_hip_L)
            angle_hip_L.append(angle)

            # Convert the rotation matrix to Euler angles (pitch, roll, yaw)
            yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_hip_L)

            # Extract the yaw angle (extension/flexion)
            flexion_extension_hip_L.append(180-yaw)

            # Extract the pitch angle (adduction/abduction
            adduction_abduction_hip_L.append(pitch)

            '--------------------------- Right hip joint angle ---------------------------'
            rotation_matrix_hip_R = rotation_matrix_thigh_R[timestep].T @ rotation_matrix_back_L[timestep]
            angle = absolute_angle_of_rotation_matrix(rotation_matrix_hip_R)
            angle_hip_R.append(angle)

            # Convert the rotation matrix to Euler angles (pitch, roll, yaw)
            yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_hip_R)

            # Extract the yaw angle (extension/flexion)
            flexion_extension_hip_R.append(180-yaw)

            # Extract the pitch angle (adduction/abduction
            adduction_abduction_hip_R.append(pitch)

            
        # Calculate absolute joint angles
        angle_ankle_L_per_trial.append(angle_ankle_L)
        angle_ankle_R_per_trial.append(angle_ankle_R)
        angle_knee_L_per_trial.append(angle_knee_L)
        angle_knee_R_per_trial.append(angle_knee_R)
        angle_hip_L_per_trial.append(angle_hip_L)
        angle_hip_R_per_trial.append(angle_hip_R)

        # Calculate joint angles
        flexion_extension_ankle_L_per_trial.append(flexion_extension_ankle_L)
        flexion_extension_ankle_R_per_trial.append(flexion_extension_ankle_R)
        flexion_extension_knee_L_per_trial.append(flexion_extension_knee_L)
        flexion_extension_knee_R_per_trial.append(flexion_extension_knee_R)
        flexion_extension_hip_L_per_trial.append(flexion_extension_hip_L)
        flexion_extension_hip_R_per_trial.append(flexion_extension_hip_R)
        adduction_abduction_hip_L_per_trial.append(adduction_abduction_hip_L)
        adduction_abduction_hip_R_per_trial.append(adduction_abduction_hip_R)

    # Package results
    results = {
        "freq": freq,
        "trial_nr": trial_nr,
        "acc_per_trial": acc_per_trial,
        "gyro_per_trial": gyro_per_trial,
        "x_starts": x_starts,
        "x_stops": x_stops,
        "ankle_nr_L": ankle_nr_L,
        "ankle_nr_R": ankle_nr_R,
        "time_list": sensor_data_dict["time_list"],
        "quat_per_trial": quat_per_trial,
        
        "flexion_extension_hip_L_per_trial": flexion_extension_hip_L_per_trial,
        "flexion_extension_hip_R_per_trial": flexion_extension_hip_R_per_trial,
        "adduction_abduction_hip_L_per_trial": adduction_abduction_hip_L_per_trial,
        "adduction_abduction_hip_R_per_trial": adduction_abduction_hip_R_per_trial,
        "flexion_extension_knee_L_per_trial": flexion_extension_knee_L_per_trial,
        "flexion_extension_knee_R_per_trial": flexion_extension_knee_R_per_trial,
        "flexion_extension_ankle_L_per_trial": flexion_extension_ankle_L_per_trial,
        "flexion_extension_ankle_R_per_trial": flexion_extension_ankle_R_per_trial,

        "angle_ankle_L_per_trial": angle_ankle_L_per_trial,
        "angle_ankle_R_per_trial": angle_ankle_R_per_trial,
        "angle_knee_L_per_trial": angle_knee_L_per_trial,
        "angle_knee_R_per_trial": angle_knee_R_per_trial,
        "angle_hip_L_per_trial": angle_hip_L_per_trial,
        "angle_hip_R_per_trial": angle_hip_R_per_trial,
    }

    return results


