"""
Script: validation.py

Description:
    This script performs a validation of the IMU-extracted joint angles 
    against VICON optical motion capture data. The validation process includes:

    1. Loading sensor data and trial times for each participant.
    2. Running the swimming algorithm to detect strokes and extract parameters.
    3. Computing joint angles from IMU data and extracting the corresponding 
       angles from VICON data for comparison.
    4. Processing joint angle data to compute errors, root mean square errors (RMSE), 
       and mean signed errors (MSE).
    5. Generating validation plots to visualize the comparison between IMU and 
       VICON data, including joint angles and rotation angles.

Dependencies:
    - Requires 'data_setip', 'swimming_algorithm', 'extract_parameters', and 
      'functions' modules from the 'src' directory.
    - Sensor and VICON data for each participant should be organized in 
      'Data/Validation_Data'.
    - The script generates validation figures and saves them in the pre-defined
      'Figures/Validation' directory.

Usage:
    Run this script in a Python environment where all dependencies are installed 
    (see requirements.txt). Ensure that the 'Data/Validation_Data' directory contains 
    sensor and VICON data for each participant.
    
Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import re
import sys
import time

# Access src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data_setup import *
from src.swimming_algorithm import *
from src.extract_parameters import *
from src.functions import *
from config import data_path, results_path, figures_path


############################### Functions ###############################


# Function to extract x, y, z coordinates from the string
def extract_coordinates(string):
    coords = re.findall(r"[-+]?\d*\.\d+|\d+", string)
    return list(map(float, coords[:3]))  # Return the first three numbers as float

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def compute_rotation_matrix(coords1, coords2, coords3):
    v1 = np.array(coords2) - np.array(coords1)
    v2 = np.array(coords3) - np.array(coords1)

    # Normalize v1
    v1 = normalize(v1)

    # Project v2 onto the plane perpendicular to v1
    v2_proj = v2 - np.dot(v2, v1) * v1
    v2_proj = normalize(v2_proj)

    # Compute the third perpendicular vector using cross product
    v3 = np.cross(v1, v2_proj)
    v3 = normalize(v3)

    # Construct the rotation matrix
    R = np.column_stack((v1, v2_proj, v3))
    return R


############################### Validation ###############################


# Get the absolute path of the current working directory
validation_data_dir = os.path.join(data_path, "Validation_Data")

# Start the timer
start_time = time.time()

for participant in sorted(os.listdir(validation_data_dir)):
    if not participant.startswith('.') and os.path.isdir(os.path.join(validation_data_dir, participant)):

        # Define the path to the participant's sensor data directory
        participant_dir = os.path.join(validation_data_dir, participant)

        print('----------------------------')
        
        # -------- .mat files --------

        # Define the path to the sensor data
        data_path = os.path.join(participant_dir, 'Sensor_Data')

        # Load sensor data for the participant
        sensor_data_dict = load_sensor_data(data_path)

        # -------- .csv files --------

        # Load trial times for the participant
        trial_times_dict = extract_trial_times(data_path)
        
        # Run swimming algorithm to extract strokes
        print('... Running swimming algorithm')
        swimming_data = swimming_algorithm(sensor_data_dict, trial_times_dict)

        # Extract swimming parameters and save them to csv files
        print('... Extracting swimming parameters')
        swimming_parameters = extract_swimming_parameters(swimming_data, results_path, 
                                                          participant, participant, 
                                                          create_files=False)

        # Start validation
        print('... Performing validation')

        # Extract relevant variables
        sensor_positions = sensor_data_dict['sensor_positions']

        quat_per_trial = swimming_data['quat_per_trial']
        flexion_extension_hip_L_per_trial_sensors = swimming_data['flexion_extension_hip_L_per_trial']
        flexion_extension_hip_R_per_trial_sensors = swimming_data['flexion_extension_hip_R_per_trial']
        adduction_abduction_hip_L_per_trial_sensors = swimming_data['adduction_abduction_hip_L_per_trial']
        adduction_abduction_hip_R_per_trial_sensors = swimming_data['adduction_abduction_hip_R_per_trial']
        flexion_extension_knee_L_per_trial_sensors = swimming_data['flexion_extension_knee_L_per_trial']
        flexion_extension_knee_R_per_trial_sensors = swimming_data['flexion_extension_knee_R_per_trial']
        flexion_extension_ankle_L_per_trial_sensors = swimming_data['flexion_extension_ankle_L_per_trial']
        flexion_extension_ankle_R_per_trial_sensors = swimming_data['flexion_extension_ankle_R_per_trial']

        RoM_ankle_L_per_trial = swimming_parameters['RoM_ankle_L_per_trial']
        RoM_ankle_R_per_trial = swimming_parameters['RoM_ankle_R_per_trial']
        RoM_knee_L_per_trial = swimming_parameters['RoM_knee_L_per_trial']
        RoM_knee_R_per_trial = swimming_parameters['RoM_knee_R_per_trial']
        RoM_hip_L_per_trial = swimming_parameters['RoM_hip_L_per_trial']
        RoM_hip_R_per_trial = swimming_parameters['RoM_hip_R_per_trial']
        RoM_hip_L_adduction_abduction_per_trial = swimming_parameters['RoM_hip_L_adduction_abduction_per_trial']
        RoM_hip_R_adduction_abduction_per_trial = swimming_parameters['RoM_hip_R_adduction_abduction_per_trial']

        mean_ankle_per_trial = swimming_parameters['mean_ankle_per_trial']
        mean_knee_per_trial = swimming_parameters['mean_knee_per_trial']
        mean_hip_per_trial = swimming_parameters['mean_hip_per_trial']
        mean_hip_add_abd_per_trial = swimming_parameters['mean_hip_add_abd_per_trial']

        # Per trial per segment
        rotation_matrices_per_trial = []

        # Calibration
        vicon_data_path = os.path.join(participant_dir, 'Vicon_Data')
        csv_file_path = os.path.join(vicon_data_path, 'calib_01.csv')
        df = pd.read_csv(csv_file_path)

        # Plot and save for each joint
        plot_dir = os.path.join(figures_path, 'Validation/', participant)
        os.makedirs(plot_dir, exist_ok=True)
        df.columns = df.columns.str.strip()

        # Assuming each segment follows 'SegmentX1', 'SegmentX2', 'SegmentX3' naming convention
        segments = set(name[:-1] for name in df.columns if name[-1] in '123')

        # Example usage within your loop structure
        rotation_matrices_dict = {segment: [] for segment in segments}
        quaternions_dict = {segment: [] for segment in segments}

        for index, row in df.iterrows():
            for segment in segments:
                segment_markers = [segment + '1', segment + '2', segment + '3']
                coords = [extract_coordinates(row[marker]) for marker in segment_markers]
                rotation_matrix = compute_rotation_matrix(*coords)
                rotation_matrices_dict[segment].append(rotation_matrix)
                quaternion_temp = R.from_matrix(rotation_matrix).as_quat()
                quaternions_dict[segment].append(quaternion_temp)

        # Now compute the average rotation matrix for each segment
        average_rotation_matrices = {}
        calibration_quat_per_segment = []
        for segment, quaternions in quaternions_dict.items():
            # Average quaternion over entire 15s calibration
            avg_quat = np.mean(quaternions, axis=0)
            calibration_quat_per_segment.append(avg_quat)
            average_rotation_matrices[segment] = R.from_quat(avg_quat).as_matrix()

            # Calculate pitch angle (y-axis) of foot sensors
            if segment == 'FrameRightFoot':
                yaw, pitch, roll = rotation_matrix_to_euler_angles(average_rotation_matrices['FrameRightFoot'])
                pitch_foot_R = pitch
                roll_foot_R = 180 + roll
                
            if segment == 'FrameLeftFoot':
                yaw, pitch, roll = rotation_matrix_to_euler_angles(average_rotation_matrices['FrameLeftFoot'])
                pitch_foot_L = pitch
                roll_foot_L = 180 - roll

        # Get sensor number of different positions
        position_name_vicon = ['FrameLeftFoot', 'FrameRightFoot', 'FrameLeftAnkle', 'FrameRightAnkle',
                            'FrameLeftThigh', 'FrameRightThigh', 'FrameUpperBack', 'FrameLowerBack']
                
        segment_nr = len(segments)
        segments = list(segments)
        segment_no = np.zeros(segment_nr)

        for j in range(segment_nr):
                for k in range(segment_nr):
                    if (segments[j] == position_name_vicon[k]):
                        idx = position_name_vicon.index(position_name_vicon[k])
                        segment_no[idx] = j

        foot_nr_L_vicon = int(segment_no[0])
        foot_nr_R_vicon = int(segment_no[1])    
        ankle_nr_L_vicon = int(segment_no[2])
        ankle_nr_R_vicon = int(segment_no[3])
        thigh_nr_L_vicon = int(segment_no[4])
        thigh_nr_R_vicon = int(segment_no[5])    
        back_nr_U_vicon = int(segment_no[6])
        back_nr_L_vicon = int(segment_no[7])

        # Define the Euler angle corrections for each segment
        corrections = {
            'FrameLeftFoot': R.from_euler('xyz', [90-pitch_foot_L, -90-roll_foot_L, 90], degrees=True),
            'FrameRightFoot': R.from_euler('xyz', [-90+pitch_foot_R, -90+roll_foot_R, 90], degrees=True),
            'FrameLeftAnkle': R.from_euler('xyz', [180, 0, 90], degrees=True),
            'FrameRightAnkle': R.from_euler('xyz', [0, 0, 90], degrees=True),
            'FrameLeftThigh': R.from_euler('xyz', [180, 0, 90], degrees=True),
            'FrameRightThigh': R.from_euler('xyz', [0, 0, 90], degrees=True),
            'FrameUpperBack': R.from_euler('xyz', [-90, 0, 90], degrees=True),
            'FrameLowerBack': R.from_euler('xyz', [-90, 0, 90], degrees=True),
        }

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

        # Initialize lists to store RMSE values
        average_rmse_ankle_L = []
        average_rmse_ankle_R = []
        average_rmse_knee_L = []
        average_rmse_knee_R = []
        average_rmse_hip_L = []
        average_rmse_hip_R = []
        average_rmse_hip_add_abd_L = []
        average_rmse_hip_add_abd_R = []

        mean_ankle_vicon_L = []
        mean_knee_vicon_L = []
        mean_hip_vicon_L = []
        mean_hip_add_abd_vicon_L = []
        mean_ankle_vicon_R = []
        mean_knee_vicon_R = []
        mean_hip_vicon_R = []
        mean_hip_add_abd_vicon_R = []

        mean_ankle_sensors_L = []
        mean_knee_sensors_L = []
        mean_hip_sensors_L = []
        mean_hip_add_abd_sensors_L = []
        mean_ankle_sensors_R = []
        mean_knee_sensors_R = []
        mean_hip_sensors_R = []
        mean_hip_add_abd_sensors_R = []

        quat_per_trial_vicon = []

        for trial in range(1, 4):

            # Calibration
            csv_file_path = os.path.join(vicon_data_path, f'swim_0{trial}.csv')
            df = pd.read_csv(csv_file_path)
            df.columns = df.columns.str.strip()
            segments = set(name[:-1] for name in df.columns if name[-1] in '123')

            rotation_matrices_dict = {segment: [] for segment in segments}

            for index, row in df.iterrows():
                for segment in segments:
                    segment_markers = [segment + '1', segment + '2', segment + '3']
                    coords = [extract_coordinates(row[marker]) for marker in segment_markers]
                    rotation_matrix = compute_rotation_matrix(*coords)
                    rotation_matrices_dict[segment].append(rotation_matrix)

            # Correct the segment positions
            corrected_rotation_matrices_dict = {segment: [] for segment in segments}
            quat_per_segment_vicon = []
            for segment in segments:
                correction = corrections[segment].as_matrix()  # Get the correction matrix for the segment
                quaternion_per_timestep = []
                for rotation_mat in rotation_matrices_dict[segment]:
                    corrected_rotation_mat = rotation_mat @ correction  # Matrix multiplication to apply the correction
                    corrected_rotation_matrices_dict[segment].append(corrected_rotation_mat)

                    # Validation of quaternions (Reviewer comment)
                    quaternion_temp = R.from_matrix(corrected_rotation_mat).as_quat()
                    quaternion_per_timestep.append(quaternion_temp)
                quat_per_segment_vicon.append(quaternion_per_timestep)
            
            quat_per_trial_vicon.append(quat_per_segment_vicon)

            ###################################################################################################


            # Rotation matrices per position of sensor
            rotation_matrix_foot_L = corrected_rotation_matrices_dict['FrameLeftFoot']
            rotation_matrix_foot_R = corrected_rotation_matrices_dict['FrameRightFoot']
            rotation_matrix_ankle_L = corrected_rotation_matrices_dict['FrameLeftAnkle']
            rotation_matrix_ankle_R = corrected_rotation_matrices_dict['FrameRightAnkle']
            rotation_matrix_thigh_L = corrected_rotation_matrices_dict['FrameLeftThigh']
            rotation_matrix_thigh_R = corrected_rotation_matrices_dict['FrameRightThigh']
            rotation_matrix_back_U = corrected_rotation_matrices_dict['FrameUpperBack']
            rotation_matrix_back_L = corrected_rotation_matrices_dict['FrameLowerBack']
            
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
                flexion_extension_knee_L.append(180+yaw)

                '--------------------------- Right knee joint angle ---------------------------'
                rotation_matrix_knee_R = rotation_matrix_thigh_R[timestep].T @ rotation_matrix_ankle_R[timestep]
                angle = absolute_angle_of_rotation_matrix(rotation_matrix_knee_R)
                angle_knee_R.append(angle)

                # Convert the rotation matrix to Euler angles (pitch, roll, yaw)
                yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_knee_R)

                # Extract the yaw angle (extension/flexion)
                flexion_extension_knee_R.append(180+yaw)

                '--------------------------- Left hip joint angle ---------------------------'
                rotation_matrix_hip_L = rotation_matrix_thigh_L[timestep].T @ rotation_matrix_back_L[timestep]
                angle = absolute_angle_of_rotation_matrix(rotation_matrix_hip_L)
                angle_hip_L.append(angle)

                # Convert the rotation matrix to Euler angles (pitch, roll, yaw)
                yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_hip_L)

                # Extract the yaw angle (extension/flexion)
                flexion_extension_hip_L.append(180+yaw)

                # Extract the pitch angle (adduction/abduction
                adduction_abduction_hip_L.append(roll)

                '--------------------------- Right hip joint angle ---------------------------'
                rotation_matrix_hip_R = rotation_matrix_thigh_R[timestep].T @ rotation_matrix_back_L[timestep]
                angle = absolute_angle_of_rotation_matrix(rotation_matrix_hip_R)
                angle_hip_R.append(angle)

                # Convert the rotation matrix to Euler angles (pitch, roll, yaw)
                yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix_hip_R)

                # Extract the yaw angle (extension/flexion)
                flexion_extension_hip_R.append(180+yaw)

                # Extract the pitch angle (adduction/abduction
                adduction_abduction_hip_R.append(roll)

            # Left ankle
            plt.title('Left Trial ' + str(trial) + ' ' + participant)
            b, a = butter(3, 0.01, 'lowpass')
            flexion_extension_ankle_L = filtfilt(b, a, np.array(flexion_extension_ankle_L))
            plt.plot(flexion_extension_ankle_L, label='vicon')
            plt.plot(flexion_extension_ankle_L_per_trial_sensors[trial-1][:len(flexion_extension_ankle_L)], label='sensors')
            plt.legend()
            # plt.savefig(plot_dir +'/ankle_angles_trial' + str(trial) + '_Left.png', dpi=300)
            plt.close()
            
            # Right ankle
            plt.title('Right Trial ' + str(trial) + ' ' + participant)
            b, a = butter(3, 0.01, 'lowpass')
            flexion_extension_ankle_R = filtfilt(b, a, np.array(flexion_extension_ankle_R))
            plt.plot(flexion_extension_ankle_R, label='vicon')
            plt.plot(flexion_extension_ankle_R_per_trial_sensors[trial-1][:len(flexion_extension_ankle_R)], label='sensors')
            plt.legend()
            # plt.savefig(plot_dir +'/ankel_angles_trial' + str(trial) + '_Right.png', dpi=300)
            plt.close()

            # Left knee
            plt.title('Left Trial ' + str(trial) + ' ' + participant)
            b, a = butter(3, 0.01, 'lowpass')
            flexion_extension_knee_L = filtfilt(b, a, np.array(flexion_extension_knee_L))
            plt.plot(flexion_extension_knee_L, label='vicon')
            plt.plot(flexion_extension_knee_L_per_trial_sensors[trial-1][:len(flexion_extension_knee_L)], label='sensors')
            plt.legend()
            # plt.savefig(plot_dir +'/knee_angles_trial' + str(trial) + '_Left.png', dpi=300)
            plt.close()
            
            # Right knee
            plt.title('Right Trial ' + str(trial) + ' ' + participant)
            b, a = butter(3, 0.01, 'lowpass')
            flexion_extension_knee_R = filtfilt(b, a, np.array(flexion_extension_knee_R))
            plt.plot(flexion_extension_knee_R, label='vicon')
            plt.plot(flexion_extension_knee_R_per_trial_sensors[trial-1][:len(flexion_extension_knee_R)], label='sensors')
            plt.legend()
            # plt.savefig(plot_dir +'/knee_angles_trial' + str(trial) + '_Right.png', dpi=300)
            plt.close()

            # Left hip
            plt.title('Left Trial ' + str(trial) + ' ' + participant)
            b, a = butter(3, 0.01, 'lowpass')
            flexion_extension_hip_L = filtfilt(b, a, np.array(flexion_extension_hip_L))
            plt.plot(flexion_extension_hip_L, label='vicon')
            plt.plot(flexion_extension_hip_L_per_trial_sensors[trial-1][:len(flexion_extension_hip_L)], label='sensors')
            plt.legend()
            # plt.savefig(plot_dir +'/hip_angles_trial' + str(trial) + '_Left.png', dpi=300)
            plt.close()
            
            # Right hip
            plt.title('Right Trial ' + str(trial) + ' ' + participant)
            b, a = butter(3, 0.01, 'lowpass')
            flexion_extension_hip_R = filtfilt(b, a, np.array(flexion_extension_hip_R))
            plt.plot(flexion_extension_hip_R, label='vicon')
            plt.plot(flexion_extension_hip_R_per_trial_sensors[trial-1][:len(flexion_extension_hip_R)], label='sensors')
            plt.legend()
            # plt.savefig(plot_dir +'/hip_angles_trial' + str(trial) + '_Right.png', dpi=300)
            plt.close()

            # Left hip abd/add
            plt.title('Left Trial ' + str(trial) + ' ' + participant)
            b, a = butter(3, 0.01, 'lowpass')
            adduction_abduction_hip_L = filtfilt(b, a, np.array(adduction_abduction_hip_L))
            plt.plot(adduction_abduction_hip_L, label='vicon')
            plt.plot(adduction_abduction_hip_L_per_trial_sensors[trial-1][:len(adduction_abduction_hip_L)], label='sensors')
            plt.legend()
            # plt.savefig(plot_dir +'/hip_abd_add_angles_trial' + str(trial) + '_Left.png', dpi=300)
            plt.close()
            
            # Right hip abd/add
            plt.title('Right Trial ' + str(trial) + ' ' + participant)
            b, a = butter(3, 0.01, 'lowpass')
            adduction_abduction_hip_R = filtfilt(b, a, np.array(adduction_abduction_hip_R))
            plt.plot(adduction_abduction_hip_R, label='vicon')
            plt.plot(adduction_abduction_hip_R_per_trial_sensors[trial-1][:len(adduction_abduction_hip_R)], label='sensors')
            plt.legend()
            # plt.savefig(plot_dir +'/hip_abd_add_angles_trial' + str(trial) + '_Right.png', dpi=300)
            plt.close()

            # Errors for each joint angle
            error_ankle_L = flexion_extension_ankle_L - flexion_extension_ankle_L_per_trial_sensors[trial-1][:len(flexion_extension_ankle_L)]
            error_ankle_R = flexion_extension_ankle_R - flexion_extension_ankle_R_per_trial_sensors[trial-1][:len(flexion_extension_ankle_R)]
            error_knee_L = flexion_extension_knee_L - flexion_extension_knee_L_per_trial_sensors[trial-1][:len(flexion_extension_knee_L)]
            error_knee_R = flexion_extension_knee_R - flexion_extension_knee_R_per_trial_sensors[trial-1][:len(flexion_extension_knee_R)]
            error_hip_L = flexion_extension_hip_L - flexion_extension_hip_L_per_trial_sensors[trial-1][:len(flexion_extension_hip_L)]
            error_hip_R = flexion_extension_hip_R - flexion_extension_hip_R_per_trial_sensors[trial-1][:len(flexion_extension_hip_R)]
            error_hip_add_abd_L = adduction_abduction_hip_L - adduction_abduction_hip_L_per_trial_sensors[trial-1][:len(adduction_abduction_hip_L)]
            error_hip_add_abd_R = adduction_abduction_hip_R - adduction_abduction_hip_R_per_trial_sensors[trial-1][:len(adduction_abduction_hip_R)]

            if participant == '01':
                # Remove entries from 1000 to 2000 (01 Trial 1) – drift error?
                error_ankle_L = np.concatenate((error_ankle_L[:1000], error_ankle_L[2000:]))

            # Calculate RMSE for each joint angle
            rmse_ankle_L = np.sqrt(np.mean(np.square(error_ankle_L)))
            rmse_ankle_R = np.sqrt(np.mean(np.square(error_ankle_R)))
            rmse_knee_L = np.sqrt(np.mean(np.square(error_knee_L)))
            rmse_knee_R = np.sqrt(np.mean(np.square(error_knee_R)))
            rmse_hip_L = np.sqrt(np.mean(np.square(error_hip_L)))
            rmse_hip_R = np.sqrt(np.mean(np.square(error_hip_R)))
            rmse_hip_add_abd_L = np.sqrt(np.mean(np.square(error_hip_add_abd_L)))
            rmse_hip_add_abd_R = np.sqrt(np.mean(np.square(error_hip_add_abd_R)))

            # --------- Mean Signed Error (MSE) ---------
            mse_ankle_L = np.mean(error_ankle_L)
            mse_ankle_R = np.mean(error_ankle_R)
            mse_knee_L = np.mean(error_knee_L)
            mse_knee_R = np.mean(error_knee_R)
            mse_hip_L = np.mean(error_hip_L)
            mse_hip_R = np.mean(error_hip_R)
            mse_hip_add_abd_L = np.mean(error_hip_add_abd_L)
            mse_hip_add_abd_R = np.mean(error_hip_add_abd_R)

            # # You can print or use the values as needed
            # print("Mean Signed Error - Ankle Left:", mse_ankle_L)
            # print("Mean Signed Error - Ankle Right:", mse_ankle_R)
            # print("Mean Signed Error - Knee Left:", mse_knee_L)
            # print("Mean Signed Error - Knee Right:", mse_knee_R)
            # print("Mean Signed Error - Hip Left:", mse_hip_L)
            # print("Mean Signed Error - Hip Right:", mse_hip_R)
            # print("Mean Signed Error - Hip Add/Abd Left:", mse_hip_add_abd_L)
            # print("Mean Signed Error - Hip Add/Abd Right:", mse_hip_add_abd_R)

            # # Output the results for each trial
            # print(f'Participant: {participant}, Trial: {trial}')
            # print(f'Left Ankle RMSE: {rmse_ankle_L}')
            # print(f'Right Ankle RMSE: {rmse_ankle_R}')
            # print(f'Left Knee RMSE: {rmse_knee_L}')
            # print(f'Right Knee RMSE: {rmse_knee_R}')
            # print(f'Left Hip RMSE: {rmse_hip_L}')
            # print(f'Right Hip RMSE: {rmse_hip_R}')
            # print(f'Left Hip add/abd RMSE: {rmse_hip_add_abd_L}')
            # print(f'Right Hip add/abd RMSE: {rmse_hip_add_abd_R}')

            # Append RMSE values to the corresponding lists
            average_rmse_ankle_L.append(rmse_ankle_L)
            average_rmse_ankle_R.append(rmse_ankle_R)
            average_rmse_knee_L.append(rmse_knee_L)
            average_rmse_knee_R.append(rmse_knee_R)
            average_rmse_hip_L.append(rmse_hip_L)
            average_rmse_hip_R.append(rmse_hip_R)
            average_rmse_hip_add_abd_L.append(rmse_hip_add_abd_L)
            average_rmse_hip_add_abd_R.append(rmse_hip_add_abd_R)
                    
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

            # Felxion/extension
            angle_hip_per_trial = [flexion_extension_hip_L_per_trial, flexion_extension_hip_R_per_trial]
            angle_hip_add_abd_per_trial = [adduction_abduction_hip_L_per_trial, adduction_abduction_hip_R_per_trial]
            angle_knee_per_trial = [flexion_extension_knee_L_per_trial, flexion_extension_knee_R_per_trial]
            angle_ankle_per_trial = [flexion_extension_ankle_L_per_trial, flexion_extension_ankle_R_per_trial]

            # Define variables per side
            stroke_nr_per_side = []
            swimming_speed_per_side = []
            average_stroke_duration_per_side = []
            variability_stroke_duration_per_side = []
            stroke_rate_per_side = []

            sides = ['L', 'R']

            # Set minimum time between strokes (1 frame)
            min_distance = 1
            
            # Set cutoff for Butterworth filter
            cut_off = 0.01

            for side in range(len(sides)):

                peaks, peak_values = find_stroke_peaks(angle_knee_per_trial[side][trial-1], 
                                                    cut_off, 
                                                    min_distance, 
                                                    plot=False)
                strokes = detect_swimming_strokes(angle_knee_per_trial[side][trial-1], 
                                                peaks, 
                                                peak_values, 
                                                plot=False)
                
            # Filter angles to remove noise
            b, a = butter(3, cut_off, 'lowpass')
            angle_knee_per_trial[0][trial-1] = filtfilt(b, a, angle_knee_per_trial[0][trial-1])
            angle_knee_per_trial[1][trial-1] = filtfilt(b, a, angle_knee_per_trial[1][trial-1])


            # Calculate range of motion (RoM) for each trial

            # Variables
            angles_filtered_ankle_L = []
            angles_filtered_ankle_R = []
            angles_filtered_knee_L = []
            angles_filtered_knee_R = []
            angles_filtered_hip_L = []
            angles_filtered_hip_R = []
            angles_filtered_hip_L_coronal = []
            angles_filtered_hip_R_coronal = []

            # For cyclogramms
            angles_filtered_ankle_L_per_stroke  = []
            angles_filtered_ankle_R_per_stroke  = []
            angles_filtered_knee_L_per_stroke = []
            angles_filtered_knee_R_per_stroke = []
            angles_filtered_hip_L_per_stroke = []
            angles_filtered_hip_R_per_stroke = []
            angles_filtered_hip_add_abd_L_per_stroke = []
            angles_filtered_hip_add_abd_R_per_stroke = []

            i = 0
            # Filter angles to only include valid strokes
            for stroke in strokes:

                # Ankle
                angles_filtered_ankle_L.extend(flexion_extension_ankle_L_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_ankle_R.extend(flexion_extension_ankle_R_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                # For cyclogramms
                angles_filtered_ankle_L_per_stroke.append(flexion_extension_ankle_L_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_ankle_R_per_stroke.append(flexion_extension_ankle_R_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])

                # Knee
                angles_filtered_knee_L.extend(flexion_extension_knee_L_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_knee_R.extend(flexion_extension_knee_R_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                # For cyclogramms
                angles_filtered_knee_L_per_stroke.append(flexion_extension_knee_L_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_knee_R_per_stroke.append(flexion_extension_knee_R_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])

                # Hip
                angles_filtered_hip_L.extend(flexion_extension_hip_L_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_hip_R.extend(flexion_extension_hip_R_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_hip_L_coronal.extend(adduction_abduction_hip_L_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_hip_R_coronal.extend(adduction_abduction_hip_R_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                # For cyclogramms
                angles_filtered_hip_L_per_stroke.append(flexion_extension_hip_L_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_hip_R_per_stroke.append(flexion_extension_hip_R_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_hip_add_abd_L_per_stroke.append(adduction_abduction_hip_L_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])
                angles_filtered_hip_add_abd_R_per_stroke.append(adduction_abduction_hip_R_per_trial[trial-1][strokes[i][0]:strokes[i][-1]])

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

            # # Print for validation
            # print('Vicon')
            # print('Trial', trial)
            # print('Left Ankle ROM', RoM_ankle_L)
            # print('Right Ankle ROM', RoM_ankle_R)
            # print('Left Knee ROM', RoM_knee_L)
            # print('Right Knee ROM', RoM_knee_R)
            # print('Left Hip ROM', RoM_hip_L)
            # print('Right Hip ROM', RoM_hip_R)
            # print('Left Hip Adduction/Abduction ROM', RoM_hip_L_adduction_abduction)
            # print('Right Hip Adduction/Abduction ROM', RoM_hip_R_adduction_abduction)

            # For cyclogramms
            angle_ankle_per_stroke_filtered = [angles_filtered_ankle_L_per_stroke, angles_filtered_ankle_R_per_stroke]
            angle_knee_per_stroke_filtered = [angles_filtered_knee_L_per_stroke, angles_filtered_knee_R_per_stroke]
            angle_hip_per_stroke_filtered = [angles_filtered_hip_L_per_stroke, angles_filtered_hip_R_per_stroke]
            angle_hip_add_abd_per_stroke_filtered = [angles_filtered_hip_add_abd_L_per_stroke, angles_filtered_hip_add_abd_R_per_stroke]
        
            save_plots = False

            for side in range(len(sides)):

                # Hip-knee
                _, _, mean_hip_vicon, mean_knee_vicon = center_angles(angle_hip_per_stroke_filtered[side],
                                                                    angle_knee_per_stroke_filtered[side])
                
                # Ankle-knee
                _, _, mean_ankle_vicon, _ = center_angles(angle_ankle_per_stroke_filtered[side],
                                                        angle_knee_per_stroke_filtered[side])
                
                # Hip adduction / abduction
                _, _, mean_hip_add_abd_vicon, _ = center_angles(angle_hip_add_abd_per_stroke_filtered[side],
                                                                angle_hip_per_stroke_filtered[side])

                if side == 0:
                    mean_ankle_vicon_L.append(mean_ankle_vicon)
                    mean_knee_vicon_L.append(mean_knee_vicon)
                    mean_hip_vicon_L.append(mean_hip_vicon)
                    mean_hip_add_abd_vicon_L.append(mean_hip_add_abd_vicon)
                else:
                    mean_ankle_vicon_R.append(mean_ankle_vicon)
                    mean_knee_vicon_R.append(mean_knee_vicon)
                    mean_hip_vicon_R.append(mean_hip_vicon)
                    mean_hip_add_abd_vicon_R.append(mean_hip_add_abd_vicon)

                if side == 0:
                    mean_ankle_sensors_L.append(mean_ankle_per_trial[trial-1][side])
                    mean_knee_sensors_L.append(mean_knee_per_trial[trial-1][side])
                    mean_hip_sensors_L.append(mean_hip_per_trial[trial-1][side])
                    mean_hip_add_abd_sensors_L.append(mean_hip_add_abd_per_trial[trial-1][side])
                else:
                    mean_ankle_sensors_R.append(mean_ankle_per_trial[trial-1][side])
                    mean_knee_sensors_R.append(mean_knee_per_trial[trial-1][side])
                    mean_hip_sensors_R.append(mean_hip_per_trial[trial-1][side])
                    mean_hip_add_abd_sensors_R.append(mean_hip_add_abd_per_trial[trial-1][side])
        
        # Average over all trials
        if participant == '01':
                std_ankle_vicon_L = np.std(mean_ankle_vicon_L[1:], axis=0)
                mean_ankle_vicon_L = np.mean(mean_ankle_vicon_L[1:], axis=0)
                std_knee_vicon_L = np.std(mean_knee_vicon_L[1:], axis=0)
                mean_knee_vicon_L = np.mean(mean_knee_vicon_L[1:], axis=0)
        
        else:
            std_ankle_vicon_L = np.std(mean_ankle_vicon_L, axis=0)
            mean_ankle_vicon_L = np.mean(mean_ankle_vicon_L, axis=0)
            std_knee_vicon_L = np.std(mean_knee_vicon_L, axis=0)
            mean_knee_vicon_L = np.mean(mean_knee_vicon_L, axis=0)

        # Sensors
        std_ankle_sensors_L = np.std(mean_ankle_sensors_L, axis=0)
        std_knee_sensors_L = np.std(mean_knee_sensors_L, axis=0)
        std_hip_sensors_L = np.std(mean_hip_sensors_L, axis=0)
        std_hip_add_abd_sensors_L = np.std(mean_hip_add_abd_sensors_L, axis=0)
        std_ankle_sensors_R = np.std(mean_ankle_sensors_R, axis=0)
        std_knee_sensors_R = np.std(mean_knee_sensors_R, axis=0)
        std_hip_sensors_R = np.std(mean_hip_sensors_R, axis=0)
        std_hip_add_abd_sensors_R = np.std(mean_hip_add_abd_sensors_R, axis=0)

        mean_ankle_sensors_L = np.mean(mean_ankle_sensors_L, axis=0)
        mean_knee_sensors_L = np.mean(mean_knee_sensors_L, axis=0)
        mean_hip_sensors_L = np.mean(mean_hip_sensors_L, axis=0)
        mean_hip_add_abd_sensors_L = np.mean(mean_hip_add_abd_sensors_L, axis=0)
        mean_ankle_sensors_R = np.mean(mean_ankle_sensors_R, axis=0)
        mean_knee_sensors_R = np.mean(mean_knee_sensors_R, axis=0)
        mean_hip_sensors_R = np.mean(mean_hip_sensors_R, axis=0)
        mean_hip_add_abd_sensors_R = np.mean(mean_hip_add_abd_sensors_R, axis=0)

        mean_ankle_sensors = [mean_ankle_sensors_L, mean_ankle_sensors_R]
        mean_knee_sensors = [mean_knee_sensors_L, mean_knee_sensors_R]
        mean_hip_sensors = [mean_hip_sensors_L, mean_hip_sensors_R]
        mean_hip_add_abd_sensors = [mean_hip_add_abd_sensors_L, mean_hip_add_abd_sensors_R]

        std_ankle_sensors = [std_ankle_sensors_L, std_ankle_sensors_R]
        std_knee_sensors = [std_knee_sensors_L, std_knee_sensors_R]
        std_hip_sensors = [std_hip_sensors_L, std_hip_sensors_R]
        std_hip_add_abd_sensors = [std_hip_add_abd_sensors_L, std_hip_add_abd_sensors_R]


        # Vicon
        std_hip_vicon_L = np.std(mean_hip_vicon_L, axis=0)
        std_hip_add_abd_vicon_L = np.std(mean_hip_add_abd_vicon_L, axis=0)
        std_ankle_vicon_R = np.std(mean_ankle_vicon_R, axis=0)
        std_knee_vicon_R = np.std(mean_knee_vicon_R, axis=0)
        std_hip_vicon_R = np.std(mean_hip_vicon_R, axis=0)
        std_hip_add_abd_vicon_R = np.std(mean_hip_add_abd_vicon_R, axis=0)

        # mean_knee_vicon_L = np.mean(mean_knee_vicon_L, axis=0)
        mean_hip_vicon_L = np.mean(mean_hip_vicon_L, axis=0)
        mean_hip_add_abd_vicon_L = np.mean(mean_hip_add_abd_vicon_L, axis=0)
        mean_ankle_vicon_R = np.mean(mean_ankle_vicon_R, axis=0)
        mean_knee_vicon_R = np.mean(mean_knee_vicon_R, axis=0)
        mean_hip_vicon_R = np.mean(mean_hip_vicon_R, axis=0)
        mean_hip_add_abd_vicon_R = np.mean(mean_hip_add_abd_vicon_R, axis=0)

        mean_ankle_vicon = [mean_ankle_vicon_L, mean_ankle_vicon_R]
        mean_knee_vicon = [mean_knee_vicon_L, mean_knee_vicon_R]
        mean_hip_vicon = [mean_hip_vicon_L, mean_hip_vicon_R]
        mean_hip_add_abd_vicon = [mean_hip_add_abd_vicon_L, mean_hip_add_abd_vicon_R]

        std_ankle_vicon = [std_ankle_vicon_L, std_ankle_vicon_R]
        std_knee_vicon = [std_knee_vicon_L, std_knee_vicon_R]
        std_hip_vicon = [std_hip_vicon_L, std_hip_vicon_R]
        std_hip_add_abd_vicon = [std_hip_add_abd_vicon_L, std_hip_add_abd_vicon_R]

        if participant == '01':
            title = 'Participant 1 – '
        else:
            title = 'Participant 2 – '

        for side in range(len(sides)): 

            # ------------ Ankle ------------
            fig = plt.figure(figsize=(10, 6))
            plt.title(sides[side], size=36)
            plt.plot(range(100), [0 - val for val in mean_ankle_vicon[side]], label='vicon', color='k')
            plt.plot(range(100), [0 - val for val in mean_ankle_sensors[side]], label='sensors', color='r')

            plt.fill_between(range(100), np.array([0 - val for val in mean_ankle_vicon[side]]) + np.array(std_ankle_vicon[side]), 
                            np.array([0 - val for val in mean_ankle_vicon[side]]) - np.array(std_ankle_vicon[side]), color='k', alpha=0.1)
            plt.fill_between(range(100), np.array([0 - val for val in mean_ankle_sensors[side]]) + np.array(std_ankle_sensors[side]), 
                            np.array([0 - val for val in mean_ankle_sensors[side]]) - np.array(std_ankle_sensors[side]), color='r', alpha=0.1)

            plt.xlabel('Stroke cycle [%]', size=30)
            plt.ylabel('Ankle dorsiflexion [deg]', size=30)
            plt.xticks(size=26)
            plt.yticks(ticks=[-60, -40, -20, 0, 20], labels=[-60, -40, -20, 0, 20], size=26)
            plt.ylim(-60, 25)

            # Remove the top and left spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if side == 0 and participant == '01':
                # Add arrow
                plt.annotate('', xy=(0.19, 0.8), xytext=(0.19, 0.4), 
                            arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                            xycoords='axes fraction')

                # Add vertical text
                plt.text(0.12, 0.6, 'dorsiflexion', rotation=90, va='center', ha='left', 
                        fontsize=26, color='black', transform=plt.gca().transAxes)
            plt.tight_layout()

            if side == 0 and participant == '01':
                plt.legend(fontsize=22)
            else:
                plt.legend().set_visible(False) 

            plt.savefig(plot_dir +'/Ankle_angles_' + sides[side] + '.pdf', dpi=300)
            plt.close()


            # ------------ Knee ------------
            fig = plt.figure(figsize=(10, 6))
            plt.plot(range(100), [180 - val for val in mean_knee_vicon[side]], label='vicon', color='k')
            plt.plot(range(100), [180 - val for val in mean_knee_sensors[side]], label='sensors', color='r')

            plt.fill_between(range(100), np.array([180 - val for val in mean_knee_vicon[side]]) + np.array(std_knee_vicon[side]), 
                            np.array([180 - val for val in mean_knee_vicon[side]]) - np.array(std_knee_vicon[side]), color='k', alpha=0.1)
            plt.fill_between(range(100), np.array([180 - val for val in mean_knee_sensors[side]]) + np.array(std_knee_sensors[side]), 
                            np.array([180 - val for val in mean_knee_sensors[side]]) - np.array(std_knee_sensors[side]), color='r', alpha=0.1)

            plt.xlabel('Stroke cycle [%]', size=30)
            plt.ylabel('Knee flexion [deg]', size=30)
            plt.xticks(size=26)
            plt.yticks(size=26)
            plt.ylim(-15, 125)

            # Remove the top and left spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if side == 0 and participant == '01':
                # Add arrow
                plt.annotate('', xy=(0.19, 0.8), xytext=(0.19, 0.4), 
                            arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                            xycoords='axes fraction')

                # Add vertical text
                plt.text(0.12, 0.6, 'flexion', rotation=90, va='center', ha='left', 
                        fontsize=26, color='black', transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.legend().set_visible(False)

            plt.savefig(plot_dir +'/Knee_angles_' + sides[side] + '.pdf', dpi=300)
            plt.close()


            # ------------ Hip ------------
            fig = plt.figure(figsize=(10, 6))
            plt.plot(range(100), [180 - val for val in mean_hip_vicon[side]], label='vicon', color='k')
            plt.plot(range(100), [180 - val for val in mean_hip_sensors[side]], label='sensors', color='r')

            plt.fill_between(range(100), np.array([180 - val for val in mean_hip_vicon[side]]) + np.array(std_hip_vicon[side]), 
                            np.array([180 - val for val in mean_hip_vicon[side]]) - np.array(std_hip_vicon[side]), color='k', alpha=0.1)
            plt.fill_between(range(100), np.array([180 - val for val in mean_hip_sensors[side]]) + np.array(std_hip_sensors[side]), 
                            np.array([180 - val for val in mean_hip_sensors[side]]) - np.array(std_hip_sensors[side]), color='r', alpha=0.1)

            plt.xlabel('Stroke cycle [%]', size=30)
            plt.ylabel('Hip flexion [deg]', size=30)
            plt.xticks(size=26)
            plt.yticks(size=26)
            plt.ylim(0, 60)

            # Remove the top and left spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if side == 0 and participant == '01':
                # Add arrow
                plt.annotate('', xy=(0.19, 0.8), xytext=(0.19, 0.4),
                            arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                            xycoords='axes fraction')

                # Add vertical text
                plt.text(0.12, 0.6, 'flexion', rotation=90, va='center', ha='left', 
                        fontsize=26, color='black', transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.legend().set_visible(False)

            plt.savefig(plot_dir +'/Hip_angles_' + sides[side] + '.pdf', dpi=300)
            plt.close()


            # ------------ Hip add abd ------------
            fig = plt.figure(figsize=(10, 6))
            if side == 0:
                i = 1
                j = 0
            else:
                i = -1
                if participant == '01':
                    j = 15
                else:
                    j = 0
            plt.plot(range(100), [i * val + j for val in mean_hip_add_abd_vicon[side]], label='vicon', color='k')
            plt.plot(range(100), [i * val + j for val in mean_hip_add_abd_sensors[side]], label='sensors', color='r')

            plt.fill_between(range(100), np.array([i * val + j for val in mean_hip_add_abd_vicon[side]]) + np.array(std_hip_add_abd_vicon[side]), 
                            np.array([i * val + j for val in mean_hip_add_abd_vicon[side]]) - np.array(std_hip_add_abd_vicon[side]), color='k', alpha=0.1)
            plt.fill_between(range(100), np.array([i * val + j for val in mean_hip_add_abd_sensors[side]]) + np.array(std_hip_add_abd_sensors[side]), 
                            np.array([i * val + j for val in mean_hip_add_abd_sensors[side]]) - np.array(std_hip_add_abd_sensors[side]), color='r', alpha=0.1)

            plt.xlabel('Stroke cycle [%]', size=30)
            plt.ylabel('Hip abduction [deg]', size=30)
            plt.xticks(size=26)
            plt.yticks(size=26)
            plt.ylim(-5, 30)

            # Remove the top and left spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if side == 0 and participant == '01':
                # Add arrow
                plt.annotate('', xy=(0.19, 0.8), xytext=(0.19, 0.4),
                            arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                            xycoords='axes fraction')

                # Add vertical text
                plt.text(0.12, 0.6, 'flexion', rotation=90, va='center', ha='left', 
                        fontsize=26, color='black', transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.legend().set_visible(False)

            plt.savefig(plot_dir + '/Hip_add_abd_angles_' + sides[side] + '.pdf', dpi=300)
            plt.close()



        # Function to calculate mean and standard deviation, then format them
        def mean_std(values):
            return f"{np.round(np.mean(values), 2)} ± {np.round(np.std(values), 2)}"

        # # Calculate and print the mean and standard deviation of each list
        # print(f'Average Left Ankle RMSE: {mean_std(average_rmse_ankle_L)}')
        # print(f'Average Right Ankle RMSE: {mean_std(average_rmse_ankle_R)}')
        # print(f'Average Left Knee RMSE: {mean_std(average_rmse_knee_L)}')
        # print(f'Average Right Knee RMSE: {mean_std(average_rmse_knee_R)}')
        # print(f'Average Left Hip RMSE: {mean_std(average_rmse_hip_L)}')
        # print(f'Average Right Hip RMSE: {mean_std(average_rmse_hip_R)}')
        # print(f'Average Left Hip add/abd RMSE: {mean_std(average_rmse_hip_add_abd_L)}')
        # print(f'Average Right Hip add/abd RMSE: {mean_std(average_rmse_hip_add_abd_R)}')



        ##################### Reviewer Comment #####################


        position_name = [
            'Foot_Left',    # foot_nr_L
            'Foot_Right',   # foot_nr_R
            'Ankle_Left',   # ankle_nr_L
            'Ankle_Right',  # ankle_nr_R
            'Thigh_Left',   # thigh_nr_L
            'Thigh_Right',  # thigh_nr_R
            'Back_Upper',   # back_nr_U
            'Back_Lower'    # back_nr_L
        ]

        position_name_title = [
            'Left foot',    # foot_nr_L
            'Right foot',   # foot_nr_R
            'Left ankle',   # ankle_nr_L
            'Right ankle',  # ankle_nr_R
            'Left thigh',   # thigh_nr_L
            'Right thigh',  # thigh_nr_R
            'Upper back',   # back_nr_U
            'Lower back'    # back_nr_L
        ]

        # Remove reference sensor segment
        for trial in range(len(quat_per_trial)):
            quat_per_trial[trial] = quat_per_trial[trial][:-1]

        imu_to_vicon_map = {
            'Foot_Left': 'FrameLeftFoot',
            'Foot_Right': 'FrameRightFoot',
            'Ankle_Left': 'FrameLeftAnkle',
            'Ankle_Right': 'FrameRightAnkle',
            'Thigh_Left': 'FrameLeftThigh',
            'Thigh_Right': 'FrameRightThigh',
            'Back_Upper': 'FrameUpperBack',
            'Back_Lower': 'FrameLowerBack',
        }

        # Create a list to store the aligned order
        aligned_order = []

        for imu_pos in position_name:
            vicon_pos = imu_to_vicon_map[imu_pos]  # Map IMU position to Vicon position
            imu_idx = sensor_positions.index(imu_pos)  # Find IMU index
            vicon_idx = list(segments).index(vicon_pos)  # Find Vicon index
            aligned_order.append((imu_idx, vicon_idx))

        # Reorder quat_per_trial_vicon based on aligned_order
        quat_per_trial_vicon = [
            [trial[vicon_idx] for _, vicon_idx in aligned_order] for trial in quat_per_trial_vicon
        ]

        quat_per_trial = [
            [trial[imu_idx] for imu_idx, _ in aligned_order] for trial in quat_per_trial
        ]

        # for (imu_idx, vicon_idx) in aligned_order:
        #     print(f"IMU Position: {position_name[imu_idx]} <-> Vicon Position: {position_name[vicon_idx]}")

        position_name = [
            'Left_Foot',    # foot_nr_L
            'Right_Foot',   # foot_nr_R
            'Left_Ankle',   # ankle_nr_L
            'Right_Ankle',  # ankle_nr_R
            'Left_Thigh',   # thigh_nr_L
            'Right_Thigh',  # thigh_nr_R
            'Upper_Back',   # back_nr_U
            'Lower_Back'    # back_nr_L
        ]


        # Separate systems (VICON and IMUs)


        def quaternion_rotation_angle(q0, qx):
            """
            Compute the rotation angle between two quaternions.
            :param q0: Initial quaternion [w, x, y, z].
            :param qx: Quaternion at time x [w, x, y, z].
            :return: Rotation angle in degrees.
            """
            # Normalize quaternions
            q0 = q0 / np.linalg.norm(q0)
            qx = qx / np.linalg.norm(qx)

            # Compute the dot product and angle
            dot_product = np.clip(np.dot(q0, qx), -1.0, 1.0)
            angle = 2 * np.arccos(np.abs(dot_product))  # Absolute to ensure valid angle
            return np.degrees(angle)

        def compute_rotation_angles(quat_data):
            """
            Compute rotation angles relative to the initial quaternion for each timestep.
            :param quat_data: List of quaternions [w, x, y, z] for each timestep.
            :return: List of rotation angles in degrees.
            """
            q0 = quat_data[0]  # Initial quaternion
            angles = [quaternion_rotation_angle(q0, qx) for qx in quat_data]
            return angles

        def plot_rotation_angles(time_imu, time_vicon, imu_angles, vicon_angles, trial, segment):
            """
            Plot rotation angles over time for IMU and VICON data.
            :param time: List of time points.
            :param imu_angles: List of rotation angles for IMU.
            :param vicon_angles: List of rotation angles for VICON.
            """
            from scipy.signal import butter, filtfilt
            
            # Define filter parameters
            cutoff_frequency = 5  # Adjust based on data characteristics
            sampling_frequency = 200  # Sampling rate in Hz
            filter_order = 4  # Butterworth filter order

            # Filter VICON angles with Butterworth filter
            nyquist = 0.5 * sampling_frequency  # Nyquist frequency (sampling frequency: 200Hz)
            normal_cutoff = cutoff_frequency / nyquist
            b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
            vicon_angles =  filtfilt(b, a, vicon_angles)

            plt.figure()
            plt.plot(time_vicon, vicon_angles, label='VICON', color='k')
            plt.plot(time_imu, imu_angles, label='IMU', color='r')
            plt.xlabel('Time [s]')
            plt.ylabel(r'Rotation Angle from $t_x$ to $t_0$ [deg]')
            plt.legend()
            plt.title(position_name_title[segment])
            plt.savefig(plot_dir + f'/Trial{trial + 1}_{position_name[segment]}.png', dpi=300)
            plt.close()

        for trial in range(3):
            for segment in range(8):
                time_points_imu = np.linspace(0, len(quat_per_trial[trial][segment])/200, len(quat_per_trial[trial][segment]))  # Generate time points
                time_points_vicon = np.linspace(0, len(quat_per_trial_vicon[trial][segment])/200, len(quat_per_trial_vicon[trial][segment]))
                imu_angles = compute_rotation_angles(quat_per_trial[trial][segment])  # First trial, first segment for IMU
                vicon_angles = compute_rotation_angles(quat_per_trial_vicon[trial][segment])  # First trial, first segment for VICON

                # Plot the rotation angles
                plot_rotation_angles(time_points_imu, time_points_vicon, imu_angles, vicon_angles, trial, segment)


        # Combined systems (VICON and IMUs)


        def quaternion_rotation_between(q1, q2):
            """
            Compute the rotation angle between two quaternions q1 and q2.
            :param q1: Quaternion from Vicon [w, x, y, z].
            :param q2: Quaternion from IMU [w, x, y, z].
            :return: Rotation angle in degrees.
            """
            # Normalize quaternions
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)

            # Compute the dot product and angle
            dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
            angle = 2 * np.arccos(np.abs(dot_product))  # Absolute ensures valid angle
            return np.degrees(angle)


        def plot_rotation_comparison(rotation_angles, trial, segment):
            """
            Plot rotation angles between Vicon and IMU quaternions over time.
            :param time: List of time points.
            :param rotation_angles: List of rotation angles between Vicon and IMU.
            """
            plt.figure()
            plt.plot(rotation_angles)
            plt.xlabel('Time [s]')
            plt.xticks(ticks=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], labels=[0, 5, 10, 15, 20, 25, 30, 35, 40])
            plt.ylabel('Rotation Angle VICON to IMU [deg]')
            plt.title(position_name[segment])
            plt.savefig(plot_dir + f'/Rotations_Trial{trial + 1}_{position_name[segment]}.pdf', dpi=300)
            plt.close()


        # # Previously used function
        # def align_quaternions(q_imu_static, q_vicon_static, quat_imu):
        #     """
        #     Align IMU quaternions to the VICON coordinate system using a static pose.
        #     :param q_imu_static: Static quaternion from IMU as [w, x, y, z].
        #     :param q_vicon_static: Static quaternion from VICON as [w, x, y, z].
        #     :param quat_imu: List of dynamic quaternions from IMU as [w, x, y, z].
        #     :return: List of aligned IMU quaternions as [w, x, y, z].
        #     """

        #     # Convert quaternions to rotation objects (x, y, z, w)
        #     q_imu_static_rot = R.from_quat([q_imu_static[1], q_imu_static[2], q_imu_static[3], q_imu_static[0]])
        #     q_vicon_static_rot = R.from_quat([q_vicon_static[1], q_vicon_static[2], q_vicon_static[3], q_vicon_static[0]])

        #     # Compute the alignment quaternion
        #     q_align = q_vicon_static_rot * q_imu_static_rot.inv()

        #     # Align dynamic IMU quaternions
        #     aligned_quat_imu = [
        #         (q_align * R.from_quat([q[1], q[2], q[3], q[0]])).as_quat() for q in quat_imu
        #     ]

        #     # Convert back to [w, x, y, z] format
        #     aligned_quat_imu = [[q[3], q[0], q[1], q[2]] for q in aligned_quat_imu]
        #     return aligned_quat_imu


        def normalize_quaternion_sign(q):
            """
            Ensure quaternion has a positive scalar component (w).
            :param q: Quaternion [w, x, y, z].
            :return: Normalized quaternion [w, x, y, z].
            """
            return q if q[0] >= 0 else [-q[0], -q[1], -q[2], -q[3]]


        def compute_rotation_angles_between(quat_vicon, quat_imu):
            """
            Compute rotation angles between corresponding VICON and IMU quaternions.
            :param quat_vicon: List of VICON quaternions [w, x, y, z].
            :param quat_imu: List of IMU quaternions [w, x, y, z].
            :return: List of rotation angles in degrees.
            """
            quat_vicon = [normalize_quaternion_sign(q) for q in quat_vicon]
            quat_imu = [normalize_quaternion_sign(q) for q in quat_imu]

            min_length = min(len(quat_imu), len(quat_vicon))
            quat_vicon = quat_vicon[:min_length]
            quat_imu = quat_imu[:min_length]

            angles = [quaternion_rotation_between(q_vicon, q_imu) for q_vicon, q_imu in zip(quat_vicon, quat_imu)]
            return angles

        def align_to_identity(q_static, quat_data):
            """
            Align quaternions to the identity matrix using the static quaternion.
            :param q_static: Static quaternion [w, x, y, z].
            :param quat_data: List of quaternions to be transformed.
            :return: List of aligned quaternions.
            """
            # Convert the static quaternion to a rotation object
            q_static_rot = R.from_quat([q_static[1], q_static[2], q_static[3], q_static[0]])
            
            # Compute the inverse rotation to bring the static quaternion to the identity
            q_to_identity = q_static_rot.inv()
            
            # Apply the transformation to all quaternions in the data
            aligned_quat_data = [
                (q_to_identity * R.from_quat([q[1], q[2], q[3], q[0]])).as_quat() for q in quat_data
            ]
            
            # Convert back to [w, x, y, z] format
            return [[q[3], q[0], q[1], q[2]] for q in aligned_quat_data]

        # # Match x, y, and z-axis of Vicon to IMUs
        # vicon_to_imu = R.from_euler('zyx', [180, 0, 90], degrees=True).as_matrix()
        vicon_to_imu = R.from_euler('zyx', [0, 0, 0], degrees=True).as_matrix()  # No change

        # Main Loop for All Trials and Segments
        for trial in range(3):
            for segment in range(8):
                # Extract static quaternions
                q_imu_static = quat_per_trial[trial][segment][0]
                q_vicon_static = quat_per_trial_vicon[trial][segment][0]

                # Convert and transform VICON static quaternion
                vicon_rot = R.from_quat([q_vicon_static[1], q_vicon_static[2], q_vicon_static[3], q_vicon_static[0]]).as_matrix()
                transformed_vicon_rot = vicon_to_imu @ vicon_rot
                transformed_q_vicon_static = R.from_matrix(transformed_vicon_rot).as_quat()
                transformed_q_vicon_static = [transformed_q_vicon_static[3], transformed_q_vicon_static[0], transformed_q_vicon_static[1], transformed_q_vicon_static[2]]

                # Align quaternions to identity
                aligned_quat_imu = align_to_identity(q_imu_static, quat_per_trial[trial][segment])
                aligned_quat_vicon = align_to_identity(transformed_q_vicon_static, quat_per_trial_vicon[trial][segment])

                # Compute rotation angles
                rotation_angles = compute_rotation_angles_between(aligned_quat_vicon, aligned_quat_imu)

                # # Check – should be close to [1, 0, 0, 0]
                # print("IMU Aligned Static:", aligned_quat_imu[0])
                # print("VICON Aligned Static:", aligned_quat_vicon[0])

                # Plot results
                plot_rotation_comparison(rotation_angles, trial, segment)

        # Stop the timer
        end_time = time.time()

        # Print execution time
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time/60:.2f} minutes")
