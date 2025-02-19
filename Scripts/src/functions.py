"""
Script: functions.py

Description:
    This script contains a collection of functions for analyzing 
    joint kinematics in swimming using wearable sensor data. It provides 
    methods for quaternion and rotation matrix calculations, 
    signal filtering, and stroke detection.

Usage:
    - This script is not meant to be run directly - the functions are imported 
      into the main processing pipeline.


Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import math
import numpy as np
from spatialmath.base import *
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks


def rotation_matrix_from_quaternion(Q):
    """Calculate rotation matrix from a quaternion.

    :arg Q: a quaternion in the form of an array.
    :returns: a rotation matrix in the form of an array.
    """

    w, x, y, z = Q[0], Q[1], Q[2], Q[3]
    R = np.empty((3, 3))

    R[0, 0] = w**2 + x**2 - y**2 - z**2
    R[0, 1] = 2*x*y - 2*w*z
    R[0, 2] = 2*w*y + 2*x*z
    R[1, 0] = 2*w*z + 2*x*y
    R[1, 1] = w**2 - x**2 + y**2 - z**2
    R[1, 2] = 2*y*z - 2*w*x
    R[2, 0] = 2*x*z - 2*w*y
    R[2, 1] = 2*w*x + 2*y*z
    R[2, 2] = w**2 - x**2 - y**2 + z**2

    return R


def absolute_angle_of_rotation_matrix(R):
    """Calculate the absolute angle of a rotation matrix. 
    This is also significant because it is the minimal angle needed 
    to rotate from the identity matrix to R, and hence is a useful 
    pseudo-norm for 3D rotations.

    :arg R: a rotation matrix in the form of an array.
    :returns: an angle in degrees.
    """

    r11 = R[0, 0]
    r22 = R[1, 1]
    r33 = R[2, 2]

    trace = r11 + r22 + r33
    angle = np.arccos((trace - 1) / 2)
    angle = np.degrees(angle)  # [deg] not [rad]

    return angle


def quaternion_multiplication(Q1, Q2):
    """Multiply two quaternions.

    :arg Q1: a quaternion in the form of an array.
    :arg Q2: a quaternion in the form of an array.    
    :returns: a quaternion in the form of an array.
    """

    w1, x1, y1, z1 = Q1[0], Q1[1], Q1[2], Q1[3]
    w2, x2, y2, z2 = Q2[0], Q2[1], Q2[2], Q2[3]
    Q = np.empty(4)

    Q[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    Q[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    Q[2] = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    Q[3] = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    return Q


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (yaw, pitch, roll) in radians.
    Note: yaw is a counterclockwise rotation about the z-axis,
    pitch is a counterclockwise rotation about the y-axis,
    roll is a counterclockwise rotation about the x-axis.
    
    :arg R (numpy array): 3x3 rotation matrix
    :returns: (tuple): Euler angles (yaw, pitch, roll) in degrees
    """
    if R.shape != (3, 3):
        raise ValueError("Input rotation matrix must be 3x3")
    
    # Yaw angle (ψ)
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    
    # Pitch angle (θ)
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    
    # Roll angle (φ)
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    
    return yaw, pitch, roll


def find_stroke_peaks(joint_angles, cut_off, min_distance, prominence_factor=None, plot=None):
    """Find stroke peaks and their positions based on angles maximum. 

    :arg joint_angles: joint angles [deg] saved in a list.
    :returns: list with timesteps where strokes are and also their values.
    """

    # Signal filter
    b, a = butter(3, cut_off, 'lowpass')
    filtered = filtfilt(b, a, joint_angles)

    peaks, properties = find_peaks(filtered,
                                   height=(50, 230), 
                                   distance=min_distance, 
                                   prominence=(10, 180))  # ST42 at 6WT & SS26 at HC need prominence (0, 180) !!!!!!
    
    peak_values = [property for property in properties.values()][0]

    if plot:
        # Plot the angles and their peaks over time
        plt.title('Absolute joint angles over time')
        plt.plot(filtered)
        plt.plot(peaks, peak_values, 'x')
        plt.xlabel('Frames')
        plt.ylabel('Angles [deg]')
        plt.show()

    return peaks, peak_values


def detect_swimming_strokes(joint_angles, stroke_peaks, peak_values, plot=None):
    """Detect swimming strokes based on angles i.e. peaks. 

    :arg stroke peaks: joint angles [deg] saved in a list.
    :returns: list with intervals of timesteps within the trial where strokes begin/end.
    """

    peak_nr = len(stroke_peaks)

    if plot:
        # Plot the angles and their peaks over time
        plt.title('Absolute joint angles over time')
        plt.plot(joint_angles)
        plt.plot(stroke_peaks, peak_values, 'x')
        for peak in range(peak_nr):
            plt.axvline(x=stroke_peaks[peak], label='axvline - full height', ls = '--', color='g')
        plt.xlabel('Timesteps')
        plt.ylabel('Degrees')
        plt.show()
    
    strokes = [[stroke_peaks[peak], stroke_peaks[peak+1]] for peak in range(peak_nr-1)]

    return strokes


def integrate_acceleration(acc_data, time_array):
    """Integrate acceleration data over one stroke to find velocity.
    We use trapezpoidal integration with a linear drift model.

    To find the direction of progress, we use an eigenvector decomposition.
    Note: the first principal component (i.e. the largest eigenvector 
    and associated largest eigenvalue) gives you the direction of 
    the maximum variability in the data.

    :arg acc_data: accelerometer data for the stroke.
    :arg time_array: the corresponding timestamps for each data point.
    :returns: the velocity over one stroke as an array, 
    which contains the integrated velocity values at each time step.
    """

    # Convert data to correct units
    acc_data = acc_data / 9.81  # [m/s^2] to [g]

    # Number of timesteps contained in time_array
    timesteps = len(time_array)

    # Remove gravity component from the z-axis of the acceleration
    gravity_component = np.concatenate((np.zeros((timesteps, 2)), np.ones((timesteps, 1))), axis=1)
    acc_data = acc_data - gravity_component

    # Initial and final condition
    initial_condition = np.array([0, 0, 0])
    final_condition = np.array([0, 0, 0])

    # Calculate velocity
    v = np.zeros((timesteps, 3))
    v[0, :] = initial_condition

    # Trapezoidal rule
    for i in range(timesteps-1):
        t = time_array[i+1] - time_array[i]
        v[i+1, :] = v[i, :] + (acc_data[i+1, :] + acc_data[i, :])/2 * t

    # Correct for drift with linear drift model
    duration = np.linspace(0, timesteps, timesteps)
    v_corr = duration[:, np.newaxis].T * ((v[-1, :] - final_condition) / duration[-1])[:, np.newaxis]
    v = v - v_corr.T

    return v


def integrate_velocity(velocity, time_array):
    """Integrate velocity over one stroke to find displacement.
    We use trapezpoidal integration with a linear drift model.

    :arg velocity: accelerometer data for the stroke.
    :arg time_array: contains the corresponding timestamps for each velocity.
    :returns: the displacement over one stroke as an array, 
    which contains the integrated displacement values at each time step.
    """

    # Number of timesteps contained in time_array
    timesteps = len(time_array)

    # Initial and final condition
    initial_condition = np.array([0, 0, 0])
    final_condition = np.array([0, 0, 0])

    # Calculate displacement
    displacement = np.zeros((timesteps, 3))
    displacement[0, :] = initial_condition

    # Integrate velocity to get displacement
    for i in range(timesteps-1):
        t = time_array[i+1] - time_array[i]
        displacement[i+1, :] = displacement[i, :] + (velocity[i+1, :] + velocity[i, :])/2 * t
    
    duration = np.linspace(0, timesteps, timesteps)

    # Correct displacement to end in zero 
    displacement_corr = duration[:, np.newaxis].T * (
            (displacement[-1, :] - final_condition)[:, np.newaxis]
            / duration[:, np.newaxis][-1])
    displacement = displacement - displacement_corr.T

    return displacement


def remove_outliers(stroke_parameter):
    """Remove strokes that are outliers using the Inter Quartile Range approach.

    :arg stroke_parameter: list of a certain parameter per stroke.
    :returns: (filtered) list of parameter without outliers.
    """

    Q1 = np.percentile(stroke_parameter, 25)
    Q3 = np.percentile(stroke_parameter, 75)
    IQR = Q3 - Q1

    # Find uper and lower limits
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # Remove the outliers
    index_upper = np.where(stroke_parameter >= upper_limit)[0]
    index_lower = np.where(stroke_parameter <= lower_limit)[0]

    parameter_filtered = []
    for index in range(len(stroke_parameter)):
        if index not in index_upper:
            if index not in index_lower:
                parameter_filtered.append(stroke_parameter[index])

    return parameter_filtered


def remove_outliers_angles(angles):
    """Remove strokes that are outliers in angles using 
    the Inter Quartile Range approach.

    :arg angles: list of lists with angles per stroke.
    :returns: (filtered) lists with angles per stroke without outliers
    and indices of removed strokes in reverse order.
    """

    # --------------- Filter for mean --------------- 

    # Indices of removed strokes
    indices = []
    indices_upper = []
    indices_lower = []

    # List of mean of each stroke
    mean_angle = []
    for angle in range(len(angles)):
        mean_angle.append(np.mean(angle))
    Q1 = np.percentile(mean_angle, 25)
    Q3 = np.percentile(mean_angle, 75)
    IQR = Q3 - Q1

    # Find uper and lower limits
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # Remove the outliers
    index_upper = np.where(mean_angle >= upper_limit)[0]
    index_lower = np.where(mean_angle <= lower_limit)[0]

    for idx in index_upper:
        indices_upper.append(idx)
    for idx in index_lower:
        indices_lower.append(idx)
    
    # --------------- Filter for length of list ---------------

    nr_angles = []
    for angle in angles:
        nr_angles.append(len(angle))
    Q1 = np.percentile(nr_angles, 25)
    Q3 = np.percentile(nr_angles, 75)
    IQR = Q3 - Q1

    # Find uper and lower limits
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # Remove the outliers
    index_upper = np.where(nr_angles >= upper_limit)[0]
    index_lower = np.where(nr_angles <= lower_limit)[0]

    for idx in index_upper:
        indices_upper.append(idx)
    for idx in index_lower:
        indices_lower.append(idx)

    # --------------- Filter for maximum ---------------

    max_angles = []
    for angle in angles:
        max_angles.append(max(angle))
    Q1 = np.percentile(max_angles, 25)
    Q3 = np.percentile(max_angles, 75)
    IQR = Q3 - Q1

    # Find uper and lower limits
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # Remove the outliers
    index_upper = np.where(max_angles >= upper_limit)[0]
    index_lower = np.where(max_angles <= lower_limit)[0]
    
    for idx in index_upper:
        indices_upper.append(idx)
    for idx in index_lower:
        indices_lower.append(idx)
    
    # --------------- Filter for minimum ---------------

    min_angles = []
    for angle in angles:
        min_angles.append(min(angle))
    Q1 = np.percentile(min_angles, 25)
    Q3 = np.percentile(min_angles, 75)
    IQR = Q3 - Q1

    # Find uper and lower limits
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # Remove the outliers
    index_upper = np.where(min_angles >= upper_limit)[0]
    index_lower = np.where(min_angles <= lower_limit)[0]

    for idx in index_upper:
        indices_upper.append(idx)
    for idx in index_lower:
        indices_lower.append(idx)
    
    # --------------- Filter for range ---------------

    range_angles = []
    for angle in range(len(angles)):
        range_angles.append(max_angles[angle] - min_angles[angle])
    Q1 = np.percentile(range_angles, 25)
    Q3 = np.percentile(range_angles, 75)
    IQR = Q3 - Q1

    # Find uper and lower limits
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # Remove the outliers
    index_upper = np.where(range_angles >= upper_limit)[0]
    index_lower = np.where(range_angles <= lower_limit)[0]

    for idx in index_upper:
        indices_upper.append(idx)
    for idx in index_lower:
        indices_lower.append(idx)
    
    # Sort list in reverse order and keep only unique values
    indices = list(set(indices_upper + indices_lower))
    indices.sort()
    indices = indices[::-1]

    angles_filtered = []
    for index in range(len(angles)):
        if index not in indices:
            angles_filtered.append(angles[index])

    return angles_filtered, indices


def interpolate_angle_cycle(angles_per_trial):
    """Calculate mean of angle cycles, when arrays
    are of different lengths by interpolating.
    """

    lengths = []
    for angle in angles_per_trial:
        lengths.append(len(angle))

    angles_interpolated = []
    for angles in angles_per_trial:
        ratio = 1/min(lengths) * len(angles)
        angles_interpolated.append(np.interp(np.arange(0, len(angles)-1, ratio), 
                                             np.arange(0, len(angles)), 
                                             angles))
    # Buffer
    for angles in angles_interpolated:
    
        if len(angles) < min(lengths)-1:
            angles_interpolated[angles_interpolated.index(angles)] = np.append(angles, angles[-1])
        
        if len(angles) < min(lengths):
            angles_interpolated[angles_interpolated.index(angles)] = np.append(angles, angles[-1])

    return angles_interpolated


def interpolate_angle_cycle_to_100(angles_per_trial):
    """Calculate mean of angle cycles, when arrays
    are of different lengths by interpolating to 100.
    """

    lengths = []
    for angle in angles_per_trial:
        lengths.append(len(angle))

    angles_interpolated = []
    for angles in angles_per_trial:
        ratio = 1/100 * len(angles)
        angles_interpolated.append(np.interp(np.arange(0, len(angles)-1, ratio), 
                                             np.arange(0, len(angles)), 
                                             angles))
    # Buffer
    for angles in angles_interpolated:

        if len(angles) < 99:
            angles_interpolated[angles_interpolated.index(angles)] = np.append(angles, angles[-1])
        
        if len(angles) < 100:
            angles_interpolated[angles_interpolated.index(angles)] = np.append(angles, angles[-1])

    return angles_interpolated


def remove_outliers_cyclogram(strokes, displacement_per_stroke, i):
    """Remove strokes that are outliers in displacement using 
    the Inter Quartile Range approach.

    :arg strokes: list with intervals of timesteps within the trial 
    where strokes begin/end.
    :arg displacement_per_stroke: list with arrays of displacement 
    in x, y and z per stroke.
    :arg i: x, y, z-axis (0: sidewards, 1: horizontal, 2: vertical)
    :returns: (filtered) list of strokes without outliers and 
    (filtered) list of displacements without outliers.
    """

    # --------------- Filter for mean of displacement --------------- 

    # List of mean displacement of each stroke
    mean_displacement = []
    for stroke in range(len(strokes)):
        mean_displacement.append(np.mean(displacement_per_stroke[stroke][:, i]))
    Q1 = np.percentile(mean_displacement, 25)
    Q3 = np.percentile(mean_displacement, 75)
    IQR = Q3 - Q1

    # Find uper and lower limits
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # Remove the outliers
    index_upper = np.where(mean_displacement >= upper_limit)[0]
    index_lower = np.where(mean_displacement <= lower_limit)[0]
    
    strokes_filtered_temp = []
    displacement_per_stroke_filtered_temp = []
    for index in range(len(strokes)):
        if index not in index_upper:
            if index not in index_lower:
                strokes_filtered_temp.append(strokes[index])
                displacement_per_stroke_filtered_temp.append(displacement_per_stroke[index])
    
   # --------------- Filter for maximum of displacement --------------- 
    max_displacement = []
    for stroke in range(len(strokes_filtered_temp)):
        max_displacement.append(np.max(np.abs(displacement_per_stroke_filtered_temp[stroke][:, i])))
    Q1 = np.percentile(max_displacement, 25)
    Q3 = np.percentile(max_displacement, 75)
    IQR = Q3 - Q1

    # Find uper and lower limits
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # Remove the outliers
    index_upper = np.where(max_displacement >= upper_limit)[0]
    index_lower = np.where(max_displacement <= lower_limit)[0]
    
    strokes_filtered_tempp = []
    displacement_per_stroke_filtered_tempp = []
    for index in range(len(strokes_filtered_temp)):
        if index not in index_upper:
            if index not in index_lower:
                strokes_filtered_tempp.append(strokes_filtered_temp[index])
                displacement_per_stroke_filtered_tempp.append(displacement_per_stroke_filtered_temp[index])
    
    # --------------- Filter for length of list of displacement --------------- 
    displacement = []
    for stroke in range(len(strokes_filtered_tempp)):
        displacement.append(len(displacement_per_stroke_filtered_tempp[stroke][:, i]))
    Q1 = np.percentile(displacement, 25)
    Q3 = np.percentile(displacement, 75)
    IQR = Q3 - Q1

    # Find uper and lower limits
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # Remove the outliers
    index_upper = np.where(displacement >= upper_limit)[0]
    index_lower = np.where(displacement <= lower_limit)[0]
    
    strokes_filtered = []
    displacement_per_stroke_filtered = []
    for index in range(len(strokes_filtered_tempp)):
        if index not in index_upper:
            if index not in index_lower:
                strokes_filtered.append(strokes_filtered_tempp[index])
                displacement_per_stroke_filtered.append(displacement_per_stroke_filtered_tempp[index])

    return strokes_filtered, displacement_per_stroke_filtered


def calculate_ACC(angles_x, angles_y, strokes):
    """ Calculate the angular component of the coefficient 
    of correspondence (ACC).

    :arg angles_x: interpolated array of angles.
    :arg angles_y: interpolated array of angles.
    :arg frames: number of frames interpolated into.
    :returns: the angular component of the coefficient 
    of correspondence (ACC) and standard deviation.
    """

    # Interpolate angles
    angles_x = interpolate_angle_cycle_to_100(angles_x)
    angles_y = interpolate_angle_cycle_to_100(angles_y)

    # Define variables
    frames_nr = len(angles_x[0])  # 100
    strokes_nr = len(strokes)
    cos_phi = np.empty((strokes_nr, frames_nr))
    sin_phi = np.empty((strokes_nr, frames_nr))

    for stroke in range(strokes_nr):
        for frame in range(frames_nr-1):
            difference_1 = angles_x[stroke][frame] - angles_x[stroke][frame+1]
            difference_2 = angles_y[stroke][frame] - angles_y[stroke][frame+1]
            line_segment = np.sqrt(difference_1**2 + difference_2**2)
            cos_phi[stroke, frame] = difference_1 / line_segment
            sin_phi[stroke, frame] = difference_2 / line_segment

    # Define variable
    mean_vector = []
    for frame in range(frames_nr-1):
        mean_cos_phi = np.mean(cos_phi[:, frame])
        mean_sin_phi = np.mean(sin_phi[:, frame])
        mean_vector.append(np.sqrt(mean_cos_phi**2 + mean_sin_phi**2))

    ACC = np.mean(mean_vector)
    ACC_std = np.std(mean_vector)

    return ACC, ACC_std


def calculate_SSD(angles_x1, angles_y1, angles_x2, angles_y2):
    """ This function calculates the sum of squared distances between a shape 
    defined by two vectors (angles_x1, angles_y1) and a comparison shape 
    (healthy control), also defined by two vectors (angles_x2, angles_y2),
    which is a measure of shape difference between two cyclograms.

    :arg angles_x1: interpolated array of angles.
    :arg angles_y1: interpolated array of angles.
    :arg angles_x2: interpolated array of angles for comparison.
    :arg angles_y2: interpolated array of angles for comparison.
    :returns: the sum of squared distances between two cyclogramms (SSD).
    """

    # Average all angles
    angles_x1_mean = np.mean(angles_x1)
    angles_y1_mean = np.mean(angles_y1)
    angles_x2_mean = np.mean(angles_x2)
    angles_y2_mean = np.mean(angles_y2)
    
    # Center each angle individually by subtracting the mean from each point (center at the origin)
    angles_x1_centered = angles_x1 - angles_x1_mean
    angles_y1_centered = angles_y1 - angles_y1_mean
    angles_x2_centered = angles_x2 - angles_x2_mean
    angles_y2_centered = angles_y2 - angles_y2_mean
    
    # Calculate the scaling factors
    aux1 = np.sum(angles_x1_centered**2 + angles_y1_centered**2)
    S1 = np.sqrt(aux1/len(angles_x1_centered))
    aux2 = np.sum(angles_x2_centered**2 + angles_y2_centered**2)
    S2 = np.sqrt(aux2/len(angles_x2_centered))
    
    # Rescale the angles
    angles_x1_rescaled = angles_x1_centered/S1
    angles_y1_rescaled = angles_y1_centered/S1
    angles_x2_rescaled = angles_x2_centered/S2
    angles_y2_rescaled = angles_y2_centered/S2

    # Compute SSD
    squared_distances = []
    for n in range(len(angles_x1_rescaled)):
        squared_distances.append((angles_x1_rescaled[n] - angles_x2_rescaled[n])**2 
                                 + (angles_y1_rescaled[n] - angles_y2_rescaled[n])**2)
    SSD = np.sqrt(np.sum(squared_distances))

    return SSD


def calculate_mean_angles(angles):

    # Number of strokes
    stroke_nr = len(angles)

    # Number of angles in each stroke
    angle_nr = [len(angles[stroke]) for stroke in range(stroke_nr)]

    # Calculate mean angles
    mean_angles = []
    for nr in angle_nr:
        for angle in range(nr):
            mean_angles.append(np.mean([angles[stroke][angle] for stroke in range(stroke_nr)]))

    # Only one stroke necessary
    mean_angles = mean_angles[:100]

    return mean_angles


def calculate_mean_shape_centered(x, y):
    """
    Calculate the mean shape of multiple shapes given by lists of lists of x and y coordinates.
    
    Parameters:
    :arg x: A list of lists, where each inner list contains the x coordinates of a shape.
    :arg y: A list of lists, where each inner list contains the y coordinates of a shape.
    
    Returns:
    :returns: mean_shape_x, mean_shape_y: The x and y coordinates of the mean shape.
    """
    num_shapes = len(x)
    num_points = len(x[0])
    
    # Initialize lists to store the centered coordinates
    centered_x = [[0] * num_points for _ in range(num_shapes)]
    centered_y = [[0] * num_points for _ in range(num_shapes)]
    
    # Calculate centroids and center each shape
    for i in range(num_shapes):
        centroid_x = sum(x[i]) / num_points
        centroid_y = sum(y[i]) / num_points
        centered_x[i] = [x_point - centroid_x for x_point in x[i]]
        centered_y[i] = [y_point - centroid_y for y_point in y[i]]
    
    # Calculate the mean shape
    mean_shape_x = [sum(points) / num_shapes for points in zip(*centered_x)]
    mean_shape_y = [sum(points) / num_shapes for points in zip(*centered_y)]
    
    return mean_shape_x, mean_shape_y


def calculate_stroke_overlap(stroke_L, stroke_R):
    """Calculate the amount of overlap between two strokes."""
    overlap = max(0, min(stroke_L[1], stroke_R[1]) - max(stroke_L[0], stroke_R[0]))
    duration_stroke_L = stroke_L[1] - stroke_L[0]

    # At least half the stroke must overlap
    if overlap > (duration_stroke_L / 2):
        return overlap
    else:
        return 0


def select_corresponding_stroke(left_stroke, right_strokes):
    """Find the right stroke that overlaps the most with the left stroke,
    i.e. the corresponding stroke."""
    
    max_overlap = 0
    corresponding_right_stroke = None

    # Most overlapping stroke on the other side
    for right_stroke in right_strokes:
        overlap = calculate_stroke_overlap(left_stroke, right_stroke)
        if overlap > max_overlap:
            max_overlap = overlap
            corresponding_right_stroke = right_stroke
    return corresponding_right_stroke


def calculate_mean_shape_centered_with_std(x, y):
    """
    Calculate the mean shape and standard deviation of multiple shapes 
    given by lists of lists of x and y coordinates.
    
    Parameters:
    :arg x: A list of lists, where each inner list contains the x coordinates of a shape.
    :arg y: A list of lists, where each inner list contains the y coordinates of a shape.
    
    Returns:
    :returns: mean_shape_x, mean_shape_y: The x and y coordinates of the mean shape.
    :returns: std_dev_x, std_dev_y: The x and y coordinates of the standard deviation.
    """
    num_shapes = len(x)
    num_points = len(x[0])
    
    # Initialize lists to store the centered coordinates
    centered_x = [[0] * num_points for _ in range(num_shapes)]
    centered_y = [[0] * num_points for _ in range(num_shapes)]
    
    # Calculate centroids and center each shape
    for i in range(num_shapes):
        centroid_x = sum(x[i]) / num_points
        centroid_y = sum(y[i]) / num_points
        centered_x[i] = [x_point - centroid_x for x_point in x[i]]
        centered_y[i] = [y_point - centroid_y for y_point in y[i]]
    
    # Calculate the mean shape
    mean_shape_x = [sum(points) / num_shapes for points in zip(*centered_x)]
    mean_shape_y = [sum(points) / num_shapes for points in zip(*centered_y)]
    
    # Calculate variance for each point in the mean shape
    var_x = [sum([(point_x - mean_x)**2 for point_x in points_x]) / num_shapes for points_x, mean_x in zip(zip(*centered_x), mean_shape_x)]
    var_y = [sum([(point_y - mean_y)**2 for point_y in points_y]) / num_shapes for points_y, mean_y in zip(zip(*centered_y), mean_shape_y)]
    
    # Calculate the standard deviation for each point in the mean shape
    std_dev_x = [math.sqrt(var) for var in var_x]
    std_dev_y = [math.sqrt(var) for var in var_y]
    
    return mean_shape_x, mean_shape_y, std_dev_x, std_dev_y


def center_angles(angles_1, angles_2):
    """Plot hip and knee angles.
    
    :arg angles_1: list of first joint angles
    :arg angles_2: list of second joint angles
    :returns: mean joint angles, centered mean joint angles
    """

    # Interpolate angle cycle
    angles_2 = interpolate_angle_cycle_to_100(angles_2)
    angles_1 = interpolate_angle_cycle_to_100(angles_1)

    # Filter noise
    b, a = butter(3, 0.1, 'lowpass')
    angles_2 = filtfilt(b, a, angles_2)
    angles_1 = filtfilt(b, a, angles_1)

    stroke_nr = len(angles_2)

    # Correct angles to end where they start 
    for stroke in range(stroke_nr):
        angle_corr_knee = np.linspace(0, 1, len(angles_2[stroke])).T * (
                (angles_2[stroke][-1] - angles_2[stroke][0]))
        angle_corr_hip = np.linspace(0, 1, len(angles_1[stroke])).T * (
                (angles_1[stroke][-1] - angles_1[stroke][0]))
    
        angles_2[stroke] = angles_2[stroke] - angle_corr_knee
        angles_1[stroke] = angles_1[stroke] - angle_corr_hip

    # Center shapes at origing
    mean_angles_centered_1, mean_angles_centered_2 = calculate_mean_shape_centered(angles_1, angles_2)

    mean_angles_2 = calculate_mean_angles(angles_2)
    mean_angles_1 = calculate_mean_angles(angles_1)

    return mean_angles_centered_1, mean_angles_centered_2, mean_angles_1, mean_angles_2
