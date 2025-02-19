"""
Script: joint_angles.py

Description:
    This script visualizes the joint angles.

Usage:
    - This script is not meant to be run directly - the function is imported 
      into the main data analysis pipeline.

Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from config import *

def analyze_joint_angles():

    # Load HC angle data
    csv_file = 'Joint_Angles_Healthy_Controls.csv'
    csv_file_path = os.path.join(results_path, csv_file)
    healthy_reference = pd.read_csv(csv_file_path)

    # Load SCI angle data
    csv_file_path = os.path.join(results_path, 'Joint_Angles_SCI_Baseline.csv')
    BL_data = pd.read_csv(csv_file_path)

    # Path for plots
    plot_folder = 'Angle_Plots'
    plot_path = os.path.join(figures_path, plot_folder)

    # Create the directory if it doesn't exist
    os.makedirs(plot_path, exist_ok=True)

    # Clinical data
    csv_file_path_sec = os.path.join(data_path, 'Participant_Data/SCI_Baseline/Clinical_Data.csv')
    clinical_data = pd.read_csv(csv_file_path_sec)

    # Make sure abduction is positive
    def adjust_hip_add_abd_angles(row):
        if row['Side'] == 'Right':
            return row.iloc[4:] * (-1)
        else:
            return row.iloc[4:]

    # Apply the adjustment
    for index, row in BL_data.iterrows():
        if row['Joint'] == 'Hip_add_abd':
            BL_data.iloc[index, 4:] = adjust_hip_add_abd_angles(row)

    # Apply the adjustment
    for index, row in healthy_reference.iterrows():
        if row['Joint'] == 'Hip_add_abd':
            healthy_reference.iloc[index, 4:] = adjust_hip_add_abd_angles(row)

    # Separate by joint
    hip_reference = healthy_reference[healthy_reference['Joint'] == 'Hip']
    knee_reference = healthy_reference[healthy_reference['Joint'] == 'Knee']
    ankle_reference = healthy_reference[healthy_reference['Joint'] == 'Ankle']
    hip_add_abd_reference = healthy_reference[healthy_reference['Joint'] == 'Hip_add_abd']

    # Hip
    hip_reference = hip_reference.groupby('Participant').mean()
    hip_reference.columns = [i+1 for i in range(100)]

    # Knee
    knee_reference = knee_reference.groupby('Participant').mean()
    knee_reference.columns = [i+1 for i in range(100)]

    # Ankle
    ankle_reference = ankle_reference.groupby('Participant').mean()
    ankle_reference.columns = [i+1 for i in range(100)]

    # Hip adduction/abduction
    hip_add_abd_reference = hip_add_abd_reference.groupby('Participant').mean()
    hip_add_abd_reference.columns = [i+1 for i in range(100)]

    # Adjusted to medical angles
    def adjust_angles(row):
        if row['Joint'] == 'Hip' or row['Joint'] == 'Knee':
            return (row.iloc[4:] - 180) * (-1)
        if row['Joint'] == 'Hip_add_abd':
            return row.iloc[4:]
        elif row['Joint'] == 'Ankle':
            return row.iloc[4:] * (-1)

    # Apply the adjustment
    for index, row in BL_data.iterrows():
        BL_data.iloc[index, 4:] = adjust_angles(row)

    # Apply the adjustment
    for index, row in healthy_reference.iterrows():
        healthy_reference.iloc[index, 4:] = adjust_angles(row)
        
    # Spetarate joints
    hip_reference = healthy_reference[healthy_reference['Joint'] == 'Hip']
    knee_reference = healthy_reference[healthy_reference['Joint'] == 'Knee']
    ankle_reference = healthy_reference[healthy_reference['Joint'] == 'Ankle']
    hip_add_abd_reference = healthy_reference[healthy_reference['Joint'] == 'Hip_add_abd']

    # Hip
    hip_reference = hip_reference.groupby('Participant').mean()
    hip_reference.columns = [i+1 for i in range(100)]

    # Knee
    knee_reference = knee_reference.groupby('Participant').mean()
    knee_reference.columns = [i+1 for i in range(100)]

    # Ankle
    ankle_reference = ankle_reference.groupby('Participant').mean()
    ankle_reference.columns = [i+1 for i in range(100)]

    # Hip adduction/abduction
    hip_add_abd_reference = hip_add_abd_reference.groupby('Participant').mean()
    hip_add_abd_reference.columns = [i+1 for i in range(100)]



    ### Plot averages ###


    # Get more impaired leg
    clinical_data['Participant'] = clinical_data['ID']
    BL_data = pd.merge(BL_data, clinical_data[['Participant', 'More impaired leg']], on='Participant')

    # Function to determine leg type
    def determine_leg_type(row):
        if row['Side'] == 'Left':
            return 'More impaired' if row['More impaired leg'] == 'L' else 'Less impaired'
        else:  # row['Side'] == 'Right'
            return 'More impaired' if row['More impaired leg'] == 'R' else 'Less impaired'

    # Apply the function to create the 'Leg type' column
    BL_data['Leg type'] = BL_data.apply(determine_leg_type, axis=1)

    # Spetarate joints
    hip_SCI = BL_data[BL_data['Joint'] == 'Hip']
    hip_SCI_more_impaired = hip_SCI[hip_SCI['Leg type'] == 'More impaired']
    hip_SCI_less_impaired = hip_SCI[hip_SCI['Leg type'] == 'Less impaired']
    knee_SCI = BL_data[BL_data['Joint'] == 'Knee']
    knee_SCI_more_impaired = knee_SCI[knee_SCI['Leg type'] == 'More impaired']
    knee_SCI_less_impaired = knee_SCI[knee_SCI['Leg type'] == 'Less impaired']
    ankle_SCI = BL_data[BL_data['Joint'] == 'Ankle']
    ankle_SCI_more_impaired = ankle_SCI[ankle_SCI['Leg type'] == 'More impaired']
    ankle_SCI_less_impaired = ankle_SCI[ankle_SCI['Leg type'] == 'Less impaired']
    hip_add_abd_SCI = BL_data[BL_data['Joint'] == 'Hip_add_abd']
    hip_add_abd_SCI_more_impaired = hip_add_abd_SCI[hip_add_abd_SCI['Leg type'] == 'More impaired']
    hip_add_abd_SCI_less_impaired = hip_add_abd_SCI[hip_add_abd_SCI['Leg type'] == 'Less impaired']


    # Ankle more impaires
    ankle_SCI_more_impaired = ankle_SCI_more_impaired.groupby('Participant').mean()
    ankle_SCI_more_impaired.columns = [i+1 for i in range(100)]
    ankle_SCI_more_impaired_mean = ankle_SCI_more_impaired.mean()
    ankle_SCI_more_impaired_std = ankle_SCI_more_impaired.std()

    # Ankle less impaires
    ankle_SCI_less_impaired = ankle_SCI_less_impaired.groupby('Participant').mean()
    ankle_SCI_less_impaired.columns = [i+1 for i in range(100)]
    ankle_SCI_less_impaired_mean = ankle_SCI_less_impaired.mean()
    ankle_SCI_less_impaired_std = ankle_SCI_less_impaired.std()

    # Knee more impaires
    knee_SCI_more_impaired = knee_SCI_more_impaired.groupby('Participant').mean()
    knee_SCI_more_impaired.columns = [i+1 for i in range(100)]
    knee_SCI_more_impaired_mean = knee_SCI_more_impaired.mean()
    knee_SCI_more_impaired_std = knee_SCI_more_impaired.std()

    # Knee less impaires
    knee_SCI_less_impaired = knee_SCI_less_impaired.groupby('Participant').mean()
    knee_SCI_less_impaired.columns = [i+1 for i in range(100)]
    knee_SCI_less_impaired_mean = knee_SCI_less_impaired.mean()
    knee_SCI_less_impaired_std = knee_SCI_less_impaired.std()

    # Hip more impaired
    hip_SCI_more_impaired = hip_SCI_more_impaired.groupby('Participant').mean()
    hip_SCI_more_impaired.columns = [i+1 for i in range(100)]
    hip_SCI_more_impaired_mean = hip_SCI_more_impaired.mean()
    hip_SCI_more_impaired_std = hip_SCI_more_impaired.std()

    # Hip less impaired
    hip_SCI_less_impaired = hip_SCI_less_impaired.groupby('Participant').mean()
    hip_SCI_less_impaired.columns = [i+1 for i in range(100)]
    hip_SCI_less_impaired_mean = hip_SCI_less_impaired.mean()
    hip_SCI_less_impaired_std = hip_SCI_less_impaired.std()

    # Hip adduction/abduction more impaires
    hip_add_abd_SCI_more_impaired = hip_add_abd_SCI_more_impaired.groupby(['Participant', 'Side']).mean()
    hip_add_abd_SCI_more_impaired.columns = [i+1 for i in range(100)]
    hip_add_abd_SCI_more_impaired_mean = hip_add_abd_SCI_more_impaired.mean()
    hip_add_abd_SCI_more_impaired_std = hip_add_abd_SCI_more_impaired.std()

    # Hip adduction/abduction less impaires
    hip_add_abd_SCI_less_impaired = hip_add_abd_SCI_less_impaired.groupby(['Participant', 'Side']).mean()
    hip_add_abd_SCI_less_impaired.columns = [i+1 for i in range(100)]
    hip_add_abd_SCI_less_impaired_mean = hip_add_abd_SCI_less_impaired.mean()
    hip_add_abd_SCI_less_impaired_std = hip_add_abd_SCI_less_impaired.std()


    ### Plot average joint angle trajectory per stoke

    # Separate HC max speed and comfort speed
    file_patch_hc_swimming_style = os.path.join(data_path, 'Participant_Data/Healthy_Controls/Swimming_Style_Healthy_Controls.csv')
    hc_swimming_style = pd.read_csv(file_patch_hc_swimming_style)

    # Convert trial numbers to floats
    healthy_reference['Trial'] = healthy_reference['Trial'].str[-1].astype(float)

    # Spetarate joints
    ankle_reference = healthy_reference[healthy_reference['Joint'] == 'Ankle']
    knee_reference = healthy_reference[healthy_reference['Joint'] == 'Knee']
    hip_reference = healthy_reference[healthy_reference['Joint'] == 'Hip']
    hip_add_abd_reference = healthy_reference[healthy_reference['Joint'] == 'Hip_add_abd']


    # Separate max speed and comfort speed for HC
    merged_df = pd.merge(ankle_reference, hc_swimming_style, on=['Participant', 'Trial'])
    ankle_reference_maxspeed = merged_df[merged_df['Style'] == 'MaxSpeed']
    ankle_reference_comfortspeed = merged_df[merged_df['Style'] == 'ComfortSpeed']
    ankle_reference_maxspeed = ankle_reference_maxspeed.drop(columns=['Style', 'Trial'])
    ankle_reference_comfortspeed = ankle_reference_comfortspeed.drop(columns=['Style', 'Trial'])

    # Ankle max speed
    ankle_reference_maxspeed = ankle_reference_maxspeed.groupby('Participant').mean()
    ankle_reference_maxspeed.columns = [i+1 for i in range(100)]
    ankle_reference_maxspeed_mean = ankle_reference_maxspeed.mean()
    ankle_reference_maxspeed_std = ankle_reference_maxspeed.std()

    # Ankle comfort speed
    ankle_reference_comfortspeed = ankle_reference_comfortspeed.groupby('Participant').mean()
    ankle_reference_comfortspeed.columns = [i+1 for i in range(100)]


    # Separate max speed and comfort speed for HC
    merged_df = pd.merge(knee_reference, hc_swimming_style, on=['Participant', 'Trial'])
    knee_reference_maxspeed = merged_df[merged_df['Style'] == 'MaxSpeed']
    knee_reference_comfortspeed = merged_df[merged_df['Style'] == 'ComfortSpeed']
    knee_reference_maxspeed = knee_reference_maxspeed.drop(columns=['Style', 'Trial'])
    knee_reference_comfortspeed = knee_reference_comfortspeed.drop(columns=['Style', 'Trial'])

    # Knee max speed
    knee_reference_maxspeed = knee_reference_maxspeed.groupby('Participant').mean()
    knee_reference_maxspeed.columns = [i+1 for i in range(100)]
    knee_reference_maxspeed_mean = knee_reference_maxspeed.mean()
    knee_reference_maxspeed_std = knee_reference_maxspeed.std()

    # Knee comfort speed
    knee_reference_comfortspeed = knee_reference_comfortspeed.groupby('Participant').mean()
    knee_reference_comfortspeed.columns = [i+1 for i in range(100)]


    # Separate max speed and comfort speed for HC
    merged_df = pd.merge(hip_reference, hc_swimming_style, on=['Participant', 'Trial'])
    hip_reference_maxspeed = merged_df[merged_df['Style'] == 'MaxSpeed']
    hip_reference_comfortspeed = merged_df[merged_df['Style'] == 'ComfortSpeed']
    hip_reference_maxspeed = hip_reference_maxspeed.drop(columns=['Style', 'Trial'])
    hip_reference_comfortspeed = hip_reference_comfortspeed.drop(columns=['Style', 'Trial'])

    # Hip max speed
    hip_reference_maxspeed = hip_reference_maxspeed.groupby('Participant').mean()
    hip_reference_maxspeed.columns = [i+1 for i in range(100)]
    hip_reference_maxspeed_mean = hip_reference_maxspeed.mean()
    hip_reference_maxspeed_std = hip_reference_maxspeed.std()

    # Hip comfort speed
    hip_reference_comfortspeed = hip_reference_comfortspeed.groupby('Participant').mean()
    hip_reference_comfortspeed.columns = [i+1 for i in range(100)]


    # Separate max speed and comfort speed for HC
    merged_df = pd.merge(hip_add_abd_reference, hc_swimming_style, on=['Participant', 'Trial'])
    hip_add_abd_reference_maxspeed = merged_df[merged_df['Style'] == 'MaxSpeed']
    hip_add_abd_reference_comfortspeed = merged_df[merged_df['Style'] == 'ComfortSpeed']
    hip_add_abd_reference_maxspeed = hip_add_abd_reference_maxspeed.drop(columns=['Style', 'Trial'])
    hip_add_abd_reference_comfortspeed = hip_add_abd_reference_comfortspeed.drop(columns=['Style', 'Trial'])

    # Hip add/abd max speed
    hip_add_abd_reference_maxspeed = hip_add_abd_reference_maxspeed.groupby('Participant').mean()
    hip_add_abd_reference_maxspeed.columns = [i+1 for i in range(100)]
    hip_add_abd_reference_maxspeed_mean = hip_add_abd_reference_maxspeed.mean()
    hip_add_abd_reference_maxspeed_std = hip_add_abd_reference_maxspeed.std()

    # Hip add/abd comfort speed
    hip_add_abd_reference_comfortspeed = hip_add_abd_reference_comfortspeed.groupby('Participant').mean()
    hip_add_abd_reference_comfortspeed.columns = [i+1 for i in range(100)]


    # Ankle
    plt.figure(figsize=(10, 6))

    plt.plot(range(100), ankle_reference_maxspeed_mean, 'dimgray', label='healthy controls')
    plt.fill_between(range(100), ankle_reference_maxspeed_mean + ankle_reference_maxspeed_std, 
                    ankle_reference_maxspeed_mean - ankle_reference_maxspeed_std, color='gray', alpha=0.1)

    plt.plot(range(100), ankle_SCI_less_impaired_mean, 'cornflowerblue', label='less impaired leg')
    plt.fill_between(range(100), ankle_SCI_less_impaired_mean + ankle_SCI_less_impaired_std, 
                    ankle_SCI_less_impaired_mean - ankle_SCI_less_impaired_std, color='cornflowerblue', alpha=0.1)

    plt.plot(range(100), ankle_SCI_more_impaired_mean, 'crimson', label='more impaired leg')
    plt.fill_between(range(100), ankle_SCI_more_impaired_mean + ankle_SCI_more_impaired_std, 
                    ankle_SCI_more_impaired_mean - ankle_SCI_more_impaired_std, color='crimson', alpha=0.05)

    plt.xlabel('Stroke cycle [%]', size=24)
    plt.ylabel('Ankle dorsiflexion [deg]', size=24)
    plt.xticks(size=20)
    plt.yticks(size=20)

    # Remove the top and left spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add arrow
    plt.annotate('', xy=(0.19, 0.8), xytext=(0.19, 0.4), 
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                xycoords='axes fraction')

    # Add vertical text
    plt.text(0.12, 0.6, 'dorsiflexion', rotation=90, va='center', ha='left', 
            fontsize=22, color='black', transform=plt.gca().transAxes)
    plt.tight_layout()

    plt.savefig(plot_path + '/Ankle.png', dpi=300)
    plt.close()


    # Knee
    plt.figure(figsize=(10, 6))

    plt.plot(range(100), knee_reference_maxspeed_mean, 'dimgray', label='healthy controls')
    plt.fill_between(range(100), knee_reference_maxspeed_mean + knee_reference_maxspeed_std, 
                    knee_reference_maxspeed_mean - knee_reference_maxspeed_std, color='gray', alpha=0.1)

    plt.plot(range(100), knee_SCI_less_impaired_mean, 'cornflowerblue', label='less impaired leg')
    plt.fill_between(range(100), knee_SCI_less_impaired_mean + knee_SCI_less_impaired_std, 
                    knee_SCI_less_impaired_mean - knee_SCI_less_impaired_std, color='cornflowerblue', alpha=0.1)

    plt.plot(range(100), knee_SCI_more_impaired_mean, 'crimson', label='more impaired leg')
    plt.fill_between(range(100), knee_SCI_more_impaired_mean + knee_SCI_more_impaired_std, 
                    knee_SCI_more_impaired_mean - knee_SCI_more_impaired_std, color='crimson', alpha=0.05)

    plt.xlabel('Stroke cycle [%]', size=24)
    plt.ylabel('Knee flexion [deg]', size=24)
    plt.xticks(size=20)
    plt.yticks(size=20)

    # Remove the top and left spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add arrow
    plt.annotate('', xy=(0.19, 0.8), xytext=(0.19, 0.4), 
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                xycoords='axes fraction')

    # Add vertical text
    plt.text(0.12, 0.6, 'flexion', rotation=90, va='center', ha='left', 
            fontsize=22, color='black', transform=plt.gca().transAxes)
    plt.tight_layout()

    plt.savefig(plot_path + '/Knee.png', dpi=300)
    plt.close()


    # Hip
    plt.figure(figsize=(10, 6))

    plt.plot(range(100), hip_reference_maxspeed_mean, 'dimgray', label='healthy controls')
    plt.fill_between(range(100), hip_reference_maxspeed_mean + hip_reference_maxspeed_std, 
                    hip_reference_maxspeed_mean - hip_reference_maxspeed_std, color='gray', alpha=0.1)

    plt.plot(range(100), hip_SCI_less_impaired_mean, 'cornflowerblue', label='less impaired leg')
    plt.fill_between(range(100), hip_SCI_less_impaired_mean + hip_SCI_less_impaired_std, 
                    hip_SCI_less_impaired_mean - hip_SCI_less_impaired_std, color='cornflowerblue', alpha=0.1)

    plt.plot(range(100), hip_SCI_more_impaired_mean, 'crimson', label='more impaired leg')
    plt.fill_between(range(100), hip_SCI_more_impaired_mean + hip_SCI_more_impaired_std, 
                    hip_SCI_more_impaired_mean - hip_SCI_more_impaired_std, color='crimson', alpha=0.05)

    # plt.title('hip flexion/extension', fontsize=16)
    plt.xlabel('Stroke cycle [%]', size=24)
    plt.ylabel('Hip flexion [deg]', size=24)
    plt.xticks(size=20)
    plt.yticks(size=20)

    # Remove the top and left spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add arrow
    plt.annotate('', xy=(0.19, 0.8), xytext=(0.19, 0.4),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                xycoords='axes fraction')

    # Add vertical text
    plt.text(0.12, 0.6, 'flexion', rotation=90, va='center', ha='left', 
            fontsize=22, color='black', transform=plt.gca().transAxes)
    plt.tight_layout()

    plt.savefig(plot_path + '/Hip.png', dpi=300)
    plt.close()


    # Hip add/abd
    plt.figure(figsize=(10, 6))

    plt.plot(range(100), hip_add_abd_reference_maxspeed_mean, 'dimgray', label='healthy controls')
    plt.fill_between(range(100), hip_add_abd_reference_maxspeed_mean + hip_add_abd_reference_maxspeed_std, 
                    hip_add_abd_reference_maxspeed_mean - hip_add_abd_reference_maxspeed_std, color='gray', alpha=0.1)

    plt.plot(range(100), hip_add_abd_SCI_less_impaired_mean, 'cornflowerblue', label='less impaired leg')
    plt.fill_between(range(100), hip_add_abd_SCI_less_impaired_mean + hip_add_abd_SCI_less_impaired_std, 
                    hip_add_abd_SCI_less_impaired_mean - hip_add_abd_SCI_less_impaired_std, color='cornflowerblue', alpha=0.2)

    plt.plot(range(100), hip_add_abd_SCI_more_impaired_mean, 'crimson', label='more impaired leg')
    plt.fill_between(range(100), hip_add_abd_SCI_more_impaired_mean + hip_add_abd_SCI_more_impaired_std, 
                    hip_add_abd_SCI_more_impaired_mean - hip_add_abd_SCI_more_impaired_std, color='crimson', alpha=0.05)

    # plt.title('hip adduction/abduction', fontsize=16)
    plt.xlabel('Stroke cycle [%]', size=24)
    plt.ylabel('Hip abduction [deg]', size=24)
    plt.xticks(size=20)
    plt.yticks(size=20)

    # Remove the top and left spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add arrow
    plt.annotate('', xy=(0.19, 0.8), xytext=(0.19, 0.4), 
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                xycoords='axes fraction')

    # Add vertical text
    plt.text(0.12, 0.6, 'abduction', rotation=90, va='center', ha='left', 
            fontsize=22, color='black', transform=plt.gca().transAxes)
    plt.tight_layout()

    plt.savefig(plot_path + '/Hip_add_abd.png', dpi=300)
    plt.close()


    # Hip-knee plot
    plt.figure(figsize=(10, 6))
    plt.plot(knee_reference_maxspeed_mean, hip_reference_maxspeed_mean, 'dimgray', label='healthy controls')
    plt.plot(knee_SCI_less_impaired_mean, hip_SCI_less_impaired_mean, 'cornflowerblue', label='less impaired leg')
    plt.plot(knee_SCI_more_impaired_mean, hip_SCI_more_impaired_mean, 'crimson', label='more impaired leg')
    plt.xlabel('Knee flexion [deg]', size=24)
    plt.ylabel('Hip flexion [deg]', size=24)
    plt.xticks(size=20)
    plt.yticks(ticks=[10, 20, 30, 40, 50], labels=[10, 20, 30, 40, 50], size=20)

    # Remove the top and left spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_path + '/Hip_knee.png', dpi=300)
    plt.close()


    # Ankle-knee plot
    plt.figure(figsize=(10, 6))
    plt.plot(ankle_reference_maxspeed_mean, knee_reference_maxspeed_mean, 'dimgray', label='healthy controls')
    plt.plot(ankle_SCI_less_impaired_mean, knee_SCI_less_impaired_mean, 'cornflowerblue', label='less impaired leg')
    plt.plot(ankle_SCI_more_impaired_mean, knee_SCI_more_impaired_mean, 'crimson', label='more impaired leg')
    plt.xlabel('Ankle dorsiflexion [deg]', size=24)
    plt.ylabel('Knee flexion [deg]', size=24)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(loc='upper left', fontsize=20)

    # Remove the top and left spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_path + '/Ankle_knee.png', dpi=300)
    plt.close()
