"""
Script: swimming_parameters.py

Description:
    This script performs a statistical analysis of swimming parameters.

Usage:
    - This script is not meant to be run directly - the function is imported 
      into the main data analysis pipeline.

Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import pandas as pd
from visualizations import *
from config import data_path, results_path


def analyze_swimming_parameters():

    # Define the paths for the directories
    hc_dir = os.path.join(figures_path, "Healthy_Controls")
    sci_baseline_dir = os.path.join(figures_path, "SCI_Baseline")

    # Create the directories if they don't already exist
    os.makedirs(hc_dir, exist_ok=True)
    os.makedirs(sci_baseline_dir, exist_ok=True)

    # Load swimming parameters
    file_path_bl = os.path.join(results_path, 'Swimming_Parameters_SCI_Baseline.csv')
    file_path_hc = os.path.join(results_path, 'Swimming_Parameters_Healthy_Controls.csv')
    df_bl = pd.read_csv(file_path_bl)
    df_hc = pd.read_csv(file_path_hc)

    # Load clinical data
    file_path_clinical_data = os.path.join(data_path, 'Participant_Data/SCI_Baseline/Clinical_Data.csv')
    file_path_clinical_data_HC = os.path.join(data_path, 'Participant_Data/Healthy_Controls/Clinical_Data.csv')
    df_clinical = pd.read_csv(file_path_clinical_data)
    df_clinical_HC = pd.read_csv(file_path_clinical_data_HC)

    # Load swimming style (max and comfort speed) for HCs
    file_path_hc_swimming_style = os.path.join(data_path, 'Participant_Data/Healthy_Controls/Swimming_Style_Healthy_Controls.csv')
    hc_swimming_style = pd.read_csv(file_path_hc_swimming_style)

    # Calculate velocity and add to data
    df_bl['Velocity'] = 5 / df_bl['Time']
    df_hc['Velocity'] = 5 / df_hc['Time']

    # Move velocity column (for spider plot)
    df_bl = df_bl[['Participant', 'Trial', 'Strokes', 'Time', 'Velocity'] + 
                [col for col in df_bl if col not in ['Participant', 'Trial', 'Strokes', 'Time', 'Velocity']]]
    df_hc = df_hc[['Participant', 'Trial', 'Strokes', 'Time', 'Velocity'] + 
                [col for col in df_hc if col not in ['Participant', 'Trial', 'Strokes', 'Time', 'Velocity']]]

    # Separate max speed and comfort speed for HC
    merged_df = pd.merge(df_hc, hc_swimming_style, on=['Participant', 'Trial'])
    df_hc_maxspeed = merged_df[merged_df['Style'] == 'MaxSpeed']
    df_hc_comfortspeed = merged_df[merged_df['Style'] == 'ComfortSpeed']
    df_hc_maxspeed = df_hc_maxspeed.drop(columns=['Style'])
    df_hc_comfortspeed = df_hc_comfortspeed.drop(columns=['Style'])


    ### Set up patient data and healthy data in the required format for data analysis

    parameters = ['Stroke duration L', 'Stroke duration R', 
                  'Stroke duration variability L', 'Stroke duration variability R', 
                  'Min ankle flexion L', 'Max ankle extension L', 'Min ankle flexion R',
                  'Max ankle extension R', 'Min knee flexion L', 'Max knee extension L',
                  'Min knee flexion R', 'Max knee extension R', 'Min hip flexion L',
                  'Max hip extension L', 'Min hip flexion R', 'Max hip extension R',
                  'Min hip adduction L', 'Max hip abduction L', 'Min hip adduction R', 
                  'Max hip abduction R', 'Horizontal displacement L', 'Horizontal displacement R',
                  'Vertical displacement L', 'Vertical displacement R', 'Sidewards displacement L', 
                  'Sidewards displacement R', 'RoM knee R (flexion/extension)', 'RoM knee L (flexion/extension)',
                  'RoM hip R (flexion/extension)', 'RoM hip L (flexion/extension)',
                  'RoM hip R (abduction/adduction)', 'RoM hip L (abduction/adduction)',
                  'RoM ankle R (flexion/extension)', 'RoM ankle L (flexion/extension)',
                  'ACC L (hip/knee)', 'ACC R (hip/knee)', 'ACC std L (hip/knee)', 
                  'ACC std R (hip/knee)', 'ACC L (ankle/knee)', 'ACC R (ankle/knee)', 
                  'ACC std L (ankle/knee)', 'ACC std R (ankle/knee)', 'SSD L (hip/knee)', 
                  'SSD R (hip/knee)', 'SSD L (ankle/knee)', 'SSD R (ankle/knee)']

    # Patients
    df_bl = distinguish_impaired_leg(df_bl, df_clinical, parameters)

    # Maximum speed HC
    df_hc_maxspeed_grouped = df_hc_maxspeed.groupby('Participant')
    df_hc_maxspeed_means = df_hc_maxspeed_grouped.apply(lambda x: calculate_combined_mean(x, df_hc_maxspeed.columns))
    df_hc_maxspeed_means = df_hc_maxspeed_means.reset_index(drop=True)

    # Comfort speed HC
    df_hc_comfortspeed_grouped = df_hc_comfortspeed.groupby('Participant')
    df_hc_comfortspeed_means = df_hc_comfortspeed_grouped.apply(lambda x: calculate_combined_mean(x, df_hc_maxspeed.columns))
    df_hc_comfortspeed_means = df_hc_comfortspeed_means.reset_index(drop=True)


    # Perform statistical analysis of swimming parameters

    # Add age and BMI to the data sets
    df_clinical['Participant'] = df_clinical['ID']
    df_clinical_HC['Participant'] = df_clinical_HC['ID']
    df_bl = df_bl.merge(df_clinical[['Age', 'BMI', 'Participant']], on='Participant', how='left')
    df_hc_comfortspeed_means = df_hc_comfortspeed_means.merge(df_clinical_HC[['Age', 'BMI', 'Participant']], on='Participant', how='left')
    df_hc_maxspeed_means = df_hc_maxspeed_means.merge(df_clinical_HC[['Age', 'BMI', 'Participant']], on='Participant', how='left')

    # Calculate significances for all parameters
    results_df = pd.DataFrame(columns=['Parameter', 'P-Value'])
    results_df_HC = pd.DataFrame(columns=['Parameter', 'P-Value'])   

    for param in df_hc_comfortspeed_means.columns[3:-2]:
        results_df = calculate_p_values(df_hc_maxspeed_means, df_bl, param, results_df)
        results_df_HC = calculate_p_values_HC(df_hc_comfortspeed_means, df_hc_maxspeed_means, param, results_df_HC)

    # Remove age and BMI from the data sets
    df_bl = df_bl.drop(columns=['Age', 'BMI'])
    df_hc_comfortspeed_means = df_hc_comfortspeed_means.drop(columns=['Age', 'BMI'])
    df_hc_maxspeed_means = df_hc_maxspeed_means.drop(columns=['Age', 'BMI'])

    # For spiderplots
    df_bl.rename(columns={'Phase shift std': 'Phase shift variability'}, inplace=True)
    df_hc_comfortspeed_means.rename(columns={'Phase shift std': 'Phase shift variability'}, inplace=True)
    df_hc_maxspeed_means.rename(columns={'Phase shift std': 'Phase shift variability'}, inplace=True)

    # Multiple testing correction
    results_df_corrected = holm_bonferroni_correction(results_df)
    # significant_parameters = results_df_corrected[results_df_corrected['Reject'] == True]
    # print('---------- PATIENTS ----------')
    # print(significant_parameters)

    results_df_corrected_HC = holm_bonferroni_correction(results_df_HC)
    # significant_parameters_HC = results_df_corrected_HC[results_df_corrected_HC['Reject'] == True]
    # print('---------- HC ----------')
    # print(significant_parameters_HC)


    ########### Create plots of significantly different swimming parameters ###########

    # Extract significant parameters and their adjusted p-values
    significant_params = results_df_corrected.loc[results_df_corrected['Reject'] == True, ['Parameter', 'Adjusted P-Value']]

    # Loop over significant parameters and pass adjusted p-values
    for _, row in significant_params.iterrows():
        param = row['Parameter']
        adj_p_value = row['Adjusted P-Value']
        plot_parameters(df_hc_maxspeed_means, df_bl, param, adj_p_value)

    significant_params_HC = results_df_corrected_HC.loc[results_df_corrected_HC['Reject'] == True, ['Parameter', 'Adjusted P-Value']]

    for _, row in significant_params_HC.iterrows():
        param = row['Parameter']
        adj_p_value = row['Adjusted P-Value']
        plot_parameters_HC(df_hc_comfortspeed_means, df_hc_maxspeed_means, param, adj_p_value)



    # ------------ Create combined blots for proximal/distal parameters HC ------------

    df_results = pd.DataFrame()

    # Compare ACC and SSD of healthy controls at different speeds
    df_results = compare_ACC_SSD(df_hc_comfortspeed_means, df_results, speed='comfort')
    df_results = compare_ACC_SSD(df_hc_maxspeed_means, df_results, speed='max')

    corrected_pvalues = list(df_results['Corrected P-Value'])

    # Healthy contorls
    plot_proximal_distal_HC(df_hc_comfortspeed_means, df_hc_maxspeed_means, corrected_pvalues)

    # SCI
    plot_proximal_distal(df_hc_maxspeed_means, df_bl)

    # Swimming profiles visualized as spider plot
    swimming_profiles(df_bl, df_hc_comfortspeed_means, df_hc_maxspeed_means)
