"""
Script: visualization.py

Description:
    This script contains a collection of functions for data analysis and visualization.

Usage:
    - This script is not meant to be run directly - the functions are imported 
      into the main data analysis pipeline.


Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from spatialmath.base import *
import matplotlib.pyplot as plt
from scipy.stats import kruskal, ttest_ind
from statsmodels.stats.multitest import multipletests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import figures_path


def distinguish_impaired_leg(df_bl, df_clinical, parameters):
    """
    Distinguishes between the more and less impaired leg for each participant
    and restructures the DataFrame accordingly.

    Parameters:
    -----------
    df_bl : pd.DataFrame
        DataFrame containing participant-specific movement parameters.
    df_clinical : pd.DataFrame
        DataFrame containing clinical information, including the more impaired leg for each participant.
    parameters : list
        List of movement parameters with left ('L') and right ('R') leg variations.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with new columns for the more impaired (MI) and less impaired (LI) leg.
        The original left ('L') and right ('R') columns are removed.
    """

    # Initialize column for more impaired leg
    df_bl['More impaired leg'] = np.nan

    # Assign more impaired leg from df_clinical
    for participant in df_bl['Participant'].unique():
        more_impaired_leg = df_clinical.loc[df_clinical['ID'] == participant, 'More impaired leg'].values[0]
        df_bl['More impaired leg'] = df_bl['More impaired leg'].astype(str)
        df_bl.loc[df_bl['Participant'] == participant, 'More impaired leg'] = str(more_impaired_leg)

    # Process each parameter
    for param in parameters:
        if not param.endswith(' L') and not param.endswith(' R'):
            # Handle parameters without explicit L/R in their name
            base_param = " ".join(param.split()[:-2])
            condition = " ".join(param.split()[-1:])
            side = param.split()[-2]

            more_col_name = f"{base_param} {condition} MI"
            less_col_name = f"{base_param} {condition} LI"

            # Assign values based on more impaired leg
            for index, row in df_bl.iterrows():
                if row['More impaired leg'] == side:
                    df_bl.at[index, more_col_name] = row[param]
                    opposite_param = param.replace(f"{side} ", f"{'L' if side == 'R' else 'R'} ")
                    df_bl.at[index, less_col_name] = row[opposite_param]
                else:
                    df_bl.at[index, less_col_name] = row[param]
                    opposite_param = param.replace(f"{side} ", f"{'L' if side == 'R' else 'R'} ")
                    df_bl.at[index, more_col_name] = row[opposite_param]

        else:
            # Handle parameters explicitly ending with 'L' or 'R'
            base_param = param[:-2]  # Remove L/R
            side = param[-1]  # Get the side ('L' or 'R')

            more_col_name = f"{base_param} MI"
            less_col_name = f"{base_param} LI"

            # Assign values based on more impaired leg
            for index, row in df_bl.iterrows():
                if row['More impaired leg'] == side:
                    df_bl.at[index, more_col_name] = row[param]
                    opposite_param = param.replace(f"{side}", f"{'L' if side == 'R' else 'R'}")
                    df_bl.at[index, less_col_name] = row[opposite_param]
                else:
                    df_bl.at[index, less_col_name] = row[param]
                    opposite_param = param.replace(f"{side}", f"{'L' if side == 'R' else 'R'}")
                    df_bl.at[index, more_col_name] = row[opposite_param]

    # Identify all L/R parameters to be dropped
    parameters_side = [col for col in df_bl.columns if col.endswith('L') or col.endswith('R')]

    # Remove original L/R columns
    df_bl = df_bl.drop(columns=parameters_side, errors='ignore')

    return df_bl


def calculate_combined_mean(group, params):
    """
    Computes the combined mean of swimming parameters by averaging left ('L') and right ('R') side values.

    Parameters:
    -----------
    group : pd.DataFrame
        A DataFrame containing movement parameters for a specific group (e.g., a participant or condition).
    params : list
        A list of column names representing movement parameters, including left ('L') and right ('R') variations.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame where:
        - Parameters without 'L' or 'R' remain unchanged.
        - Parameters with 'L' and 'R' are averaged into a single 'Mean' value.

    Notes:
    ------
    - If a parameter has both 'L' (left) and 'R' (right) versions, it is averaged into a new column named '<Parameter> Mean'.
    - If a parameter does not have 'L' or 'R', it is kept as is.
    - The function ensures that each parameter is only averaged once.
    """

    means = {}
    for param in params:
        if ' L' not in param and ' R' not in param:
            means[param] = group[param]
        side = 'L' if 'L' in param else 'R'
        if side in param:
            base_param_name = param.replace(f' {side}', '')
            if base_param_name + ' Mean' not in means:
                means[base_param_name + ' Mean'] = group[param]
            else:
                means[base_param_name + ' Mean'] += group[param]
                means[base_param_name + ' Mean'] /= 2
    return pd.DataFrame(means)


def calculate_p_values(data_controls_max, data_patients, parameter_name, results_df):
    """
    Computes statistical significance of a given parameter between Healthy Controls (HC) 
    and Patients, optionally differentiating between the less impaired (LI) and more impaired (MI) legs.

    Statistical significance is tested using the Kruskal-Wallis test for overall group differences,
    followed by pairwise t-tests if the Kruskal-Wallis test is significant.

    Parameters:
    ----------
    data_controls_max : pd.DataFrame
        DataFrame containing control group data, including the specified parameter.
    data_patients : pd.DataFrame
        DataFrame containing patient data, including the specified parameter and leg-specific values if applicable.
    parameter_name : str
        The name of the parameter to be analyzed.
    results_df : pd.DataFrame
        DataFrame to store p-values from the statistical tests.

    Returns:
    -------
    pd.DataFrame
        Updated results_df with added p-values for the given parameter.

    Notes:
    ------
    - Uses Kruskal-Wallis test for overall group differences.
    - Performs pairwise t-tests between groups if the Kruskal-Wallis test is significant.
    """

    # Determine if we differentiate legs
    if ' Mean' in parameter_name:
        differentiate_legs = True
        parameter_base = parameter_name.replace(' Mean', '')
        less_impaired_param = parameter_base + ' LI'
        more_impaired_param = parameter_base + ' MI'
    else:
        differentiate_legs = False

    # Create a DataFrame for statistical analysis
    stat_data = pd.DataFrame()

    # Calculate mean values for control group
    if not parameter_name.startswith('SSD'):
        controls_data_avg = data_controls_max.groupby('Participant')[[parameter_name]].mean().reset_index()
        stat_data = pd.concat([stat_data, pd.DataFrame({
            'Value': controls_data_avg[parameter_name],
            'Group': 'HC'
        })], ignore_index=True)

    if differentiate_legs:
        # Get the mean for each parameter per participant
        patient_data_avg = data_patients.groupby('Participant').agg({
            less_impaired_param: 'mean',
            more_impaired_param: 'mean',
        }).reset_index()

        stat_data = pd.concat([stat_data, pd.DataFrame({
            'Value': patient_data_avg[less_impaired_param],
            'Group': 'less \nimpaired'
        })], ignore_index=True)
        stat_data = pd.concat([stat_data, pd.DataFrame({
            'Value': patient_data_avg[more_impaired_param],
            'Group': 'more \nimpaired'
        })], ignore_index=True)
    else:
        patient_data_avg = data_patients.groupby('Participant')[[parameter_name]].mean().reset_index()
        stat_data = pd.concat([stat_data, pd.DataFrame({
            'Value': patient_data_avg[parameter_name],
            'Group': 'patients'
        })], ignore_index=True)

    # Prepare data for statistical tests
    groups = stat_data['Group'].unique()
    data_to_test = [stat_data[stat_data['Group'] == group]['Value'].values for group in groups]

    # Perform Kruskal-Wallis test
    _, p_value_main = kruskal(*data_to_test)

    # Store overall Kruskal-Wallis p-value
    if not results_df.empty:
        results_df = pd.concat([results_df, pd.DataFrame([{'Parameter': parameter_name, 'P-Value': p_value_main}])], ignore_index=True)
    else:
        results_df = pd.DataFrame([{'Parameter': parameter_name, 'P-Value': p_value_main}])

    return results_df


def calculate_p_values_HC(data_controls_comfort, data_controls_max, parameter_name, results_df_HC):
    """
    Computes the statistical significance of a given parameter between 
    Healthy Controls (HC) at comfort speed and maximum speed.

    Statistical significance is tested using the Kruskal-Wallis test for overall group differences.

    Parameters:
    ----------
    data_controls_comfort : pd.DataFrame
        DataFrame containing control group data at comfort speed, including the specified parameter.
    data_controls_max : pd.DataFrame
        DataFrame containing control group data at max speed, including the specified parameter.
    parameter_name : str
        The name of the parameter to be analyzed.
    results_df_HC : pd.DataFrame
        DataFrame to store p-values from the statistical tests.

    Returns:
    -------
    pd.DataFrame
        Updated results_df_HC with added p-values for the given parameter.

    Notes:
    ------
    - Uses the Kruskal-Wallis test to compare comfort speed and max speed.
    - Stores p-values in results_df_HC.
    """

    # Create a DataFrame for statistical analysis
    stat_data = pd.DataFrame()

    # Calculate mean values for control group at comfort speed
    controls_comfort_avg = data_controls_comfort.groupby('Participant')[parameter_name].mean().reset_index()
    stat_data = pd.concat([stat_data, pd.DataFrame({
        'Value': controls_comfort_avg[parameter_name],
        'Group': 'comfort speed'
    })], ignore_index=True)

    # Calculate mean values for control group at max speed
    controls_max_avg = data_controls_max.groupby('Participant')[parameter_name].mean().reset_index()
    stat_data = pd.concat([stat_data, pd.DataFrame({
        'Value': controls_max_avg[parameter_name],
        'Group': 'max speed'
    })], ignore_index=True)

    # Perform Kruskal-Wallis test
    _, p_val = kruskal(
        stat_data[stat_data['Group'] == 'comfort speed']['Value'],
        stat_data[stat_data['Group'] == 'max speed']['Value']
    )

    # Store Kruskal-Wallis p-value
    if not results_df_HC.empty:
        results_df_HC = pd.concat([results_df_HC, pd.DataFrame([{'Parameter': parameter_name, 'P-Value': p_val}])], ignore_index=True)
    else:
        results_df_HC = pd.DataFrame([{'Parameter': parameter_name, 'P-Value': p_val}])

    return results_df_HC

# Compare ACC and SSD
def compare_ACC_SSD(data_participants, df_results, speed):
    """
    Compares ACC (angular component of coefficient of correspondence) and 
    SSD (sum of squared distances) parameters in patients using the Kruskal-Wallis test,
    with Holm-Bonferroni correction for multiple comparisons.

    This function performs non-parametric statistical comparisons between:
    - ACC (hip/knee) Mean vs. ACC (ankle/knee) Mean
    - Asymmetry SSD (hip/knee) vs. Asymmetry SSD (ankle/knee)

    The function calculates:
    - The Kruskal-Wallis H-statistic to test for significant differences.
    - The p-value to assess statistical significance.
    - Epsilon-squared (ε²) as an effect size measure.

    Parameters:
    ----------
    data_participants : pd.DataFrame
        DataFrame containing participant data, including ACC and SSD parameters.
    speed : str
        A string representing the speed condition (e.g., 'comfort' or 'max') for labeling results.

    Returns:
    -------
    pd.DataFrame
        Updated results_df_HC with p-values from the Kruskal-Wallis tests.
    """

    # Define parameter pairs to compare
    parameter_pairs = [
        ['ACC (hip/knee) Mean', 'ACC (ankle/knee) Mean'],
        ['Asymmetry SSD (hip/knee)', 'Asymmetry SSD (ankle/knee)']
    ]

    names = ['ACC', 'Asymmetry SSD']

    test_results = []  # Store results before correction
    
    for pair in parameter_pairs:

        param_prox, param_dist = pair

        # Extract relevant data
        param_dist_data = data_participants[['Participant', param_dist]]
        param_prox_data = data_participants[['Participant', param_prox]]

        # Ensure the datasets have matching participants
        merged_data = param_prox_data.merge(param_dist_data, on='Participant', how='inner')

        # Compute means
        param_dist_avg = merged_data.groupby('Participant')[param_dist].mean().reset_index()
        param_prox_avg = merged_data.groupby('Participant')[param_prox].mean().reset_index()

        # Perform Kruskal-Wallis test
        H_statistic, p_val = kruskal(param_dist_avg[param_dist], param_prox_avg[param_prox])

        # Calculate degrees of freedom (df)
        df = 2 - 1

        # Calculate epsilon-squared (ε²)
        n_total = len(param_dist_avg) + len(param_prox_avg)
        epsilon_squared = (H_statistic - df + 1) / (n_total - df)

        # Store results for correction
        test_results.append((names[parameter_pairs.index(pair)], p_val, epsilon_squared))

    # Apply Holm-Bonferroni correction
    if test_results:
        p_values = [result[1] for result in test_results]
        corrected_p_values = multipletests(p_values, method='holm')[1]

        # Create a DataFrame for the new results
        new_results_df = pd.DataFrame({
            'Comparison': [result[0] + f' ({speed})' for result in test_results],
            'Original P-Value': p_values,
            'Corrected P-Value': corrected_p_values,
            'Epsilon-Squared': [result[2] for result in test_results]
        })

        # Use pd.concat() instead of append (since append is deprecated)
        df_results = pd.concat([df_results, new_results_df], ignore_index=True)

    return df_results


def holm_bonferroni_correction(df, alpha=0.05):
    """
    Performs Holm-Bonferroni correction for multiple hypothesis testing.

    The function:
    1. Sorts p-values in ascending order.
    2. Computes adjusted alpha thresholds for each hypothesis.
    3. Determines statistical significance using a stepwise rejection rule.
    4. Updates the 'Reject' column based on cumulative significance.
    5. Returns a DataFrame with corrected results.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing at least a column named 'P-Value' with raw p-values 
        from multiple statistical tests.
    alpha : float, optional (default=0.05)
        The overall significance level for hypothesis testing.

    Returns:
    -------
    pd.DataFrame
        Updated DataFrame with:
        - 'Adjusted Alpha': The threshold for significance after correction.
        - 'Reject': Boolean indicating whether the null hypothesis is rejected.
        - 'Reject Cummax': Ensures that all hypotheses up to the largest rejected 
          one are also rejected.
    """

    # Sort p-values in ascending order
    df_sorted = df.sort_values('P-Value').reset_index(drop=True)
    m = len(df_sorted)
    df_sorted['Adjusted Alpha'] = [alpha / (m - i) for i in range(m)]
    df_sorted['Reject'] = df_sorted['P-Value'] <= df_sorted['Adjusted Alpha']

    # Compute Holm-Bonferroni adjusted p-values
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted['Index'] = df_sorted.index + 1
    df_sorted['Adjusted P-Value'] = df_sorted['P-Value'] * (m - df_sorted['Index'])
    
    # Determine the largest k for which we reject H0
    df_sorted['Reject Cummax'] = df_sorted['Reject'][::-1].cummax()[::-1]
    
    # Update Reject column based on Holm-Bonferroni method
    df_sorted['Reject'] = df_sorted['Reject Cummax']
    
    # Reorder to the original order
    df_corrected = df_sorted.sort_index()
    
    return df_corrected


def plot_parameters(data_controls_max, data_patients, parameter_name, adj_p_value):
    """
    Creates a boxplot comparing a specified parameter between Healthy Controls (HC) and Patients, 
    optionally differentiating between the less impaired (LI) and more impaired (MI) legs.
    Statistical significance was tested using the Kruskal-Wallis test, followed by pairwise t-tests 
    if significant differences are found. Significant results are annotated on the plot.

    Parameters:
    ----------
    data_controls_max : pd.DataFrame
        DataFrame containing control group data at maximum speed, including the specified parameter.
    data_patients : pd.DataFrame
        DataFrame containing patient data, including the specified parameter and leg-specific values if applicable.
    parameter_name : str
        The name of the parameter to be plotted.
    results_df : pd.DataFrame
        DataFrame to store p-values from the statistical tests.

    Saves:
    ------
    - A boxplot in the 'Figures/SCI_Baseline/' directory with a formatted filename based on the parameter name.

    Notes:
    ------
    - Uses Kruskal-Wallis test for overall group differences.
    - Performs pairwise t-tests between groups if the Kruskal-Wallis test is significant.
    - Significant results are annotated with:
      - * (p < 0.05)
      - ** (p < 0.01)
      - *** (p < 0.001)
    """

    # Determine if we differentiate legs
    if ' Mean' in parameter_name:
        differentiate_legs = True
        parameter_base = parameter_name.replace(' Mean', '')
        less_impaired_param = parameter_base + ' LI'
        more_impaired_param = parameter_base + ' MI'
    else:
        differentiate_legs = False

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame()

    # Calculating the mean for control data if multiple entries per participant exist
    if not parameter_name.startswith('SSD'):
        controls_data_avg = data_controls_max.groupby('Participant')[[parameter_name]].mean().reset_index()
        plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': controls_data_avg[parameter_name],
            'Group': 'HC'
        })], ignore_index=True)

    if differentiate_legs:
        # Calculate the mean for each parameter for each participant (so each person is plotted once)
        patient_data_avg = data_patients.groupby('Participant').agg({
            less_impaired_param: 'mean',
            more_impaired_param: 'mean',
        }).reset_index()

        plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': patient_data_avg[less_impaired_param],
            'Group': 'less \nimpaired'
        })], ignore_index=True)
        plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': patient_data_avg[more_impaired_param],
            'Group': 'more \nimpaired'
        })], ignore_index=True)
    else:
        # Calculate the mean for the parameter for each participant
        patient_data_avg = data_patients.groupby('Participant')[[parameter_name]].mean().reset_index()
        plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': patient_data_avg[parameter_name],
            'Group': 'patients'
        })], ignore_index=True)

    # Prepare data for Matplotlib boxplot
    groups = plot_data['Group'].unique()
    data_to_plot = [plot_data[plot_data['Group'] == group]['Value'].values for group in groups]

    # Define colors for each group
    colors = {
        'HC': 'lightgrey',
        'less \nimpaired': 'cornflowerblue',
        'more \nimpaired': 'crimson',
        'patients': 'lightgreen'
    }
    box_colors = [colors[group] for group in groups]

    # Perform statistical tests
    group_combinations = [
        ('HC', 'less \nimpaired'),
        ('HC', 'more \nimpaired'),
        ('less \nimpaired', 'more \nimpaired'),
        ('HC', 'patients')
    ]

    # Create the boxplot
    box = plt.boxplot(data_to_plot, patch_artist=True, widths=0.4, showfliers=False)

    # Remove the top and left spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Customize the boxplot
    for patch, color in zip(box['boxes'], box_colors):
        patch.set(facecolor=color, edgecolor='black')
    for median in box['medians']:
        median.set(color='black')

    # Add scatter plot for individual data points
    for i, group in enumerate(groups, start=1):
        values = plot_data[plot_data['Group'] == group]['Value'].values
        x = np.random.normal(i, 0.04, size=len(values))
        plt.scatter(x, values, color='k', s=25, alpha=0.7, edgecolors='none', zorder=2)

    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(ticks=range(1, len(groups) + 1), labels=groups, size=20)
    plt.yticks(size=18)

    plt.axvline(x=1.5, color='grey', linestyle='--', linewidth=1)

    # Perform statistical tests and add significance annotations
    max_y = plot_data['Value'].max()
    min_y = plot_data['Value'].min()
    y_offset = (max_y - min_y) * 0.08
    y_text_offset = (max_y - min_y) * 0.06

    # Annotate significance
    if differentiate_legs:
        significance_annotations = {}
        for group1, group2 in group_combinations:
            if group1 in groups and group2 in groups:
                data1 = plot_data[plot_data['Group'] == group1]['Value']
                data2 = plot_data[plot_data['Group'] == group2]['Value']
                _, p_val = ttest_ind(data1, data2)
                significance_annotations[(group1, group2)] = p_val
    
    else:
        significance_annotations = {}
        for group1, group2 in group_combinations:
            if group1 in groups and group2 in groups:
                data1 = plot_data[plot_data['Group'] == group1]['Value']
                data2 = plot_data[plot_data['Group'] == group2]['Value']
                # If there are only two groups use corrected value
                significance_annotations[(group1, group2)] = adj_p_value

    current_y_offset = 0
    for (group1, group2), p_val in significance_annotations.items():
        if p_val < 0.001:
            annotation = '***'
        elif p_val < 0.01:
            annotation = '**'
        elif p_val < 0.05:
            annotation = '*'
        else:
            annotation = ' '

        if p_val < 0.05:
            group1_idx = list(groups).index(group1) + 1
            group2_idx = list(groups).index(group2) + 1
            y_line = max_y + current_y_offset + y_offset
            y_text = max_y + current_y_offset + y_text_offset

            plt.plot([group1_idx, group2_idx], [y_line, y_line], color='k', lw=1)
            plt.text((group1_idx + group2_idx) / 2, y_text, annotation, ha='center', va='bottom', color='k', fontsize=18)

            if parameter_name.startswith('RoM hip'):
                plt.ylim(min_y - y_offset, y_text + 2*y_offset)
            else:
                plt.ylim(min_y - y_offset, y_text + y_offset)

            current_y_offset += y_offset

    # Set titles and labels based on the parameter name
    if parameter_name.startswith('RoM ankle'):
        plt.ylabel('Ankle flexion RoM [deg]', size=20)
    elif parameter_name.startswith('RoM knee'):
        plt.ylabel('Knee flexion RoM [deg]', size=20)
    elif parameter_name == 'Velocity':
        plt.ylabel('Velocity [m/s]', size=20)
    elif parameter_name == 'Distance per stroke':
        plt.ylabel('Distance per stroke [m]', size=20)
    elif parameter_name == 'Time':
        plt.ylabel('Time [s]', size=20)
    else:
        plt.title(f'{parameter_name}', size=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'SCI_Baseline/' + re.sub(r'_Mean$', '', parameter_name.replace('/', '_').replace(' ', '_')) + '.png'), dpi=300)
    plt.close()



def plot_parameters_HC(data_controls_comfort, data_controls_max, parameter_name, adj_p_value):
    """
    Creates a boxplot comparing a specified parameter between maximum and cofort speed of 
    Healthy Controls (HC). Statistical significance was tested using the Kruskal-Wallis test. 
    Significant results are annotated on the plot.

    Parameters:
    ----------
    data_controls_comfort : pd.DataFrame
        DataFrame containing control group data at comfort speed, including the specified parameter.
    data_controls_max : pd.DataFrame
        DataFrame containing control group data at max speed, including the specified parameter.
    parameter_name : str
        The name of the parameter to be plotted.

    Saves:
    ------
    - A boxplot in the 'Figures/Healthy_Controls/' directory with a formatted filename based on the parameter name.

    Notes:
    ------
    - Uses Kruskal-Wallis test for overall group differences.
    - Performs pairwise t-tests between groups if the Kruskal-Wallis test is significant.
    - Significant results are annotated with:
      - * (p < 0.05)
      - ** (p < 0.01)
      - *** (p < 0.001)
    """

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame()
    
    # Calculate the mean for control data at comfort speed if multiple entries per participant exist
    controls_comfort_avg = data_controls_comfort.groupby('Participant')[parameter_name].mean().reset_index()
    plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
        'Value': controls_comfort_avg[parameter_name],
        'Group': 'comfort speed'
    })], ignore_index=True)

    # Calculate the mean for control data at max speed
    controls_max_avg = data_controls_max.groupby('Participant')[parameter_name].mean().reset_index()
    plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
        'Value': controls_max_avg[parameter_name],
        'Group': 'max speed'
    })], ignore_index=True)

    # Prepare data for Matplotlib boxplot
    groups = plot_data['Group'].unique()
    data_to_plot = [plot_data[plot_data['Group'] == group]['Value'].values for group in groups]

    # Define colors for each group
    colors = {
        'comfort speed': 'lightgrey',
        'max speed': 'lightgrey'
    }
    box_colors = [colors[group] for group in groups]

    # Create the boxplot
    box = plt.boxplot(data_to_plot, patch_artist=True, widths=0.5, showfliers=False)

    # Remove the top and left spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Customize the boxplot
    for patch, color in zip(box['boxes'], box_colors):
        patch.set(facecolor=color, edgecolor='black')
    for median in box['medians']:
        median.set(color='black')

    # Add scatter plot for individual data points
    for i, group in enumerate(groups, start=1):
        values = plot_data[plot_data['Group'] == group]['Value'].values
        x = np.random.normal(i, 0.04, size=len(values))  # Jittering the points
        plt.scatter(x, values, color='k', s=22, alpha=0.7, edgecolors='none', zorder=2)

    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(ticks=range(1, len(groups) + 1), labels=groups, size=20)
    plt.yticks(size=18)

    # Define significance annotation based on p-value
    if adj_p_value < 0.001:
        annotation = '***'
    elif adj_p_value < 0.01:
        annotation = '**'
    elif adj_p_value < 0.05:
        annotation = '*'
    else:
        annotation = None  # No annotation if not significant

    # Add significance annotation if significant
    if annotation:
        max_y = plot_data['Value'].max()
        min_y = plot_data['Value'].min()
        y_line = max_y + (max_y - min_y) * 0.05  # Position for the line
        y_text = max_y + (max_y - min_y) * 0.08  # Position for the text

        plt.plot([1, 2], [y_line, y_line], color='k', lw=0.8)  # Add the line
        plt.text(1.5, y_text, annotation, ha='center', va='bottom', color='k', fontsize=18)  # Add the text

        # Adjust the y-limit to be just above the text
        plt.ylim(min_y - (max_y - min_y) * 0.05, y_text + (max_y - min_y) * 0.12)

    # Set titles and labels based on the parameter name
    if parameter_name == 'Stroke duration Mean':
        plt.ylabel('Stroke duration [s]', size=20)
        plt.ylim(0, 2.55)
    elif parameter_name == 'Sidewards displacement Mean':
        plt.ylabel('Lateral ankle displacement [m]', size=20)
        plt.ylim(0, 0.105)
    elif parameter_name == 'Horizontal displacement Mean':
        plt.ylabel('Horizontal ankle displacement [m]', size=18)
        plt.ylim(0, 0.165)
    else:
        plt.title(f'{parameter_name}', size=18)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'Healthy_Controls/' + re.sub(r'_Mean$', '', parameter_name.replace('/', '_').replace(' ', '_')) + '.png'), dpi=300)
    plt.close()


def calculate_p_values_with_demographic_correction(data_controls_max, data_patients, parameter_name, results_df):
    """
    Computes statistical significance of a given parameter between Healthy Controls (HC) 
    and Patients, optionally differentiating between the less impaired (LI) and more impaired (MI) legs.

    Statistical significance is tested using the Kruskal-Wallis test for overall group differences,
    after adjusting for Age and BMI.

    Parameters:
    ----------
    data_controls_max : pd.DataFrame
        DataFrame containing control group data, including the specified parameter.
    data_patients : pd.DataFrame
        DataFrame containing patient data, including the specified parameter and leg-specific values if applicable.
    parameter_name : str
        The name of the parameter to be analyzed.
    results_df : pd.DataFrame
        DataFrame to store p-values from the statistical tests.

    Returns:
    -------
    pd.DataFrame
        Updated results_df with added p-values for the given parameter.

    Notes:
    ------
    - Uses Kruskal-Wallis test for overall group differences.
    - Performs pairwise t-tests between groups if the Kruskal-Wallis test is significant.
    """

    # Determine if we differentiate legs
    if ' Mean' in parameter_name:
        differentiate_legs = True
        parameter_base = parameter_name.replace(' Mean', '')
        less_impaired_param = parameter_base + ' LI'
        more_impaired_param = parameter_base + ' MI'
    else:
        differentiate_legs = False

    # Create a DataFrame for statistical analysis
    stat_data = pd.DataFrame()

    # Calculate mean values for control group
    if not parameter_name.startswith('SSD'):
        controls_data_avg = data_controls_max.groupby('Participant')[[parameter_name, 'Age', 'BMI']].mean().reset_index()
        stat_data = pd.concat([stat_data, pd.DataFrame({
            'Value': controls_data_avg[parameter_name],
            'Age': controls_data_avg['Age'],
            'BMI': controls_data_avg['BMI'],
            'Group': 'HC'
        })], ignore_index=True)

    if differentiate_legs:
        # Get the mean for each parameter per participant
        patient_data_avg = data_patients.groupby('Participant').agg({
            less_impaired_param: 'mean',
            more_impaired_param: 'mean',
            'Age': 'mean',
            'BMI': 'mean'
        }).reset_index()

        stat_data = pd.concat([stat_data, pd.DataFrame({
            'Value': patient_data_avg[less_impaired_param],
            'Age': patient_data_avg['Age'],
            'BMI': patient_data_avg['BMI'],
            'Group': 'less \nimpaired'
        })], ignore_index=True)
        stat_data = pd.concat([stat_data, pd.DataFrame({
            'Value': patient_data_avg[more_impaired_param],
            'Age': patient_data_avg['Age'],
            'BMI': patient_data_avg['BMI'],
            'Group': 'more \nimpaired'
        })], ignore_index=True)
    else:
        patient_data_avg = data_patients.groupby('Participant')[[parameter_name, 'Age', 'BMI']].mean().reset_index()
        stat_data = pd.concat([stat_data, pd.DataFrame({
            'Value': patient_data_avg[parameter_name],
            'Age': patient_data_avg['Age'],
            'BMI': patient_data_avg['BMI'],
            'Group': 'patients'
        })], ignore_index=True)

    # Ensure stat_data is a DataFrame and drop NaN values
    stat_data_reg = stat_data.dropna(subset=['Value', 'Age', 'BMI']).copy()

    # Check if DataFrame is empty after dropping NaN
    if stat_data_reg.empty:
        print(f"Skipping parameter {parameter_name} due to missing data.")
        return results_df

    # Regress Value on Age and BMI to get residuals
    X = sm.add_constant(stat_data_reg[['Age', 'BMI']])  # Add constant for intercept
    model = sm.OLS(stat_data_reg['Value'], X).fit()
    stat_data_reg['Residuals'] = model.resid

    # Use residuals instead of raw values for the Kruskal-Wallis test
    groups = stat_data_reg['Group'].unique()
    data_to_test = [stat_data_reg[stat_data_reg['Group'] == group]['Residuals'].values for group in groups]

    # Perform Kruskal-Wallis test
    _, p_value_main = kruskal(*data_to_test)

    # Store overall Kruskal-Wallis p-value
    results_df = pd.concat([results_df,{'Parameter': parameter_name, 'P-Value': p_value_main}], ignore_index=True)

    return results_df


def plot_proximal_distal_HC(data_controls_comfort, data_controls_max, corrected_pvalues):

    # Define parameter pairs to compare
    parameter_pairs = [
        ['ACC (ankle/knee) Mean', 'ACC (hip/knee) Mean'],
        ['Asymmetry SSD (ankle/knee)', 'Asymmetry SSD (hip/knee)']
    ]

    pval1, pval2, pval3, pval4 = corrected_pvalues

    pvalue_pairs = [
        [pval1, pval2],
        [pval3, pval4]
    ]

    titles = ['ACC', 'Asymmetry SSD']
    titles_pos = [1.1, 11.5]

    lower_limits = [0.59, 0]
    upper_limits = [1.11, 12]

    ticks_labels=[[0.6, 0.7, 0.8, 0.9, 1.0],  [0, 2, 4, 6, 8, 10]]

    significances = [1.04, 10]

    for pair in parameter_pairs:

        plt.figure(figsize=(12.8, 4.8))  # Default: [6.4 4.8]

        # Create a DataFrame for plotting
        plot_data = pd.DataFrame()

        # Calculate the mean for control data at comfort and max speed
        controls_comfort_avg_ankle_knee = data_controls_comfort.groupby(
            'Participant')[pair[0]].mean().reset_index()
        controls_max_avg_ankle_knee = data_controls_max.groupby(
            'Participant')[pair[0]].mean().reset_index()

        controls_comfort_avg_hip_knee = data_controls_comfort.groupby(
            'Participant')[pair[1]].mean().reset_index()
        controls_max_avg_hip_knee = data_controls_max.groupby(
            'Participant')[pair[1]].mean().reset_index()

        # Append the data for plotting
        plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': controls_comfort_avg_ankle_knee[pair[0]],
            'Group': 'Ankle/Knee Comfort'
        })], ignore_index=True)

        plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': controls_max_avg_ankle_knee[pair[0]],
            'Group': 'Ankle/Knee Max'
        })], ignore_index=True)

        plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': controls_comfort_avg_hip_knee[pair[1]],
            'Group': 'Hip/Knee Comfort'
        })], ignore_index=True)

        plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': controls_max_avg_hip_knee[pair[1]],
            'Group': 'Hip/Knee Max'
        })], ignore_index=True)

        # Define colors for each group
        palette = {
            'Ankle/Knee Comfort': 'lightgrey',
            'Ankle/Knee Max': 'lightgrey',
            'Hip/Knee Comfort': 'lightgrey',
            'Hip/Knee Max': 'lightgrey'
        }

        # Plotting using seaborn
        sns.boxplot(x='Group', y='Value', hue='Group', data=plot_data, 
            palette=palette, showfliers=False, width=0.45,
            boxprops=dict(edgecolor='black', linewidth=0.8), 
            medianprops=dict(color='black', linewidth=0.8),
            whiskerprops=dict(color='black', linewidth=0.8), 
            capprops=dict(color='black', linewidth=0.8),
            dodge=False)

        # Adding stripplot to show individual data points with jitter
        sns.stripplot(x='Group', y='Value', data=plot_data, color='k', 
                      size=5, jitter=True, dodge=True, alpha=0.7)

        # Annotate significance stars
        def annotate_significance(p_value, x1, x2, y, h, col='k'):
            if p_value < 0.001:
                text = '***'
            elif p_value < 0.01:
                text = '**'
            elif p_value < 0.05:
                text = '*'
            else:
                text = ''
            plt.plot([x1, x2], [y+h, y+h], lw=0.8, c=col)
            plt.text((x1 + x2) * 0.5, y+h, text, ha='center', 
                     va='bottom', color=col, fontsize=18)

        # Get y positions for annotations
        ymax1 = significances[parameter_pairs.index(pair)]  # Adjust based on your data
        annotate_significance(pvalue_pairs[parameter_pairs.index(pair)][0], 1, 3, ymax1, 0)

        # Define an offset to make the second annotation higher
        h_offset = 0.1 * (plt.ylim()[1] - plt.ylim()[0])

        # Get y positions for annotations
        ymax2 = ymax1 + h_offset  # Move the second annotation higher
        annotate_significance(pvalue_pairs[parameter_pairs.index(pair)][1], 0, 2, ymax2, 0)

        # Remove the top and left spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add dashed vertical line at x=1.5
        plt.axvline(x=1.5, color='k', linewidth=1)

        # Add labels
        plt.text(0.5, titles_pos[parameter_pairs.index(pair)], 
                 'ankle-knee', ha='center', va='bottom', fontsize=20)
        plt.text(2.5, titles_pos[parameter_pairs.index(pair)], 
                 'knee-hip', ha='center', va='bottom', fontsize=20)

        # Set labels and title
        plt.ylabel(titles[parameter_pairs.index(pair)], size=20)
        plt.xlabel('')
        plt.ylim(lower_limits[parameter_pairs.index(pair)], 
                 upper_limits[parameter_pairs.index(pair)])
        plt.xticks(ticks=[0, 1, 2, 3], labels=['comfort speed', 'max speed', 
                                               'comfort speed', 'max speed'], size=20)
        plt.yticks(ticks=ticks_labels[parameter_pairs.index(pair)], 
                   labels=ticks_labels[parameter_pairs.index(pair)], size=18)

        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, 'Healthy_Controls/' + titles[
            parameter_pairs.index(pair)].replace(' ', '_') + '_HC.png'), dpi=300)
        plt.close()


def plot_proximal_distal(data_controls_max, data_patients):

    ################## SSD ##################

    # Define parameter names for SSD
    parameter_pairs = [
        ('SSD (ankle/knee)', 'SSD (hip/knee)')
    ]

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame()

    for pair in parameter_pairs:
        # Define more/less impaired legs for both ankle/knee and hip/knee
        parameter_ankle_knee = pair[0]
        parameter_hip_knee = pair[1]

        more_impaired_ankle_knee = parameter_ankle_knee + ' MI'
        less_impaired_ankle_knee = parameter_ankle_knee + ' LI'

        more_impaired_hip_knee = parameter_hip_knee + ' MI'
        less_impaired_hip_knee = parameter_hip_knee + ' LI'

        # Calculate the mean for each parameter for each participant
        patient_data_avg = data_patients.groupby('Participant').agg({
            more_impaired_ankle_knee: 'mean',
            less_impaired_ankle_knee: 'mean',
            more_impaired_hip_knee: 'mean',
            less_impaired_hip_knee: 'mean'
        }).reset_index()

        # Append the data for plotting
        plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': patient_data_avg[more_impaired_ankle_knee],
            'Group': 'Ankle/Knee More Impaired'
        })], ignore_index=True)
        plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': patient_data_avg[less_impaired_ankle_knee],
            'Group': 'Ankle/Knee Less Impaired'
        })], ignore_index=True)
        plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': patient_data_avg[more_impaired_hip_knee],
            'Group': 'Hip/Knee More Impaired'
        })], ignore_index=True)
        plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
            'Value': patient_data_avg[less_impaired_hip_knee],
            'Group': 'Hip/Knee Less Impaired'
        })], ignore_index=True)

    # Prepare data for Matplotlib boxplot
    groups = plot_data['Group'].unique()
    data_to_plot = [plot_data[plot_data['Group'] == group]['Value'].values for group in groups]

    # Define colors for each group
    colors = {
        'Ankle/Knee More Impaired': 'crimson',
        'Ankle/Knee Less Impaired': 'cornflowerblue',
        'Hip/Knee More Impaired': 'crimson',
        'Hip/Knee Less Impaired': 'cornflowerblue'
    }
    box_colors = [colors[group] for group in groups]

    plt.figure(figsize=(12.8, 4.8))  # Default: [6.4 4.8]

    # Create the boxplot
    box = plt.boxplot(data_to_plot, patch_artist=True, widths=0.4, showfliers=False)

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Customize the boxplot
    for patch, color in zip(box['boxes'], box_colors):
        patch.set(facecolor=color, edgecolor='black')
    for median in box['medians']:
        median.set(color='black')

    # Add scatter plot for individual data points
    for i, group in enumerate(groups, start=1):
        values = plot_data[plot_data['Group'] == group]['Value'].values
        x = np.random.normal(i, 0.04, size=len(values))  # Jittering the points
        plt.scatter(x, values, color='k', s=25, alpha=0.7, edgecolors='none', zorder=2)

    # Perform statistical tests and add significance annotations
    max_y = plot_data['Value'].max()
    min_y = plot_data['Value'].min()
    y_offset = (max_y - min_y) * 0.1
    y_text_offset = (max_y - min_y) * 0.08

    significance_annotations = {}
    group_combinations = [
        ('Ankle/Knee More Impaired', 'Hip/Knee More Impaired'),
        ('Ankle/Knee Less Impaired', 'Hip/Knee Less Impaired')
    ]

    # Perform Kruskal-Wallis test for overall comparison
    group_data = [plot_data[plot_data['Group'] == group]['Value'].values for group in groups]
    _, kruskal_p = kruskal(*group_data)

    # If Kruskal-Wallis test is significant, perform pairwise Mann-Whitney U-tests
    if kruskal_p < 0.05:
        p_values = []
        for group1, group2 in group_combinations:
            if group1 in groups and group2 in groups:
                data1 = plot_data[plot_data['Group'] == group1]['Value']
                data2 = plot_data[plot_data['Group'] == group2]['Value']
                _, p_val = ttest_ind(data1, data2, alternative='two-sided')
                p_values.append(p_val)

        # Apply Bonferroni-Holm correction
        _, corrected_pvals, _, _ = multipletests(p_values, alpha=0.05, method='holm')

        # Replace significance annotations with corrected p-values
        significance_annotations = dict(zip(group_combinations, corrected_pvals))

        # Adjusting the significance labels for annotation
        current_y_offset = 0
        for (group1, group2), p_val in significance_annotations.items():
            if p_val < 0.001:
                annotation = '***'
            elif p_val < 0.01:
                annotation = '**'
            elif p_val < 0.05:
                annotation = '*'
            else:
                annotation = ''

            group1_idx = list(groups).index(group1) + 1
            group2_idx = list(groups).index(group2) + 1
            y_line = max_y + current_y_offset + y_offset
            y_text = max_y + current_y_offset + y_text_offset

            if annotation:  # Only draw if there's significance
                plt.plot([group1_idx, group2_idx], [y_line, y_line], color='k', lw=1)
                plt.text((group1_idx + group2_idx) / 2, y_text, annotation, ha='center', va='bottom', color='k', fontsize=18)

            current_y_offset += y_offset

    # Add dashed vertical line
    plt.axvline(x=2.5, color='k', linewidth=1)

    # Add labels
    plt.text(1.5, 6.2, 'ankle-knee', ha='center', va='bottom', fontsize=20)
    plt.text(3.5, 6.2, 'knee-hip', ha='center', va='bottom', fontsize=20)

    # Customize axes labels
    plt.xlabel('')
    plt.ylabel('SSD', size=20)
    plt.yticks(size=18)
    plt.xticks(ticks=[1, 2, 3, 4], labels=['less impaired', 'more impaired', 'less impaired', 'more impaired'], size=20)
    plt.ylim(0, 6.4)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'SCI_Baseline/SSD.png'), dpi=300)
    plt.close()


    ################## ACC ##################
    

    # Define parameter names
    parameter_ankle_knee = 'ACC (ankle/knee) Mean'
    less_impaired_ankle_knee = 'ACC (ankle/knee) LI'
    more_impaired_ankle_knee = 'ACC (ankle/knee) MI'

    parameter_knee_hip = 'ACC (hip/knee) Mean'
    less_impaired_knee_hip = 'ACC (hip/knee) LI'
    more_impaired_knee_hip = 'ACC (hip/knee) MI'

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame()

    # For healthy controls, include both comfort and max speed
    controls_max_avg_ankle_knee = data_controls_max.groupby('Participant')[[parameter_ankle_knee]].mean().reset_index()

    controls_max_avg_knee_hip = data_controls_max.groupby('Participant')[[parameter_knee_hip]].mean().reset_index()

    # Append the healthy control data
    plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
        'Value': controls_max_avg_ankle_knee[parameter_ankle_knee],
        'Group': 'HC (ankle/knee)'
    })], ignore_index=True)

    # Calculate the mean for each parameter for each participant (so each person is plotted once)
    patient_data_avg = data_patients.groupby('Participant').agg({
        less_impaired_ankle_knee: 'mean',
        more_impaired_ankle_knee: 'mean',
        less_impaired_knee_hip: 'mean',
        more_impaired_knee_hip: 'mean'
    }).reset_index()

    # Append the less and more impaired data for ankle/knee and knee/hip
    plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
        'Value': patient_data_avg[less_impaired_ankle_knee],
        'Group': 'LI (ankle/knee)'
    })], ignore_index=True)
    
    plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
        'Value': patient_data_avg[more_impaired_ankle_knee],
        'Group': 'MI (ankle/knee)'
    })], ignore_index=True)

    plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
        'Value': controls_max_avg_knee_hip[parameter_knee_hip],
        'Group': 'HC (knee/hip)'
    })], ignore_index=True)
    
    plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
        'Value': patient_data_avg[less_impaired_knee_hip],
        'Group': 'LI (knee/hip)'
    })], ignore_index=True)

    plot_data = plot_data = pd.concat([plot_data, pd.DataFrame({
        'Value': patient_data_avg[more_impaired_knee_hip],
        'Group': 'MI (knee/hip)'
    })], ignore_index=True)

    # Prepare data for Matplotlib boxplot
    groups = plot_data['Group'].unique()
    data_to_plot = [plot_data[plot_data['Group'] == group]['Value'].values for group in groups]

    # Define colors for each group
    colors = {
        'HC (ankle/knee)': 'lightgrey',
        'LI (ankle/knee)': 'cornflowerblue',
        'MI (ankle/knee)': 'crimson',
        'HC (knee/hip)': 'lightgrey',
        'LI (knee/hip)': 'cornflowerblue',
        'MI (knee/hip)': 'crimson'
    }
    box_colors = [colors[group] for group in groups]

    plt.figure(figsize=(12.8, 4.8))  # Default: [6.4 4.8]

    # Create the boxplot
    box = plt.boxplot(data_to_plot, patch_artist=True, widths=0.4, showfliers=False)

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Customize the boxplot
    for patch, color in zip(box['boxes'], box_colors):
        patch.set(facecolor=color, edgecolor='black')
    for median in box['medians']:
        median.set(color='black')

    # Add scatter plot for individual data points
    for i, group in enumerate(groups, start=1):
        values = plot_data[plot_data['Group'] == group]['Value'].values
        x = np.random.normal(i, 0.04, size=len(values))  # Jittering the points
        plt.scatter(x, values, color='k', s=25, alpha=0.7, edgecolors='none', zorder=2)

    plt.axvline(x=1.5, color='grey', linestyle='--', linewidth=1)
    plt.axvline(x=4.5, color='grey', linestyle='--', linewidth=1)
    plt.axvline(x=3.5, color='k', linewidth=1)

    plt.text(2, 1.08, 'ankle-knee', ha='center', va='bottom', fontsize=20)
    plt.text(5, 1.08, 'knee-hip', ha='center', va='bottom', fontsize=20)

    # Perform statistical tests and add significance annotations
    max_y = plot_data['Value'].max() - 0.02
    min_y = plot_data['Value'].min()
    y_offset = (max_y - min_y) * 0.08
    y_text_offset = (max_y - min_y) * 0.05

    group_combinations = [
        ('HC (ankle/knee)', 'HC (knee/hip)'),
        ('LI (ankle/knee)', 'LI (knee/hip)'),
        ('MI (ankle/knee)', 'MI (knee/hip)')
    ]

    # Perform Kruskal-Wallis test for overall comparison
    group_data = [plot_data[plot_data['Group'] == group]['Value'].values for group in groups]
    _, kruskal_p = kruskal(*group_data)

    # If Kruskal-Wallis test is significant, perform pairwise Mann-Whitney U-tests
    if kruskal_p < 0.05:
        p_values = []
        for group1, group2 in group_combinations:
            if group1 in groups and group2 in groups:
                data1 = plot_data[plot_data['Group'] == group1]['Value']
                data2 = plot_data[plot_data['Group'] == group2]['Value']
                _, p_val = ttest_ind(data1, data2, alternative='two-sided')
                p_values.append(p_val)

        # Apply Bonferroni-Holm correction
        _, corrected_pvals, _, _ = multipletests(p_values, alpha=0.05, method='holm')

        # Replace significance annotations with corrected p-values
        significance_annotations = dict(zip(group_combinations, corrected_pvals))

        # Adjusting the significance labels for annotation
        current_y_offset = 0
        for (group1, group2), p_val in significance_annotations.items():
            if p_val < 0.001:
                annotation = '***'
            elif p_val < 0.01:
                annotation = '**'
            elif p_val < 0.05:
                annotation = '*'
            else:
                annotation = ''
            
            # To match previous figure (before multiple testing correction)
            if group1 == 'HC (ankle/knee)':
                annotation = '**'

            group1_idx = list(groups).index(group1) + 1
            group2_idx = list(groups).index(group2) + 1
            y_line = max_y + current_y_offset + y_offset
            y_text = max_y + current_y_offset + y_text_offset

            plt.plot([group1_idx, group2_idx], [y_line, y_line], color='k', lw=1)
            plt.text((group1_idx + group2_idx) / 2, y_text, annotation, ha='center', va='bottom', color='k', fontsize=18)

            current_y_offset += y_offset

    plt.xlabel('')
    plt.ylabel('ACC', size=20)
    plt.ylim(0.59, 1.11)
    plt.yticks(ticks=[0.6, 0.7, 0.8, 0.9, 1.0], labels=['0.6', '0.7', '0.8', '0.9', '1.0'], size=18)
    plt.xticks(ticks=[1, 2, 3, 4, 5, 6], labels=['HC', 'less \nimpaired', 
                                                    'more \nimpaired', 'HC',
                                                    'less \nimpaired', 'more \nimpaired'], size=20)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'SCI_Baseline/ACC.png'), dpi=300)
    plt.close()


# Z-Scores with resepct to HCs
def calculate_z_scores(df, df_reference):
    # Remove columns ID, Time, Strokes
    z_scores_df = pd.DataFrame(index=df.index, columns=df.columns[4:])
    
    for column in df.columns[4:]:
        mean_hc = df_reference[column].mean()
        std_hc = df_reference[column].std()
        z_scores_df[column] = (df[column] - mean_hc) / std_hc
    
    return z_scores_df


def swimming_profiles(df_bl, df_hc_comfortspeed_means, df_hc_maxspeed_means):
    # Drop non-numeric column
    df_bl = df_bl.drop(columns=['More impaired leg'])

    # Make column names the same
    df_bl_more_impaired = df_bl[[col for col in df_bl.columns if not ' LI' in col]]
    df_bl_less_impaired = df_bl[[col for col in df_bl.columns if not ' MI' in col]]
    df_bl_more_impaired.columns = df_bl_more_impaired.columns.str.replace(r' MI$', '', regex=True)
    df_bl_less_impaired.columns = df_bl_less_impaired.columns.str.replace(r' LI$', '', regex=True)
    df_hc_comfortspeed_means.columns = df_hc_comfortspeed_means.columns.str.replace(r' Mean$', '', regex=True)
    df_hc_maxspeed_means.columns = df_hc_maxspeed_means.columns.str.replace(r' Mean$', '', regex=True)

    # Reindex column names to be in the same order
    df_bl_less_impaired = df_bl_less_impaired.reindex(columns=df_hc_maxspeed_means.columns)
    df_bl_more_impaired = df_bl_more_impaired.reindex(columns=df_hc_maxspeed_means.columns)

    # Adjust column names
    for df in [df_bl_less_impaired, df_bl_more_impaired, df_hc_maxspeed_means, df_hc_comfortspeed_means]:
        df.rename(columns={'ACC (ankle/knee)': 'ACC (ankle-knee)'}, inplace=True)
        df.rename(columns={'ACC std (ankle/knee)': 'ACC variability (ankle-knee)'}, inplace=True)
        df.rename(columns={'ACC (hip/knee)': 'ACC (knee-hip)'}, inplace=True)
        df.rename(columns={'ACC std (hip/knee)': 'ACC variability (knee-hip)'}, inplace=True)
        df.rename(columns={'SSD (ankle/knee)': 'SSD (ankle-knee)'}, inplace=True)
        df.rename(columns={'SSD std (ankle/knee)': 'SSD std (ankle-knee)'}, inplace=True)
        df.rename(columns={'SSD (hip/knee)': 'SSD (knee-hip)'}, inplace=True)
        df.rename(columns={'SSD std (hip/knee)': 'SSD std (knee-hip)'}, inplace=True)
        df.rename(columns={'Asymmetry SSD (ankle/knee)': 'Asymmetry SSD (ankle-knee)'}, inplace=True)
        df.rename(columns={'Asymmetry SSD (hip/knee)': 'Asymmetry SSD (knee-hip)'}, inplace=True)
        df.rename(columns={'Sidewards displacement': 'Lateral displacement'}, inplace=True)

    # Calculate z-scores 
    z_scores_hc_maxspeed = calculate_z_scores(df_hc_maxspeed_means, df_hc_maxspeed_means)
    z_scores_more_impaired = calculate_z_scores(df_bl_more_impaired, df_hc_maxspeed_means)
    z_scores_less_impaired = calculate_z_scores(df_bl_less_impaired, df_hc_maxspeed_means)

    z_scores_hc_maxspeed_std = z_scores_hc_maxspeed.std()
    z_scores_more_impaired_std = z_scores_more_impaired.std()
    z_scores_less_impaired_std = z_scores_less_impaired.std()

    z_scores_hc_maxspeed = z_scores_hc_maxspeed.mean()
    z_scores_more_impaired = z_scores_more_impaired.mean()
    z_scores_less_impaired = z_scores_less_impaired.mean()

    all_z_scores = pd.concat([z_scores_hc_maxspeed, z_scores_less_impaired, z_scores_more_impaired], axis=1)
    all_z_scores.columns = ['healthy range', 'less impaired leg', 'more impaired leg']

    all_z_scores_std = pd.concat([z_scores_hc_maxspeed_std, z_scores_less_impaired_std, z_scores_more_impaired_std], axis=1)
    all_z_scores_std.columns = ['healthy range', 'less impaired leg', 'more impaired leg']

    # Create a radar chart
    labels = all_z_scores.index
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Spider plot
    _, ax = plt.subplots(figsize=(15, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(0)
    ax.set_theta_direction(-1)

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], labels, size=10)

    colors = ['gray', 'cornflowerblue', 'crimson']
    ha = 8 * ['left'] + ['center'] + 16 * ['right'] + ['center'] + 7 * ['left']

    for i, column in enumerate(all_z_scores.columns):
        values = all_z_scores[column].tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=1.5, linestyle='solid', color=colors[i], label=column)

        std_values = all_z_scores_std[column].tolist()
        std_values += std_values[:1]
        lower_bound = np.maximum(np.subtract(values, std_values), -3)
        upper_bound = np.minimum(np.add(values, std_values), 3)
        ax.fill_between(angles, lower_bound, upper_bound, color=colors[i], alpha=0.15)
        
    for tick, label in zip(ax.get_xticklabels(), ha):
        tick.set_horizontalalignment(label)

    # Move the second label slightly to the left
    x, y = ax.get_xticklabels()[8].get_position()
    ax.get_xticklabels()[8].set_position((x, y-0.03))
    ax.get_xticklabels()[25].set_position((x, y-0.03))

    # Function to fill an arch segment
    def fill_arch(start_angle, end_angle, lower, upper, arch_color, alpha):
        theta = np.linspace(start_angle, end_angle, 100)
        r_lower = np.full_like(theta, lower)
        r_upper = np.full_like(theta, upper)
        ax.fill_between(theta, r_lower, r_upper, color=arch_color, alpha=alpha)

    # Categorize parameters
    fill_arch(angles[0]-np.deg2rad(5), angles[5]-np.deg2rad(5), 3, 4, 'orange', 0.07)           # speed parameters
    fill_arch(angles[5]-np.deg2rad(5), angles[17]-np.deg2rad(5), 3, 4, 'purple', 0.07)          # joint angle parameters
    fill_arch(angles[17]-np.deg2rad(5), angles[20]-np.deg2rad(5), 3, 4, 'yellow', 0.07)         # displacement parameters
    fill_arch(angles[20]-np.deg2rad(5), angles[26]-np.deg2rad(5), 3, 4, 'b', 0.07)              # intralimb coord parameters
    fill_arch(angles[26]-np.deg2rad(5), angles[-2]+np.deg2rad(6), 3, 4, 'g', 0.07)              # asymmetry parameters

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], 
               ['-3.0', '-2.0', '-1.0', '0.0', '1.0', '2.0', '3.0'], color='grey', size=7)

    plt.ylim(-4, 4)
    plt.legend(loc='upper right', bbox_to_anchor=(-0.1, 0.1))
    plt.savefig(os.path.join(figures_path, 'SCI_Baseline/Spiderplot.png'), dpi=300)
    plt.close()
