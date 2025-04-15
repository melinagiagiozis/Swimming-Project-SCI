"""
Script: clustering.py

Description:
    This script performs a k-means clustering and analysis of extracted 
    swimming parameters.

Dependencies:
    - Swimming parameters should be stored in 'Results/Swimming_Parameters_SCI_Baseline.csv'
      and 'Results/Swimming_Parameters_Healthy_Controls.csv'.
    - Clinical data should be stored in 'Participant_Data/SCI_Baseline/Clinical_Data.csv'
      and 'Participant_Data/Healthy_Controls/Clinical_Data.csv'.
    - Swimming style categories should be defined in 
      'Participant_Data/Healthy_Controls/Swimming_Style_Healthy_Controls.csv'.
    - Figures are saved in predefined figures directory.

Usage:
    - This script is not meant to be run directly - the function is imported 
      into the main data analysis pipeline.

Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from config import *
from visualizations import *
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu


def perform_clustering():
        
    # Set up directory
    cluster_dir = os.path.join(figures_path, "Clustering")
    os.makedirs(cluster_dir, exist_ok=True)

    # Load swimming parameters
    file_path_bl = os.path.join(results_path, 'Swimming_Parameters_SCI_Baseline.csv')
    file_path_hc = os.path.join(results_path, 'Swimming_Parameters_Healthy_Controls.csv')
    df_bl = pd.read_csv(file_path_bl)
    df_hc = pd.read_csv(file_path_hc)

    # Load clinical data
    file_path_clinical_data = os.path.join(data_path, 'Participant_Data/SCI_Baseline/Clinical_Data.csv')
    file_path_clinical_data_HC = os.path.join(data_path, 'Participant_Data/Healthy_Controls/Clinical_Data.csv')
    df_clinical = pd.read_csv(file_path_clinical_data)

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

    df_bl = df_bl.groupby('Participant').mean(numeric_only=True)

    # Remove Trials
    df_bl = df_bl.drop('Trial', axis=1)

    # Maximum speed HC
    df_hc_maxspeed_grouped = df_hc_maxspeed.groupby('Participant')
    df_hc_maxspeed_means = df_hc_maxspeed_grouped.apply(lambda x: calculate_combined_mean(x, df_hc_maxspeed.columns))
    df_hc_maxspeed_means = df_hc_maxspeed_means.reset_index(drop=True)

    # Comfort speed HC
    df_hc_comfortspeed_grouped = df_hc_comfortspeed.groupby('Participant')
    df_hc_comfortspeed_means = df_hc_comfortspeed_grouped.apply(lambda x: calculate_combined_mean(x, df_hc_maxspeed.columns))
    df_hc_comfortspeed_means = df_hc_comfortspeed_means.reset_index(drop=True)

    # Standardizing the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_bl)

    # Fit PCA
    pca = PCA().fit(data_scaled)

    # Plotting the cumulative sum of explained variance
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot')
    # Determine number of PCs to explain >70%
    plt.axhline(y=0.7, color='r', linestyle='--')
    plt.grid(True)
    plt.close()

    # Initialize PCA and fit it to the scaled data
    pca = PCA(n_components=5)
    data_pca = pca.fit_transform(data_scaled)

    # Most contributing parameters in the principal components
    loadings = pca.components_

    # Iterate over each principal component
    for i, pc in enumerate(loadings):
        
        # Sort variables based on their absolute loading values
        sorted_indices = np.argsort(np.abs(pc))[::-1]
        
        # # Print the top contributing variables
        # print(f"Principal Component {i+1}:")
        # for j in sorted_indices[:5]:
        #     print(f"{df_bl.columns[j]}: = {pc[j]}")

    ks = range(1, 11)
    inertias = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=0)
        model.fit(data_pca)
        inertias.append(model.inertia_)

    # Plotting the elbow plot
    plt.figure(figsize=(8, 4))
    plt.plot(ks, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(ks)
    plt.grid(True)
    plt.close()

    # kMeans
    k = 2 
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(data_pca)

    # Add clusters back to the original data for plotting
    df_bl['Cluster'] = clusters
    df_bl['Cluster'] = df_bl['Cluster'].replace({0: 1, 1: 0}) ### Better for analysis

    df_hc_maxspeed_temp = df_hc_maxspeed_means.copy()
    df_hc_comfortspeed_temp = df_hc_comfortspeed_means.copy()

    df_hc_maxspeed = df_hc_maxspeed_temp.groupby('Participant').mean()
    df_hc_comfortspeed = df_hc_comfortspeed_temp.groupby('Participant').mean()

    # Combining data frames
    df_hc_maxspeed['Cluster'] = 'HC max'
    df_hc_comfortspeed['Cluster'] = 'HC comfort'
    df_hc_maxspeed.index = df_hc_maxspeed['Cluster']
    df_hc_comfortspeed.index = df_hc_comfortspeed['Cluster']
    df_bl.index = df_bl['Cluster']
    plot_data = pd.concat([df_bl, df_hc_comfortspeed, df_hc_maxspeed])

    # Exclude the 'Cluster' column
    unilateral_parameters = plot_data.dropna(axis=1)
    unilateral_parameters = unilateral_parameters.columns
    unilateral_parameters = unilateral_parameters.drop('Cluster')

    # Add average stroke duration 
    plot_data.loc[0, 'Stroke duration'] = np.mean(np.vstack([np.array(plot_data.loc[0, 'Stroke duration MI']), 
                                                             np.array(plot_data.loc[0, 'Stroke duration LI'])]), axis=0)
    plot_data.loc[1, 'Stroke duration'] = np.mean(np.vstack([np.array(plot_data.loc[1, 'Stroke duration MI']), 
                                                             np.array(plot_data.loc[1, 'Stroke duration LI'])]), axis=0)

    # Add average stroke duration 
    plot_data.loc[0, 'Stroke duration variability'] = np.mean(np.vstack([np.array(plot_data.loc[0, 'Stroke duration variability MI']), 
                                                             np.array(plot_data.loc[0, 'Stroke duration variability LI'])]), axis=0)
    plot_data.loc[1, 'Stroke duration variability'] = np.mean(np.vstack([np.array(plot_data.loc[1, 'Stroke duration variability MI']), 
                                                             np.array(plot_data.loc[1, 'Stroke duration variability LI'])]), axis=0)
    unilateral_parameters = unilateral_parameters.tolist() + ['Stroke duration', 'Stroke duration variability']

    # Multiple testing correction
    results_df_clusters = pd.DataFrame(columns=['Parameter', 'P-Value'])

    # Compare unilateral swimming parameters between clusters
    for parameter in unilateral_parameters:
        plt.figure(figsize=(8, 6))
        
        # Extract data for boxplot
        cluster_1_data = plot_data.loc[0, parameter]
        cluster_2_data = plot_data.loc[1, parameter]
        
        # Combine data for boxplot
        data = [cluster_1_data, cluster_2_data]
        
        # Perform Kruskal-Wallis test
        _, p_value_main = kruskal(*data)

        num_tests = len(unilateral_parameters)
        corrected_pvalue_main = min(p_value_main * num_tests, 1)  # Bonferroni correction
        new_row = pd.DataFrame([{'Parameter': parameter, 'P-Value': p_value_main, 'Corrected P-Value': corrected_pvalue_main}])
        if results_df_clusters.empty:
            results_df_clusters = new_row  # Assign directly if DataFrame is empty
        else:
            results_df_clusters = pd.concat([results_df_clusters, new_row], ignore_index=True)
    
    bilateral_parameters = ['RoM ankle (flexion/extension)', 'RoM knee (flexion/extension)', 
                            'RoM hip (flexion/extension)', 'RoM hip (abduction/adduction)',
                            'Min ankle flexion', 'Max ankle extension', 'Min knee flexion', 
                            'Max knee extension', 'Min hip flexion', 'Max hip extension', 
                            'Min hip adduction', 'Max hip abduction', 'Horizontal displacement', 
                            'Vertical displacement', 'Sidewards displacement', 'RoM knee (flexion/extension)', 
                            'RoM hip (flexion/extension)', 'RoM hip (abduction/adduction)',
                            'RoM ankle (flexion/extension)', 'ACC (hip/knee)', 'ACC std (hip/knee)', 
                            'ACC (ankle/knee)', 'ACC std (ankle/knee)', 'SSD (hip/knee)', 'SSD (ankle/knee)']

    # Compare bilateral swimming parameters (more and less impaired leg) between clusters
    for parameter in bilateral_parameters:

        # Define the data for boxplots
        data = []
        positions = []
        colors = []

        indices = ['HC max', '0_MI', '0_LI', '1_MI', '1_LI']

        data = [plot_data.loc[0, parameter + ' LI'],
                plot_data.loc[0, parameter + ' MI'],
                plot_data.loc[1, parameter + ' LI'], 
                plot_data.loc[1, parameter + ' MI']]
        positions = [1, 2, 4, 5]
        colors = ['cornflowerblue', 'crimson', 'cornflowerblue', 'crimson']

        for i in range(len(data)):
            data[i].index = [indices[i]] * len(data[i].index)

        # Perform Kruskal-Wallis test
        _, p_value_main = kruskal(*data)
        
        num_tests = len(parameters)
        corrected_pvalue_main = min(p_value_main * num_tests, 1)  # Bonferroni correction
        results_df_clusters = pd.concat([results_df_clusters, 
                                         pd.DataFrame([{'Parameter': parameter, 
                                                'P-Value': p_value_main,
                                                'Corrected P-Value': corrected_pvalue_main}])], 
                                         ignore_index=True)

        
    # Multiple testing correction
    results_df_clusters = results_df_clusters.drop_duplicates().reset_index(drop=True)
    results_df_corrected_cluster = holm_bonferroni_correction(results_df_clusters)
    significant_params_cluster = results_df_corrected_cluster.loc[results_df_corrected_cluster['Reject'] == True, 
                                                                  ['Parameter', 'Adjusted P-Value']]
    
    # Plot swimming parameters
    for parameter in significant_params_cluster['Parameter']:
        if parameter in unilateral_parameters:
            plt.figure(figsize=(8, 6))
            
            # Extract data for boxplot
            cluster_1_data = plot_data.loc[0, parameter]
            cluster_2_data = plot_data.loc[1, parameter]
            
            # Combine data for boxplot
            data = [cluster_1_data, cluster_2_data]
            
            # Plot boxplot
            bp = plt.boxplot(data, positions=[0, 1], 
                            labels=['cluster 1', 'cluster 2'], 
                            showfliers=False, patch_artist=True,
                            widths=0.25)
            
            # Adding individual points
            for i, d in enumerate(data):
                sns.stripplot(x=[[0, 1][i]] * len(d), y=d, size=7, jitter=True, color='k', alpha=0.7)

            # Remove the top and left spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Set color for the boxes
            colors = ['green', 'darkorchid']
            for box, color in zip(bp['boxes'], colors):
                box.set_facecolor(color)

            # Set color for the median lines
            for median in bp['medians']:
                median.set(color='black')

            if parameter == 'Velocity':
                plt.ylabel('Velocity [m/s]', fontsize=24)
            elif parameter == 'Stroke rate':
                plt.ylabel('Stroke rate [strokes/min]', fontsize=18)
            elif parameter == 'Phase shift std':
                plt.ylabel('Phase shift variability [%]', fontsize=24)
            elif parameter == 'Stroke duration':
                plt.ylabel('Stroke duration [s]', fontsize=18)
            elif parameter == 'Stroke duration variability':
                plt.ylabel('Stroke duration variability [s]', fontsize=24)
                plt.yticks(ticks=[0.05, 0.1, 0.15, 0.20],
                        labels=[0.05, 0.1, 0.15, 0.20])
            elif parameter == 'Distance per stroke':
                plt.ylabel('Distance per stroke [m]', fontsize=24)
            else:
                plt.ylabel(parameter, fontsize=24)

            # Adding significance indicators
            ax = plt.gca()

            # Starting height above the maximum outlier
            _, ymax = plt.ylim()
            base_y = ymax * 1.01
            step_h = ymax * 0.05 
            positions = [0, 1]

            # Calculate the corrected significance level
            alpha = 0.05
            num_comparisons = len(data) * (len(data) - 1) / 2  # Number of pairwise comparisons
            alpha_corrected = alpha / num_comparisons

            # Calculate and annotate p-values between relevant groups
            for i in range(len(data) - 1, 0, -1):
                for j in range(i - 1, -1, -1):
                    p_value = mannwhitneyu(data[i], data[j]).pvalue
                    if p_value < alpha_corrected:
                        if parameter == 'Phase shift':
                            sig = '*'
                        else:
                            if p_value < 0.001:
                                sig = '***'
                            elif p_value < 0.01:
                                sig = '**'
                            elif p_value < 0.05:
                                sig = '*'
                        y = base_y + step_h
                        ax.annotate(sig, ((positions[i] + positions[j]) / 2, y), ha='center', va='bottom', fontsize=18)
                        ax.plot([positions[i], positions[j]], [y, y], lw=1, c='black')
                        base_y = y 
            
            # Titles and labels, modify as necessary
            plt.xticks(rotation=0, fontsize=24)
            plt.yticks(fontsize=22)
            plt.tight_layout()
            plt.xlim(-0.4, 1.4)
            plt.savefig(os.path.join(figures_path, 'Clustering/' + parameter.replace('/', '_').replace(' ', '_') + '.png'), dpi=300)
            plt.close()

    # Plot swimming parameters (more and less impaired leg) between clusters
    for parameter in significant_params_cluster['Parameter']:
        if parameter in bilateral_parameters:
            plt.figure(figsize=(8, 6))

            # Define the data for boxplots
            data = []
            positions = []
            colors = []

            indices = ['HC max', '0_MI', '0_LI', '1_MI', '1_LI']

            data = [plot_data.loc[0, parameter + ' LI'],
                    plot_data.loc[0, parameter + ' MI'],
                    plot_data.loc[1, parameter + ' LI'], 
                    plot_data.loc[1, parameter + ' MI']]
            positions = [1, 2, 4, 5]
            colors = ['cornflowerblue', 'crimson', 'cornflowerblue', 'crimson']

            for i in range(len(data)):
                data[i].index = [indices[i]] * len(data[i].index)

            # Plot boxplots
            bp = plt.boxplot(data, positions=positions, patch_artist=True, widths=0.6, showfliers=False)

            # Remove the top and left spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Customize boxplot appearance
            for box, color in zip(bp['boxes'], colors):
                box.set_facecolor(color)
            for median in bp['medians']:
                median.set(color='black')

            for i, (pos, values) in enumerate(zip(positions, data)):        
                scatter_x = np.random.normal(loc=pos, scale=0.1, size=len(values))
                plt.scatter(scatter_x, values, s=35, color='black', alpha=0.7, edgecolors='none', zorder=2)

            # Adding significance indicators
            ax = plt.gca()
            _, ymax = plt.ylim()
            base_y = ymax * 1.01

            # Increment for each annotation to avoid overlap
            if parameter.startswith('ACC'):
                step_h = 0.012
            elif parameter.startswith('SSD (hip'):
                step_h = 1  # Works if it's just one *
            else:
                step_h = ymax * 0.05

            # Collect p-values and comparison pairs
            p_values = []
            comparisons = []

            for i in range(len(data) - 1, 0, -1):
                for j in range(i - 1, -1, -1):
                    p_value = mannwhitneyu(data[i], data[j]).pvalue
                    p_values.append(p_value)
                    comparisons.append((i, j))  # Store index pairs for reference

            # Apply Holm-Bonferroni correction
            _, corrected_pvals, _, _ = multipletests(p_values, alpha=0.05, method='holm')

            # Annotate significant results
            current_y_offset = 0
            for (i, j), p_val in zip(comparisons, corrected_pvals):
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    continue  # Skip non-significant results

                y = base_y + step_h
                if parameter.startswith('ACC'):
                    ax.annotate(sig, ((positions[i] + positions[j]) / 2, y * 0.996), ha='center', va='bottom', fontsize=18)
                else:
                    ax.annotate(sig, ((positions[i] + positions[j]) / 2, y * 0.985), ha='center', va='bottom', fontsize=18)
                
                ax.plot([positions[i], positions[j]], [y, y], lw=0.8, c='black')
                base_y = y  

            # Set titles and labels
            if parameter.startswith('RoM ankle'):
                plt.ylabel('Ankle flexion RoM [deg]', size=24)
            elif parameter.startswith('RoM knee'):
                plt.ylabel('Knee flexion RoM [deg]', size=24)
            elif parameter.startswith('RoM hip (flexion/extension)'):
                plt.ylabel('Hip flexion RoM [deg]', size=18)
            elif parameter.startswith('RoM hip (abduction/adduction)'):
                plt.ylabel('Hip abduction RoM [deg]', size=18)
            elif parameter.startswith('ACC (ankle'):
                plt.ylabel('ACC (ankle-knee)', size=24)
                plt.ylim(0.79, 1.08)
                plt.yticks(ticks=[0.8, 0.9, 1.0])
            elif parameter.startswith('ACC (hip'):
                plt.ylabel('ACC (knee-hip)', size=24)
                plt.ylim(0.79, 1.08)
                plt.yticks(ticks=[0.8, 0.9, 1.0])
            elif parameter.startswith('SSD (ankle'):
                plt.ylabel('SSD (ankle-knee)', size=18)
                plt.ylim(0, 6)
            elif parameter.startswith('SSD (hip'):
                plt.ylabel('SSD (knee-hip)', size=18)
                plt.ylim(0, 6)
                legend_labels = ['less impaired leg', 'more impaired leg']
                legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, legend_labels)]
                plt.legend(handles=legend_patches)
            elif parameter == 'Phase shift':
                plt.ylabel('Phase shift [%]', size=18)
            else:
                plt.title(parameter, size=18)

            plt.xticks([1.5, 4.5], ['cluster 1', 'cluster 2'], fontsize=24)
            plt.yticks(fontsize=22)

            # Save the figure
            plt.savefig(os.path.join(figures_path, 'Clustering/' + parameter.replace('/', '_').replace(' ', '_') + '.png'), dpi=300)
            plt.close()



    # Swimming profiles per cluster (spiderlplots)
    df_cluster_1 = df_bl[df_bl['Cluster'] == 0]
    df_cluster_2 = df_bl[df_bl['Cluster'] == 1]

    df_cluster_1 = df_cluster_1.drop(columns=['Cluster'])
    df_cluster_2 = df_cluster_2.drop(columns=['Cluster'])

    cluster_1_li_data = df_cluster_1.loc[:, ~df_cluster_1.columns.str.endswith('MI')].reset_index()
    cluster_2_li_data = df_cluster_2.loc[:, ~df_cluster_2.columns.str.endswith('MI')].reset_index()

    cluster_1_li_data = cluster_1_li_data.drop(columns=['Cluster'])
    cluster_2_li_data = cluster_2_li_data.drop(columns=['Cluster'])

    cluster_1_li_data.columns = cluster_1_li_data.columns.str.replace(' LI$', '', regex=True)
    cluster_2_li_data.columns = cluster_2_li_data.columns.str.replace(' LI$', '', regex=True)

    cluster_1_mi_data = df_cluster_1.loc[:, ~df_cluster_1.columns.str.endswith('LI')].reset_index()
    cluster_2_mi_data = df_cluster_2.loc[:, ~df_cluster_2.columns.str.endswith('LI')].reset_index()

    cluster_1_mi_data = cluster_1_mi_data.drop(columns=['Cluster'])
    cluster_2_mi_data = cluster_2_mi_data.drop(columns=['Cluster'])

    cluster_1_mi_data.columns = cluster_1_mi_data.columns.str.replace(' MI$', '', regex=True)
    cluster_2_mi_data.columns = cluster_2_mi_data.columns.str.replace(' MI$', '', regex=True)
    df_hc_maxspeed_means.columns = df_hc_maxspeed_means.columns.str.replace(r' Mean$', '', regex=True)
    
    cluster_1_li_data = cluster_1_li_data.reindex(columns=df_hc_maxspeed_means.columns)
    cluster_1_mi_data = cluster_1_mi_data.reindex(columns=df_hc_maxspeed_means.columns)
    cluster_2_li_data = cluster_2_li_data.reindex(columns=df_hc_maxspeed_means.columns)
    cluster_2_mi_data = cluster_2_mi_data.reindex(columns=df_hc_maxspeed_means.columns)

    # Adjust column names
    for df in [cluster_1_li_data, cluster_1_mi_data, cluster_2_li_data, cluster_2_mi_data, df_hc_maxspeed_means]:
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
        df.rename(columns={'Phase shift std': 'Phase shift variability'}, inplace=True)

    # Calculate z-scores 
    z_scores_cluster_1_li = calculate_z_scores(cluster_1_li_data, df_hc_maxspeed_means)
    z_scores_cluster_1_mi = calculate_z_scores(cluster_1_mi_data, df_hc_maxspeed_means)
    z_scores_cluster_2_li = calculate_z_scores(cluster_2_li_data, df_hc_maxspeed_means)
    z_scores_cluster_2_mi = calculate_z_scores(cluster_2_mi_data, df_hc_maxspeed_means)
    z_scores_hc_maxspeed = calculate_z_scores(df_hc_maxspeed_means, df_hc_maxspeed_means)

    z_scores_cluster_1_li_std = z_scores_cluster_1_li.std()
    z_scores_cluster_1_mi_std = z_scores_cluster_1_mi.std()
    z_scores_cluster_2_li_std = z_scores_cluster_2_li.std()
    z_scores_cluster_2_mi_std = z_scores_cluster_2_mi.std()
    z_scores_hc_maxspeed_std = z_scores_hc_maxspeed.std()

    z_scores_cluster_1_li = z_scores_cluster_1_li.mean()
    z_scores_cluster_1_mi = z_scores_cluster_1_mi.mean()
    z_scores_cluster_2_li = z_scores_cluster_2_li.mean()
    z_scores_cluster_2_mi = z_scores_cluster_2_mi.mean()
    z_scores_hc_maxspeed = z_scores_hc_maxspeed.mean()

    all_z_scores = pd.concat([z_scores_hc_maxspeed, z_scores_cluster_1_li, z_scores_cluster_1_mi, 
                              z_scores_cluster_2_li, z_scores_cluster_2_mi], axis=1)
    all_z_scores.columns = ['healthy range', 'cluster 1: less impaired leg', 'cluster 1: more impaired leg', 
                            'cluster 2: less impaired leg', 'cluster 2: more impaired leg']

    all_z_scores_std = pd.concat([z_scores_hc_maxspeed_std, z_scores_cluster_1_li_std, z_scores_cluster_1_mi_std, 
                                  z_scores_cluster_2_li_std, z_scores_cluster_2_mi_std], axis=1)
    all_z_scores_std.columns = ['healthy range', 'cluster 1: less impaired leg', 'cluster 1: more impaired leg', 
                                'cluster 2: less impaired leg', 'cluster 2: more impaired leg']

    # Create a radar chart
    labels = all_z_scores.index
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    #More impaired leg
    _, ax = plt.subplots(figsize=(15, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(0)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, size=16)

    colors = ['gray', 'darkgreen', 'darkorchid']
    columns = ['healthy range', 'cluster 1: more impaired leg', 'cluster 2: more impaired leg']
    legend_labels = ['healthy range', 'cluster 1', 'cluster 2']
    ha = 8 * ['left'] + ['center'] + 16 * ['right'] + ['center'] + 7 * ['left']

    for i, column in enumerate(columns):
        values = all_z_scores[column].tolist() + [all_z_scores[column].tolist()[0]]
        ax.plot(angles, values, label=legend_labels[i], color=colors[i])
        std_values = all_z_scores_std[column].tolist() + [all_z_scores_std[column].tolist()[0]]
        lower_bound = np.maximum(np.subtract(values, std_values), -5)
        upper_bound = np.minimum(np.add(values, std_values), 5)
        ax.fill_between(angles, lower_bound, upper_bound, color=colors[i], alpha=0.15)
        
    for tick, label in zip(ax.get_xticklabels(), ha):
        tick.set_horizontalalignment(label)

    # Move the second label slightly to the left
    x, y = ax.get_xticklabels()[8].get_position()
    ax.get_xticklabels()[8].set_position((x, y-0.11))
    ax.get_xticklabels()[25].set_position((x, y-0.11))

    x, y = ax.get_xticklabels()[7].get_position()
    ax.get_xticklabels()[7].set_position((x, y-0.041))
    ax.get_xticklabels()[26].set_position((x, y-0.041))

    x, y = ax.get_xticklabels()[9].get_position()
    ax.get_xticklabels()[9].set_position((x, y-0.041))
    ax.get_xticklabels()[24].set_position((x, y-0.041))

    # Function to fill an arch segment
    def fill_arch(start_angle, end_angle, lower, upper, arch_color, alpha):
        theta = np.linspace(start_angle, end_angle, 100)
        r_lower = np.full_like(theta, lower)
        r_upper = np.full_like(theta, upper)
        ax.fill_between(theta, r_lower, r_upper, color=arch_color, alpha=alpha)

    # Categorize parameters
    fill_arch(angles[0]-np.deg2rad(5), angles[5]-np.deg2rad(5), 4, 5, 'orange', 0.07)           # speed parameters
    fill_arch(angles[5]-np.deg2rad(5), angles[17]-np.deg2rad(5), 4, 5, 'purple', 0.07)          # joint angle parameters
    fill_arch(angles[17]-np.deg2rad(5), angles[20]-np.deg2rad(5), 4, 5, 'yellow', 0.07)         # displacement parameters
    fill_arch(angles[20]-np.deg2rad(5), angles[26]-np.deg2rad(5), 4, 5, 'b', 0.07)              # intralimb coord parameters
    fill_arch(angles[26]-np.deg2rad(5), angles[-2]+np.deg2rad(6), 4, 5, 'g', 0.07)              # asymmetry parameters

    ax.set_rlabel_position(0)
    plt.yticks([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], color='k', size=12)
    plt.ylim(-5, 5)
    # plt.legend(loc='upper right', bbox_to_anchor=(-0.1, 0.1))
    plt.title('More impaired leg', y=1.15, size=24)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'Clustering/Spiderplot_Clusters_MI.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # Less impaired leg
    _, ax = plt.subplots(figsize=(15, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(0)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, size=16)

    colors = ['gray', 'darkgreen', 'darkorchid']
    columns = ['healthy range', 'cluster 1: less impaired leg', 'cluster 2: less impaired leg']
    legend_labels = ['healthy range', 'cluster 1', 'cluster 2']
    ha = 8 * ['left'] + ['center'] + 16 * ['right'] + ['center'] + 7 * ['left']

    for i, column in enumerate(columns):
        values = all_z_scores[column].tolist() + [all_z_scores[column].tolist()[0]]
        ax.plot(angles, values, label=legend_labels[i], color=colors[i])
        std_values = all_z_scores_std[column].tolist() + [all_z_scores_std[column].tolist()[0]]
        ax.fill_between(angles, np.subtract(values, std_values), np.add(values, std_values), color=colors[i], alpha=0.15)

    for tick, label in zip(ax.get_xticklabels(), ha):
        tick.set_horizontalalignment(label)

    # Move the second label slightly to the left
    x, y = ax.get_xticklabels()[8].get_position()
    ax.get_xticklabels()[8].set_position((x, y-0.11))
    ax.get_xticklabels()[25].set_position((x, y-0.1))

    x, y = ax.get_xticklabels()[7].get_position()
    ax.get_xticklabels()[7].set_position((x, y-0.041))
    ax.get_xticklabels()[26].set_position((x, y-0.041))

    x, y = ax.get_xticklabels()[9].get_position()
    ax.get_xticklabels()[9].set_position((x, y-0.041))
    ax.get_xticklabels()[24].set_position((x, y-0.041))

    # Categorize parameters
    fill_arch(angles[0]-np.deg2rad(5), angles[5]-np.deg2rad(5), 4, 5, 'orange', 0.07)           # speed parameters
    fill_arch(angles[5]-np.deg2rad(5), angles[17]-np.deg2rad(5), 4, 5, 'purple', 0.07)          # joint angle parameters
    fill_arch(angles[17]-np.deg2rad(5), angles[20]-np.deg2rad(5), 4, 5, 'yellow', 0.07)         # displacement parameters
    fill_arch(angles[20]-np.deg2rad(5), angles[26]-np.deg2rad(5), 4, 5, 'b', 0.07)              # intralimb coord parameters
    fill_arch(angles[26]-np.deg2rad(5), angles[-2]+np.deg2rad(6), 4, 5, 'g', 0.07)              # asymmetry parameters

    ax.set_rlabel_position(0)
    plt.yticks([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], color='k', size=12)
    plt.ylim(-5, 5)
    plt.legend(loc='upper right', bbox_to_anchor=(-0.2, 0.1), fontsize=18)
    plt.title('Less impaired leg', y=1.15, size=24)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'Clustering/Spiderplot_Clusters_LI.png'), dpi=300, bbox_inches='tight')
    plt.close()




    # # Demographic analysis of clusters

    # # Filter secondary parameters for clusters
    # cluster_1_data = df_clinical[df_clinical['ID'].isin(cluster_1)]
    # cluster_2_data = df_clinical[df_clinical['ID'].isin(cluster_2)]

    # # DEMOGRAPHIC CLUSTER ANALYSIS
    # for cluster, data in [(1, cluster_1_data), (2, cluster_2_data)]:
    #     print(f'Cluster {cluster} - DEMOGRAPHIC ANALYSIS:')
    #     print('SCI Age: ',  np.mean(data['Age']), '+/-', np.std(data['Age']))
    #     print('SCI Sex: ', data['Gender'].value_counts())
    #     print('SCI BMI: ', np.mean(data['BMI']), '+/-', np.std(data['BMI']))
    #     print('SCI NLI: ', data['Level_of_injury'].value_counts())
    #     print('AIS_A-E: ', data['AIS_A-E'].value_counts())
    #     print('SCI Aetiology: ', data['Aetiology'].value_counts())
    #     print('SCI SCIM Mobility: ', np.mean(data['S_SCIM']), '+/-', np.std(data['S_SCIM']))
    #     print()

    # demographic_variables = ['Age', 'BMI', 'S_SCIM', 'Gender_Binary',
    #                          'Level_of_injury_Binary', 'AIS_A-E_Binary',
    #                          'Aetiology_Binary']

    # # Make variables binary
    # cluster_1_data.loc[:, 'Level_of_injury_Binary'] = np.where(cluster_1_data.loc[:, 'Level_of_injury'].str.startswith('T'), 1, 0)
    # cluster_2_data.loc[:, 'Level_of_injury_Binary'] = np.where(cluster_2_data.loc[:, 'Level_of_injury'].str.startswith('T'), 1, 0)
    # cluster_1_data.loc[:, 'AIS_A-E_Binary'] = np.where(cluster_1_data.loc[:, 'AIS_A-E'].isin(['D']), 1, 0)
    # cluster_2_data.loc[:, 'AIS_A-E_Binary'] = np.where(cluster_2_data.loc[:, 'AIS_A-E'].isin(['D']), 1, 0)
    # cluster_1_data.loc[:, 'Aetiology_Binary'] = np.where(cluster_1_data.loc[:, 'Aetiology'] != 'traumatic', 1, 0)
    # cluster_2_data.loc[:, 'Aetiology_Binary'] = np.where(cluster_2_data.loc[:, 'Aetiology'] != 'traumatic', 1, 0)
    # cluster_1_data.loc[:, 'Gender_Binary'] = np.where(cluster_1_data.loc[:, 'Gender'] == 'F', 1, 0)
    # cluster_2_data.loc[:, 'Gender_Binary'] = np.where(cluster_2_data.loc[:, 'Gender'] == 'F', 1, 0)

    # # Reset indices
    # cluster_1_data.reset_index(drop=True, inplace=True)
    # cluster_2_data.reset_index(drop=True, inplace=True)

    # # Loop through each demographic variable
    # for variable in demographic_variables:

    #     # Extract data for each cluster
    #     cluster_1_values = cluster_1_data.loc[:, variable]
    #     cluster_2_values = cluster_2_data.loc[:, variable]


    #     # Check if there are data points in both clusters
    #     if not cluster_1_values.empty and not cluster_2_values.empty:
    #         # Perform the appropriate statistical test based on the variable type
    #         if variable in ['Gender_Binary', 'Level_of_injury_Binary', 'Aetiology_Binary']:  # exlude AIS_A-E_Binary as 100% are D
    #             combined_values = pd.concat([cluster_1_values, cluster_2_values], axis=0)
                
    #             # Create a DataFrame with the combined values and a new column indicating the cluster
    #             contingency_df = pd.DataFrame({'Values': combined_values, 'Cluster': ['Cluster 1'] * len(cluster_1_values) + ['Cluster 2'] * len(cluster_2_values)})
                
    #             # Create a contingency table from the DataFrame
    #             contingency_table = pd.crosstab(contingency_df['Cluster'], contingency_df['Values'])
                
    #             # Perform Fisher's exact test
    #             odds_ratio, p_value = fisher_exact(contingency_table)
                
    #             # Print the results
    #             print(f"Fisher's exact test for {variable}:")
    #             print(f"p-value: {p_value}")
    #             print()

    #         else:
    #             # For numerical variables like Age, BMI, and S_SCIM, perform the appropriate test
    #             # Check for normality
    #             if variable not in ['Gender_Binary', 'Level_of_injury_Binary', 'AIS_A-E_Binary', 'Aetiology_Binary']:
    #                 u_stat, p_value = mannwhitneyu(cluster_1_values, cluster_2_values, alternative='two-sided')
    #                 test_type = 'Mann-Whitney U test'

    #                 # Print the results
    #                 print(f'{test_type} test for {variable}:')
    #                 print(f'p-value: {p_value}')
    #                 print()
    #     else:
    #         print(f'No data available for {variable} in one or both clusters.')


    # # Define parameters for analysis
    # parameters = ['Gender', 'Level_of_injury_Binary', 'AIS_A-E', 'Aetiology_Binary']
    # parameter_names = ['gender', 'NLI', 'AIS', 'aetiology']
    