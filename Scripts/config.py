"""
Script: config.py

Description:
    This script defines the absolute paths for data storage, results and figures output. 

Usage:
    Modify the `data_path`, `results_path`, and `figures_path` variables as needed 
    to suit your local setup.

Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

import os

# Path to where the data is stored locally
data_path = '/Volumes/green_groups_bds_jumpusers/melina/1_Swimming/_Publication/Swimming-Project-SCI/Data'
os.makedirs(data_path, exist_ok=True)

# Path to where the results will be stored locally
results_path = '/Volumes/green_groups_bds_jumpusers/melina/1_Swimming/_Publication/Swimming-Project-SCI/Results'
os.makedirs(results_path, exist_ok=True)

# Path to where the figures will be stored locally
figures_path = '/Volumes/green_groups_bds_jumpusers/melina/1_Swimming/_Publication/Swimming-Project-SCI/Figures'
os.makedirs(figures_path, exist_ok=True)