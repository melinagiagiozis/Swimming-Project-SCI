"""
Script: run_analysis.py

Description:
    This script performs the data analysis included in the publication.

Usage:
    Run the script in a Python environment where all dependencies are installed 
    (see requirements.txt). Ensure the 'Results' directory contains extracted 
    swimming parameters and joint angles.

Author: Melina Giagiozis (Melina.Giagiozis@balgrist.ch)
Date: 2025-02-19
"""

from swimming_parameters import analyze_swimming_parameters
from clustering import perform_clustering
from joint_angles import analyze_joint_angles

def main():
    # Perform data analysis steps
    analyze_swimming_parameters()
    perform_clustering()
    analyze_joint_angles()
    
    # Handle or save results if necessary
    print("Analysis complete.")

if __name__ == "__main__":
    main()
