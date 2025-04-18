# Quantification of lower limb kinematics during swimming in individuals with spinal cord injury

🎯 Objectives: The aim of this project is to detect and analyze swimming patterns in individuals with spinal cord injury (SCI), with the objective to characterize motor deficits and enhance personalized rehabilitation.

## Summary 

🔍 Authors: Melina Giagiozis, Sabrina Imhof, Sibylle Achermann, Catherine R. Jutzeler, László Demkó, Björn Zörner

📝 Summary: Spinal cord injuries (SCI) often result in impaired motor functions. To quantify these impairments, swimming patterns were analyzed in individuals with SCI. Water provides a unique rehabilitation environment where buoyancy supports weight-bearing activities and can reveal deficits that might otherwise go unnoticed. Data was collected of 30 individuals with chronic, motor-incomplete SCI and 20 healthy controls during breaststroke swimming on a kick-board. Using eight wearable inertial sensors attached to the lower limbs, we captured detailed kinematic data. Spatiotemporal parameters were then calculated and compared between the two groups to assess differences in swimming patterns. Analysis of the parameters revealed significant differences in velocity and distance per stroke, indicating decreased swimming speeds in individuals with SCI compared to controls. Further-more, individuals with SCI demonstrated a reduced range of motion (RoM) in the ankle and knee joints. The limited RoM likely contributes to the shorter distance covered per stroke. These observations underscore the impact of SCI on function-al capabilities. The developed algorithm holds promise for enhancing the assessment of motor deficits after neurological injuries.

🗝️ Keywords: spinal cord injury; swimming kinematics; breaststroke; lower limbs; inertial sensors; IMU

## Getting Started

First, clone this project to your local environment.

```sh
git clone https://gitlab.ethz.ch/BMDSlab/publications/sci/sci-swimming-analysis
```
Create a virtual environment with python 3.9.13.

```sh
conda create --name swim_env python=3.9.13
conda activate swim_env
```

Install python dependencies.

```sh
pip install -r requirements.txt
```

## Path Setup

In `Scripts/config.py` change the paths for data, results, and figures based on the local setup.

## Datasets Preparation

Ensure that the pre-processed datasets is in the `Data` folder located in the pre-defined path (`Scripts/config.py`). The data of a healthy participant is available.

## Extracting Swimming Parameters

To extract the swimming parameters run `Scripts/main.py`. The script will use the templates in `Data/Templates` and save them to `Results` located in the pre-defined path (`Scripts/config.py`).

If you want to extract more healthy reference data run `Scripts/extract_reference.py`. The script will access the template in `Data/Templates` and save it to `Data/Healthy_Reference_Data` located in the pre-defined path (`Scripts/config.py`).

## Validation

To validate the sensor orientations against optical motion capture (VICON, Oxford, UK) run `Scripts/src/analysis/validation.py`. Figures will be saved to `Figures` located in the pre-defined path (`Scripts/config.py`).

## Data Analysis

To perform a comprehensive analysis of the extracted swimming parameters (as included in the publication) run `Scripts/src/analysis/run_analysis.py`. This script conducts statistical analyses, visualizes the data, and applies k-means clustering for pattern identification. Figures will be saved to `Figures` located in the pre-defined path (`Scripts/config.py`).

## Contact 

📧 For comments or questions related to this repository or the manuscript contact [Melina Giagiozis](Melina.Giagiozis@balgrist.ch).

## Funding

💰 This research was funded by the Swiss National Science Foundation (#PZ00P3_186101, Jutzeler and #IZLIZ3_200275, Curt), the Swiss Paraplegic Center, and Swiss Paraplegic Research.
