# Performance of a Chinese Cognitive Decline Risk Model in a Japanese Cohort: A Validation Study

This repository contains the analysis code and processed data for the study titled "Performance of a Chinese Cognitive Decline Risk Model in a Japanese Cohort: A Validation Study".

## Project Overview

This study develops a cognitive decline risk prediction model using the China Health and Retirement Longitudinal Study (CHARLS) and performs a rigorous internal and external validation in a later CHARLS wave and the Japanese Study of Aging and Retirement (JSTAR) cohort, respectively.

## Repository Structure

- `/data`: Contains the sample data for the analysis. Full dataset can be found online:https://g2aging.org/ 
- `/scripts`: Contains the Python scripts to reproduce the analysis.

## How to Run

The analysis is conducted in several steps. The scripts are numbered in the recommended order of execution:

1.  `01_feature_selection.py`: Script for selecting the final predictor variables.
2.  `02_model_benchmarking.py`: Script for comparing multiple ML algorithms to select the final model type.
3.  `03_data_harmonization.py`: Script for aligning all variables across the three cohorts.
4.  `04_main_validation_analysis.py`: Script to run the primary analysis (MI, validation, figure/table generation).
5.  `05_sensitivity_analyses.py`: Script to run the two sensitivity analyses.

## Dependencies

This analysis requires Python 3 and the following major libraries:
- pandas
- numpy
- scikit-learn
- lifelines
- sksurv
- dcurves
- tableone
- matplotlib

## Citation

If you use this code or data in your research, please cite our paper:
[在这里填入您未来发表论文的引用信息]
