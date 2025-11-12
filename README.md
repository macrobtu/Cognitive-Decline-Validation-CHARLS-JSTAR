# Cognitive Decline Validation (CHARLS–JSTAR)
This repository contains the Python scripts and processed data used for validating machine learning models predicting cognitive decline, based on the CHARLS and JSTAR cohorts.
The aim is to build and validate risk prediction models for cognitive decline using multi-year longitudinal datasets, combining traditional survival models and ensemble machine learning methods.

---
## 📂 Project Structure
data/ 
scripts/ 
output/ → model outputs and figures (ignored in .gitignore)
requirements.txt 

---
## ⚙️ Environment Setup
Python ≥ 3.9 is recommended.
```bash
# create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate
# install dependencies
pip install -r requirements.txt

Run the Analysis
After setting up the environment, run the scripts in order (scripts files);
Each script will read data from data/ and generate results under output/.

📊 Notes
The dataset provided in data/ are a processed demo files used for code testing and demonstration.
For formal research use, please refer to the official CHARLS and JSTAR data release policies.

📘 Citation
If you use or adapt this code, please cite:
Tu, L., et al. (2025). Performance of a Chinese Cognitive Decline Risk Model in a Japanese Cohort: A Validation Study.medRxiv 2025

🪶 License
MIT License
