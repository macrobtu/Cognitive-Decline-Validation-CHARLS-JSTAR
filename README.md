# Cognitive Decline Validation (CHARLS–JSTAR)

This repository contains the validation pipeline for predicting cognitive decline across different aging cohorts.

## 📂 Research Scripts
The analysis follows a sequential workflow:
* **01_model_comparison.py**: Compares different machine learning models (GMM, K-Means).
* **02_CoxPH_model_performance.py**: Evaluates core model metrics on the development set.
* **03_charls_validation.py**: Temporal validation using the **CHARLS** dataset (2015–2018).
* **04_jstar_validation.py**: External validation using the **JSTAR** dataset.
* **05_sensitivity_analyses.py**: Robustness checks for 2-year vs. 4-year horizons.

---

## ⚠️ Data Disclaimer

### JSTAR Data (Privacy & Compliance)
Due to the strict privacy policies of the **Japan Study on Aging and Retirement (JSTAR)**, raw data cannot be publicly shared. 
* **Synthetic Sample**: The `jstar` CSV files provided in the `data/` folder are **randomized sample files** for code demonstration only.
* **Result Note**: Running script `04` with this sample data **will not produce valid results or the figures** shown in the publication due to intentional data randomization. 
* **Official Access**: To obtain the original data, please apply through the [JSTAR official portal](https://www.rieti.go.jp/en/projects/jstar/).

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
