"""
Project: Model Sensitivity Analysis (2-year vs. 4-year ROC/calibration plot)
Author: macrobtu
Date: February 2026
Description: This script evaluates the robustness of the risk model across different 
time horizons (2 years and 4 years) using the development cohort (CHARLS).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib as mpl

# --- Font Configuration ---
try:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Verdana']
    mpl.rcParams['axes.unicode_minus'] = False 
except Exception as e:
    print(f"Font setting warning: {e}")

# --- Relative Path Configuration ---
try:
    # Set base directory to project root
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File Paths
TRAIN_PATH = DATA_DIR / "charls_model_ready.csv"
LIFELINES_SUMMARY_PATH = OUTPUT_DIR / "table3_Final_CoxPH_Model_Detailed_Summary_lifelines_80Train_IterImp.xlsx"

# --- 1. Variable Definitions & Preprocessing Rules ---
num_features = ['orient', 'memory', 'grip_max']
cat_features = ['sex', 'edu_3cat', 'age_group2_safe']
all_features = num_features + cat_features
TIME_COL, EVENT_COL = "surv_time", "event"

high_missing_continuous = ['grip_max']
low_missing_continuous = [col for col in num_features if col not in high_missing_continuous]

# Preprocessing Pipeline (Must remain consistent with training script)
low_miss_num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])
high_miss_num_pipeline = Pipeline([
    ("imputer", IterativeImputer(estimator=BayesianRidge(), max_iter=10, 
                                 random_state=42, imputation_order='ascending')),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')), 
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("low_miss_num", low_miss_num_pipeline, low_missing_continuous),
        ("high_miss_num", high_miss_num_pipeline, high_missing_continuous),
        ("cat", cat_pipeline, cat_features)
    ],
    remainder='drop'
)

# --- 2. Load Model Coefficients & Training Data ---
try:
    # Load coefficients from trained lifelines model
    lifelines_summary = pd.read_excel(LIFELINES_SUMMARY_PATH, index_col=0)
    sig_vars = lifelines_summary[lifelines_summary['p'] < 0.05].copy()
    if sig_vars.empty:
        raise ValueError("No significant variables found (p < 0.05).")

    # Use raw logHR coefficients for precise risk score calculation
    risk_scorecard_coef = sig_vars[['coef']] 
    ll_names = risk_scorecard_coef.index.tolist()

    # Load Training Dataset
    df_train_source = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    df_train_source = df_train_source.dropna(subset=[TIME_COL, EVENT_COL]).copy()
    X_train = df_train_source[all_features].copy()
    df_train_source['time'] = df_train_source[TIME_COL]
    df_train_source['event'] = df_train_source[EVENT_COL].astype(bool)

    # Fit preprocessor and transform data
    preprocess.fit(X_train) 
    actual_feature_names = preprocess.get_feature_names_out()
    X_train_processed_df = pd.DataFrame(preprocess.transform(X_train), columns=actual_feature_names, index=X_train.index)

    # Map lifelines feature names to preprocessor output names
    name_mapping = {}
    for ll_n in ll_names:
        if ll_n in actual_feature_names:
            name_mapping[ll_n] = ll_n
        else:
            base = ll_n.split('__')[-1]
            matches = [act for act in actual_feature_names if act.endswith(base)]
            if len(matches) == 1:
                name_mapping[ll_n] = matches[0]
            else:
                raise ValueError(f"Feature mapping failed for: {ll_n}")

    # Calculate Total Risk Score (logHR) for all individuals
    score_cols = [name_mapping[ll_n] for ll_n in ll_names]
    coef_vector = risk_scorecard_coef.loc[ll_names, 'coef'].values
    df_for_analysis = df_train_source.copy() 
    df_for_analysis['Total_Risk_Score'] = X_train_processed_df[score_cols].dot(coef_vector)

except Exception as e:
    print(f"Initialization failed: {e}")
    traceback.print_exc()
    exit()

# --- 3. Robust Time-Dependent ROC Analysis (Landmark Filtering) ---

def calculate_auc_at_time(df, time_point, score_col, label_name):
    """
    Logic:
    - Case (1): Experienced event within 'time_point'.
    - Control (0): Survived past 'time_point' (at risk).
    - Exclude: Censored before 'time_point' without event (unknown status).
    """
    temp = df.copy()
    is_case = (temp['event'] == 1) & (temp['time'] <= time_point)
    is_control = (temp['time'] > time_point)
    
    temp.loc[is_case, 'target_label'] = 1
    temp.loc[is_control, 'target_label'] = 0
    
    # Drop unknown/censored cases for landmark analysis
    subset = temp.dropna(subset=['target_label'])
    
    y_true = subset['target_label']
    y_scores = subset[score_col]
    
    auc_val = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    print(f"[{label_name}] Sample Size: {len(subset)} (Excluded {len(temp)-len(subset)} censored cases)")
    return auc_val, fpr, tpr

# Execute calculations for 2-year and 4-year horizons
auc_2, fpr_2, tpr_2 = calculate_auc_at_time(df_for_analysis, 2.0, 'Total_Risk_Score', "2-Year")
auc_4, fpr_4, tpr_4 = calculate_auc_at_time(df_for_analysis, 4.0, 'Total_Risk_Score', "4-Year")

# --- 4. Plotting Comparison ---
plt.figure(figsize=(8, 8))
palette = sns.color_palette("colorblind")

plt.plot(fpr_2, tpr_2, color=palette[1], lw=2.5, label=f'2-Year Horizon (AUC = {auc_2:.3f})')
plt.plot(fpr_4, tpr_4, color=palette[0], lw=2.5, label=f'4-Year Horizon (AUC = {auc_4:.3f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', alpha=0.6)

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.tick_params(labelsize=15)
plt.xlabel('1 - Specificity', fontsize=20)
plt.ylabel('Sensitivity', fontsize=20)
plt.title('2-Year vs 4-Year Sensitivity Analysis', fontsize=22)
plt.legend(loc="lower right", fontsize=15)

# Style borders
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)

# Save Results
out_path = OUTPUT_DIR / ""
"FigS4A_CHARLS_Robust_Sensitivity_Analysis.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()

# --- Summary Recommendation for Discussion ---
diff = abs(auc_2 - auc_4)
print(f"\nAnalysis Result: Delta-AUC = {diff:.3f}")
if diff < 0.03:
    print("Interpretation: Consistent discriminative power across horizons indicates model robustness.")


# %% [markdown]
# ### Calibration Sensitivity Analysis (4-year vs. 2-year)
# Rationale: To evaluate the stability of absolute risk estimates on the 
# training set across different time horizons using a calibration plot with inset.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

# --- Relative Path Configuration ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATH_TRAIN = DATA_DIR / "charls_model_ready.csv"

# --- Feature Definitions ---
num_features = ['orient', 'memory', 'grip_max']
cat_features = ['sex', 'edu_3cat', 'age_group2_safe']
all_features = num_features + cat_features
TIME_COL, EVENT_COL = "surv_time", "event"

# %% [markdown]
# ### 1. Calibration Calculation Function

# %%
def get_calib_data(df, time_horizon, name, model):
    """
    Calculates calibration metrics (Slope, Intercept, Brier Score) 
    using a landmark filtering approach.
    """
    # 1. Predict risk probability [0, 1]
    surv_funcs = model.predict_survival_function(df[all_features])
    risk_probs = np.array([1 - fn(time_horizon) for fn in surv_funcs]) 
    
    # 2. Landmark Filtering
    temp = df.copy()
    temp['risk_prob'] = risk_probs
    temp['is_case'] = (temp[EVENT_COL] == 1) & (temp[TIME_COL] <= time_horizon)
    temp['is_control'] = (temp[TIME_COL] > time_horizon)
    
    # Extract valid subset for the specific horizon
    valid_mask = temp['is_case'] | temp['is_control'] 
    subset = temp[valid_mask].copy() 
    
    y_true = subset['is_case'].astype(int)
    y_prob = subset['risk_prob']
    
    # 3. Calibration Curve Points
    prob_true_pts, prob_pred_pts = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    
    # 4. Slope, Intercept & Brier Score
    lr = LinearRegression().fit(prob_pred_pts.reshape(-1, 1), prob_true_pts)
    brier = brier_score_loss(y_true, y_prob)
    
    print(f"[{name}] n={len(subset)}, Slope={lr.coef_[0]:.3f}, Int={lr.intercept_:.3f}, Brier={brier:.4f}")

    return {
        'name': name,
        'true_pts': prob_true_pts,
        'pred_pts': prob_pred_pts,
        'slope': lr.coef_[0],
        'intercept': lr.intercept_,
        'brier': brier
    }

# %% [markdown]
# ### 2. Main Analysis and Plotting

# %%
def run_calibration_sensitivity():
    print("\n--- Running Training Set Calibration Sensitivity Analysis ---")
    
    # Load and fit model (development cohort)
    try:
        df_train = pd.read_csv(PATH_TRAIN, encoding="utf-8-sig")
        X_train = df_train[all_features]
        y_train = Surv.from_arrays(event=df_train[EVENT_COL].astype(bool), time=df_train[TIME_COL])
        
        # Pipeline Definition
        preprocess = ColumnTransformer(transformers=[
            ("low_miss_num", Pipeline([("imp", SimpleImputer(strategy='median')), ("sc", StandardScaler())]), ['orient', 'memory']),
            ("high_miss_num", Pipeline([("imp", IterativeImputer(random_state=42)), ("sc", StandardScaler())]), ['grip_max']),
            ("cat", Pipeline([("imp", SimpleImputer(strategy='most_frequent')), ("oh", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))]), cat_features)
        ])
        
        model_pipeline = make_pipeline(preprocess, CoxPHSurvivalAnalysis(ties="efron"))
        model_pipeline.fit(X_train, y_train)
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Execute calculations for different horizons
    res_4yr = get_calib_data(df_train, 4.0, "4-Year Horizon", model_pipeline)
    res_2yr = get_calib_data(df_train, 2.0, "2-Year Horizon", model_pipeline)

    # Plotting
    plt.figure(figsize=(8, 8))
    palette = sns.color_palette("colorblind")
    
    # Main Axes: 4-Year Calibration
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Perfectly Calibrated")
    
    label_4 = f"4Y (Slope={res_4yr['slope']:.2f}, Int={res_4yr['intercept']:.2f})"
    ax.plot(res_4yr['pred_pts'], res_4yr['true_pts'], marker='o', markersize=7,
            color=palette[0], lw=2.5, label=label_4)

    ax.set_xlabel("Mean Predicted Probability", fontsize=20)
    ax.set_ylabel("Observed Fraction (Positives)", fontsize=20)
    ax.set_title("Calibration Sensitivity (Training Set)", fontsize=22, pad=12)

    # Scale main plot to 4Y data
    limit_main = max(res_4yr['true_pts'].max(), res_4yr['pred_pts'].max()) + 0.05
    ax.set_xlim(0, limit_main)
    ax.set_ylim(0, limit_main)
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Inset: Zoomed 2-Year Calibration (0-0.08 range)
    ax_ins = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.5)
    ax_ins.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax_ins.plot(res_2yr['pred_pts'], res_2yr['true_pts'], marker='s', markersize=5,
                color=palette[1], lw=2.0, linestyle='--', label="2Y")

    ax_ins.set_xlim(0, 0.08)
    ax_ins.set_ylim(0, 0.08)
    ticks_ins = [0, 0.04, 0.08]
    ax_ins.set_xticks(ticks_ins)
    ax_ins.set_yticks(ticks_ins)
    ax_ins.tick_params(labelsize=10)
    ax_ins.set_title("2Y Zoomed", fontsize=14)

    ax.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='black')

    # Border style
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    for spine in ax_ins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    out_path = OUTPUT_DIR / "FigS4B_Calibration_Sensitivity_4Y_main_2Y_inset.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Calibration sensitivity plot saved to: {out_path}")

# Run the function if needed
if __name__ == "__main__":
    run_calibration_sensitivity()