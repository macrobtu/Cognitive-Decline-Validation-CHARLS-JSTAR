"""
Project: CoxPH Model Performance Evaluation (ROC & Calibration)
Author: macrob_tu
Date: updated in February 2026
Description: This script evaluates the performance of a 6-predictor Cox model 
across CHARLS and JSTAR cohorts, generating ROC and Calibration curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

# --- Path Configuration ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATH_TRAIN = DATA_DIR / "charls_model_ready.csv"
PATH_VAL_C = DATA_DIR / "charls_validate_ready.csv"
PATH_VAL_J = DATA_DIR / "sample_jstar_validate_ready.csv"

# --- Feature Definitions ---
num_features = ['orient', 'memory', 'grip_max']
cat_features = ['sex', 'edu_3cat', 'age_group2_safe']
all_features = num_features + cat_features
TIME_COL, EVENT_COL = "surv_time", "event"

# --- Preprocessing Pipeline ---
low_miss_num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')), 
    ("scaler", StandardScaler())
])
high_miss_num_pipeline = Pipeline([
    ("imputer", IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)), 
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')), 
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
])

preprocess = ColumnTransformer(transformers=[
    ("low_miss_num", low_miss_num_pipeline, [x for x in num_features if x != 'grip_max']),
    ("high_miss_num", high_miss_num_pipeline, ['grip_max']),
    ("cat", cat_pipeline, cat_features)
])

model_pipeline = make_pipeline(preprocess, CoxPHSurvivalAnalysis(ties="efron"))

# --- Data Loading Utility ---
def load_data(path):
    encodings = ['utf-8-sig', 'utf-8', 'latin1', 'gbk']
    for enc in encodings:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    raise ValueError(f"Could not read file: {path}")

# --- Training ---
df_train = load_data(PATH_TRAIN)
df_val_c = load_data(PATH_VAL_C)
df_val_j = load_data(PATH_VAL_J)

X_train = df_train[all_features]
y_train = Surv.from_arrays(event=df_train[EVENT_COL].astype(bool), time=df_train[TIME_COL])
model_pipeline.fit(X_train, y_train)

# --- Performance Metric Calculation ---
def get_metrics_and_calibration(df, time_horizon, name):
    # Predicted risk at specified horizon
    surv_funcs = model_pipeline.predict_survival_function(df[all_features])
    risk_probs = np.array([1 - fn(time_horizon) for fn in surv_funcs])
    
    # Landmark selection logic
    temp = df.copy()
    temp['risk_prob'] = risk_probs
    temp['is_case'] = (temp[EVENT_COL] == 1) & (temp[TIME_COL] <= time_horizon)
    temp['is_control'] = (temp[TIME_COL] > time_horizon)
    subset = temp[temp['is_case'] | temp['is_control']].copy()
    
    y_true = subset['is_case'].astype(int)
    y_prob = subset['risk_prob']
    
    # ROC and Brier Score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    brier = brier_score_loss(y_true, y_prob)
    
    # Calibration Slope & Intercept via linear regression on binned points
    prob_true_pts, prob_pred_pts = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    
    slope, intercept = np.nan, np.nan
    if len(prob_pred_pts) > 1:
        lr = LinearRegression()
        lr.fit(prob_pred_pts.reshape(-1, 1), prob_true_pts)
        slope = lr.coef_[0]
        intercept = lr.intercept_
    
    print(f"[{name}] AUC={roc_auc:.3f}, Brier={brier:.3f}, Slope={slope:.3f}, Int={intercept:.3f}")
    
    return {
        'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 
        'y_true': y_true, 'y_prob': y_prob, 
        'brier': brier, 'slope': slope, 'intercept': intercept,
        'prob_true_pts': prob_true_pts, 'prob_pred_pts': prob_pred_pts
    }

# --- Execute Metrics ---
res_t = get_metrics_and_calibration(df_train, 4.0, "Training")
res_c = get_metrics_and_calibration(df_val_c, 3.0, "CHARLS Val")
res_j = get_metrics_and_calibration(df_val_j, 2.0, "JSTAR Val")

# --- Visualizations ---

# Fig A: Combined ROC Curves
plt.figure(figsize=(8, 8))
colors = ["#333333", "#0072B2", "#D55E00"] 

plt.plot(res_t['fpr'], res_t['tpr'], color=colors[0], lw=1, alpha=0.6, 
         label=f"Training (CHARLS 4y): AUC={res_t['auc']:.3f}")
plt.plot(res_c['fpr'], res_c['tpr'], color=colors[1], lw=1.5, 
         label=f"Validation (CHARLS 3y): AUC={res_c['auc']:.3f}")
plt.plot(res_j['fpr'], res_j['tpr'], color=colors[2], lw=1.5, 
         label=f"Validation (JSTAR 2y): AUC={res_j['auc']:.3f}")

plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.tick_params(labelsize=15)
plt.xlabel('1 - Specificity', fontsize=20)
plt.ylabel('Sensitivity', fontsize=20)
plt.title('ROC Analysis Across Cohorts', fontsize=22)
plt.legend(loc='lower right', fontsize=15, frameon=True, facecolor='white', framealpha=1.0, edgecolor='black')

plt.savefig(OUTPUT_DIR / "Fig2A_Combined_ROC.png", dpi=300, bbox_inches='tight')
plt.show()

# Fig B: Combined Calibration Plot
plt.figure(figsize=(8, 8))
colors_sns = sns.color_palette("colorblind")

plt.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Perfectly calibrated")

def plot_calib_line(res, label, color):
    prob_true, prob_pred = calibration_curve(res['y_true'], res['y_prob'], n_bins=10, strategy='quantile')
    plt.plot(prob_pred, prob_true, marker='o', markersize=6, linestyle="-", color=color, label=label)

plot_calib_line(res_c, f"CHARLS Val (S={res_c['slope']:.2f}, I={res_c['intercept']:.2f})", colors_sns[0])
plot_calib_line(res_j, f"JSTAR Val (S={res_j['slope']:.2f}, I={res_j['intercept']:.2f})", colors_sns[1])

plt.xlabel("Mean Predicted Probability", fontsize=20)
plt.ylabel("Observed Fraction of Events", fontsize=20)
plt.title("Calibration Curves", fontsize=22)
plt.legend(loc="upper left", fontsize=15, frameon=True, edgecolor="black", facecolor="white")

ax = plt.gca()
plt.xlim(-0.01, 0.3)
plt.ylim(-0.01, 0.3)
ax.set_xticks([0.0, 0.1, 0.2, 0.3])
ax.set_yticks([0.0, 0.1, 0.2, 0.3])
plt.tick_params(labelsize=15)

plt.savefig(OUTPUT_DIR / "Fig2B_Calibration.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
"""
Project: GMM Clustering & Survival Analysis (Kaplan-Meier)
Author: macrobtu
Date: updated in February 2026
Description: This script performs Bayesian Gaussian Mixture Model (GMM) clustering 
based on risk scores and visualizes survival probabilities using KM curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from sklearn.mixture import BayesianGaussianMixture

# --- Font Settings ---
import matplotlib as mpl
try:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Verdana']
    mpl.rcParams['axes.unicode_minus'] = False 
except: pass

# %%
# ==================================
# 1. Path Configuration
# ==================================
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DATA_PATH = DATA_DIR / "charls_model_ready.csv"
COEF_PATH = OUTPUT_DIR / "table3_Final_CoxPH_Model_Detailed_Summary_lifelines_80Train_IterImp.xlsx"

# Feature definitions
num_features = ['orient', 'memory', 'grip_max']
cat_features = ['sex', 'edu_3cat', 'age_group2_safe']
all_features = num_features + cat_features
TIME_COL, EVENT_COL = "surv_time", "event"

# %%
# ==================================
# 2. Risk Score Calculation
# ==================================
print("Step 1: Calculating Risk Scores...")
low_miss_num_pipeline = Pipeline([("imputer", SimpleImputer(strategy='median')), ("scaler", StandardScaler())])
high_miss_num_pipeline = Pipeline([("imputer", IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)), ("scaler", StandardScaler())])
cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy='most_frequent')), ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))])

preprocess = ColumnTransformer(transformers=[
    ("low", low_miss_num_pipeline, ['orient', 'memory']), 
    ("high", high_miss_num_pipeline, ['grip_max']), 
    ("cat", cat_pipeline, cat_features)
])

df_source = pd.read_csv(DATA_PATH, encoding="utf-8-sig").dropna(subset=[TIME_COL, EVENT_COL]).copy()
X_data = df_source[all_features].copy()
preprocess.fit(X_data)
X_df = pd.DataFrame(preprocess.transform(X_data), columns=preprocess.get_feature_names_out(), index=X_data.index)

coef_df = pd.read_excel(COEF_PATH, index_col=0)
sig_coefs = coef_df[coef_df['p'] < 0.05]['coef']

score_cols = []
final_coefs = []
for name in sig_coefs.index:
    if name in X_df.columns:
        score_cols.append(name); final_coefs.append(sig_coefs[name])
    else:
        matches = [f for f in X_df.columns if f.endswith(name.split('__')[-1])]
        if matches: score_cols.append(matches[0]); final_coefs.append(sig_coefs[name])

df_source['Total_Risk_Score'] = X_df[score_cols].dot(np.array(final_coefs))
df_source['time'] = df_source[TIME_COL]
df_source['event'] = df_source[EVENT_COL].astype(int)

# %%
# ==================================
# 3. GMM Clustering
# ==================================
print("Step 2: Bayesian GMM Clustering...")
df_train, df_test = train_test_split(df_source, test_size=0.20, random_state=42, stratify=df_source['event'])

X_train_gmm = df_train['Total_Risk_Score'].values.reshape(-1, 1)
X_test_gmm = df_test['Total_Risk_Score'].values.reshape(-1, 1)

bgmm = BayesianGaussianMixture(n_components=3, weight_concentration_prior=1e-3, random_state=42, max_iter=1000).fit(X_train_gmm)

# Sort labels by mean risk
means = [(i, X_train_gmm[bgmm.predict(X_train_gmm) == i].mean()) for i in range(3)]
means.sort(key=lambda x: x[1])
label_map = {means[0][0]: 'Low Risk', means[1][0]: 'Medium Risk', means[2][0]: 'High Risk'}

df_train = df_train.copy(); df_test = df_test.copy()
df_train['Risk_Group'] = pd.Series(bgmm.predict(X_train_gmm), index=df_train.index).map(label_map)
df_test['Risk_Group'] = pd.Series(bgmm.predict(X_test_gmm), index=df_test.index).map(label_map)

# %%
# ==================================
# 4. Visualization Functions
# ==================================

def plot_km_fixed(df, title_suffix):
    fig = plt.figure(figsize=(12.5, 9)) 
    ax = plt.subplot(111)
    
    groups = ['Low Risk', 'Medium Risk', 'High Risk']
    colors = ['green', 'orange', 'red']
    kmfs = []
    
    for g, c in zip(groups, colors):
        mask = df['Risk_Group'] == g
        n = mask.sum()
        if n > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[mask, 'time'], df.loc[mask, 'event'], label=f"{g} (N={n})")
            kmf.plot(ax=ax, color=c, ci_show=True, lw=2.5)
            kmfs.append(kmf)
    
    SMALL_FONT = 22 
    ax.set_title(f"Kaplan-Meier Analysis", fontsize=28, pad=10)
    ax.set_xlim(-0.1, 4.5)
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks([0, 2, 4])
    ax.set_xlabel("Follow-up Time (Years)", fontsize=26, labelpad=4)
    ax.xaxis.set_label_position('bottom') 
    ax.set_ylabel("Survival Probability", fontsize=26, labelpad=4)
    ax.tick_params(axis='both', which='major', labelsize=SMALL_FONT)
    
    ax.grid(False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.legend(loc='lower left', fontsize=22, frameon=True, edgecolor='black')

    if len(kmfs) > 0:
        risk_ax=add_at_risk_counts(*kmfs, ax=ax, rows_to_show=['At risk', 'Censored', 'Events'], 
                                   fontsize=SMALL_FONT)
        risk_ax.tick_params(axis='y', pad=1)

    plt.subplots_adjust(bottom=0.35, left=0.30, right=0.90, top=0.92)
    save_path = OUTPUT_DIR / f"Fig3KM_{title_suffix.replace(' ', '_')}_GMM.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_density_gmm(df, title_suffix):
    plt.figure(figsize=(10, 10))
    groups = ['Low Risk', 'Medium Risk', 'High Risk']
    colors = ['green', 'orange', 'red']
    
    for g, c in zip(groups, colors):
        subset = df[df['Risk_Group'] == g]
        if len(subset) > 0:
            sns.kdeplot(data=subset, x='Total_Risk_Score', color=c, fill=True, alpha=0.4, linewidth=2, label=f"{g} (N={len(subset)})")
            
    plt.title(f"Bayesian GMM-based Clustering", fontsize=28)
    plt.xlabel("Total Risk Score (LogHR)", fontsize=26)
    plt.ylabel("Density", fontsize=22)
    plt.legend(loc='upper right', fontsize=22, frameon=True, edgecolor='black')
    plt.xlim(-2,2); plt.ylim(0, 2)
    plt.xticks([-2, -1, 0, 1, 2]); plt.yticks([0, 0.5, 1, 1.5, 2])
    plt.grid(False)
    
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=22)
    for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(1.0)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"Fig3Density_{title_suffix.replace(' ', '_')}_GMM.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

# %%
# ==================================
# 5. Execute Visualization
# ==================================
print("Step 3: Generating all 4 plots...")
# Train Set plots
plot_km_fixed(df_train, "80 Train Set")
plot_density_gmm(df_train, "80 Train Set")

# Test Set plots
plot_km_fixed(df_test, "20 Test Set")
plot_density_gmm(df_test, "20 Test Set")

# %% [markdown]
# ### Final Risk Stratification Bar Plot (Original Style Preserved)
# This script maintains the exact visual style of your original plots 
# while fixing the yerr error and translating comments to English.

# %%

