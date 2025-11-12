#!/usr/bin/env python
# coding: utf-8

# ==================================
# ✅ Master Analysis Script: Model Comparison and Final Model Training (CHARLS)
# This script performs the full analysis workflow on the CHARLS dataset.
# It is divided into sequential blocks.
#
# This script corresponds to the combination of multiple notebook cells
# (In[12], In[13], In[14], In[18], In[20]) from the original file.
#
# ❗️ WARNING: Step 7 (Cross-Validation) is computationally intensive
# and may take several hours to run due to the use of IterativeImputer
# within each fold for multiple models.
# ==================================

print("--- Initializing script and importing all necessary libraries... ---")
# %%
# ==================================
# 0. Library Imports
# ==================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from pathlib import Path
import warnings
import os
import matplotlib as mpl

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.base import clone, BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer # Enable experimental
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

# Scikit-survival
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import (
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score
)
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator

# Other Machine Learning
import xgboost as xgb
import lightgbm as lgb

# Statistics and Plotting
from scipy.stats import t
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
from dcurves import dca
from statsmodels.nonparametric.smoothers_lowess import lowess

# Ignore common warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="The objective` survival:cox` is deprecated")

print("✅ All libraries imported successfully.")

# ==================================
# 1. Global Configuration (Paths and Variables)
# ==================================
print("\n--- Step 1: Configuring paths and variables ---")

# --- Define Project Structure ---
# Assumes this script is in the root folder: C:\Users\tolph\Desktop\Cognitive-Decline-Validation-CHARLS-JSTAR
BASE_DIR = Path(r"C:\Users\tolph\Desktop\Cognitive-Decline-Validation-CHARLS-JSTAR")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Project base directory set to: {BASE_DIR}")
print(f"Input data will be read from: {DATA_DIR}")
print(f"Output files will be saved to: {OUTPUT_DIR}")

# --- Define Feature and Target Column Names ---
TIME_COL, EVENT_COL = "surv_time", "event"

# Define raw feature lists based on 'charls_model_ready.csv'
num_features = ['orient', 'memory', 'grip_max']
cat_features = ['sex', 'edu_3cat', 'age_group2_safe']
all_features = num_features + cat_features

# Define preprocessing groups
high_missing_continuous = ['grip_max']
low_missing_continuous = [col for col in num_features if col not in high_missing_continuous]

# --- Set global plot styling ---
try:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Verdana']
    mpl.rcParams['axes.unicode_minus'] = False
    print("✅ Matplotlib font set to prefer Arial (or other sans-serif).")
except Exception as e_font:
    print(f"⚠️ Could not set font (Arial may be missing), will use default: {e_font}")

# ==================================
# 2. Define Global Preprocessing Pipeline
# ==================================
print("\n--- Step 2: Defining the universal preprocessing pipeline ---")
# This pipeline will be used by all models for consistency.
# It uses IterativeImputer for high-missingness variables.

# Pipeline 1: Low-missingness continuous variables (Median + Scale)
low_miss_num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])

# Pipeline 2: High-missingness continuous variables (Iterative + Scale)
high_miss_num_pipeline = Pipeline([
    ("imputer", IterativeImputer(estimator=BayesianRidge(), max_iter=10, 
                               random_state=42, imputation_order='ascending')),
    ("scaler", StandardScaler())
])

# Pipeline 3: Categorical variables (Mode + OneHotEncode)
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
])

# Create the master ColumnTransformer
preprocess = ColumnTransformer(
    transformers=[
        ("low_miss_num", low_miss_num_pipeline, low_missing_continuous),
        ("high_miss_num", high_miss_num_pipeline, high_missing_continuous),
        ("cat", cat_pipeline, cat_features)
    ],
    remainder='drop' # Explicitly drop any columns not specified
)
print("✅ Full preprocessing pipeline (ColumnTransformer) defined successfully.")

# ==================================
# 3. Load Main Data (CHARLS)
# ==================================
print("\n--- Step 3: Loading main modeling data ---")
data_path = DATA_DIR / "charls_model_ready.csv"

encodings = ["utf-8-sig", "utf-8", "cp932", "gbk", "latin1"]
df = None
for enc in encodings:
    try:
        df = pd.read_csv(data_path, encoding=enc)
        print(f"✅ File loaded successfully (Encoding: {enc})")
        break
    except UnicodeDecodeError:
        continue
if df is None:
    print(f"❌ ERROR: Could not read file {data_path} with any encoding.")
    exit()

# Drop rows where outcome is missing
df = df.dropna(subset=[TIME_COL, EVENT_COL]).copy()
print(f"Data loaded and cleaned. Total samples for analysis: {df.shape[0]}")

# ==================================
# 4. Prepare Full X and y
# ==================================
print("\n--- Step 4: Preparing feature matrix (X) and target (y) ---")
# Feature matrix X only contains the raw specified features
X = df[all_features].copy()

# Target variable y (sksurv format)
y = Surv.from_arrays(event=df[EVENT_COL].astype(bool), time=df[TIME_COL].astype(float))
print(f"Feature matrix X shape: {X.shape}, Target y length: {len(y)}")

# ==================================
# 5. 80/20 Train/Test Split
# ==================================
print("\n--- Step 5: Splitting data into 80% train and 20% test sets ---")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
try:
    train_idx, test_idx = next(sss.split(X, y["event"]))
    X_train_full, X_test_final = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test_final = y[train_idx], y[test_idx]
    print(f"Training samples: {X_train_full.shape[0]}, Final test samples: {X_test_final.shape[0]}")
except ValueError as e:
     print(f"❌ ERROR: Could not split data, likely too few events for stratification: {e}")
     exit()

# ==================================
# 6. Define Model Wrappers
# ==================================
print("\n--- Step 6: Defining model wrappers ---")

class SurvFunc:
    def __init__(self, times, probs): self.x = times; self.y = probs
    def __call__(self, t): return np.interp(t, self.x, self.y)

class BaseSurvivalWrapper(BaseEstimator):
    def predict_survival_function(self, X):
        risk_scores = self.predict(X)
        risk_scores_clipped = np.clip(risk_scores, np.percentile(risk_scores, 5), np.percentile(risk_scores, 95))
        hazard_ratios = np.exp(risk_scores_clipped - np.mean(risk_scores_clipped))
        
        if hasattr(self.model_, 'baseline_survival_') and self.model_.baseline_survival_ is not None:
             base_surv = self.model_.baseline_survival_
             times = base_surv.x
             surv_probs = base_surv.y ** np.exp(risk_scores[:, np.newaxis])
             return [SurvFunc(times, p) for p in surv_probs.T]
        else:
            try:
                times_train = np.unique(y_train_full['time'][y_train_full['event']])
                times_train = np.sort(times_train)
                if len(times_train) == 0: # Handle case with no events in training (unlikely)
                    times_train = np.linspace(0, y_train_full['time'].max(), 100)
            except Exception: # Fallback
                times_train = np.linspace(0, 10, 100) # Generic time
                
            lambda_ = 0.1 
            base_surv_probs = np.exp(-times_train * lambda_)
            surv_probs = base_surv_probs ** hazard_ratios[:, np.newaxis]
            return [SurvFunc(times_train, p) for p in surv_probs]

class CoxPHWrapper(BaseSurvivalWrapper):
    def __init__(self, alpha=0, ties="efron"):
        self.alpha, self.ties = alpha, ties
    def fit(self, X, y):
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha, ties=self.ties)
        try:
             self.model_.fit(X, y)
        except Exception as e:
             print(f"   ERROR fitting CoxPH: {e}"); self.model_ = None
        return self
    def predict(self, X):
        if self.model_ is None: return np.zeros(X.shape[0])
        return self.model_.predict(X)
    def predict_survival_function(self, X):
        if self.model_ is None or not hasattr(self.model_, 'predict_survival_function'):
            return super().predict_survival_function(X)
        return self.model_.predict_survival_function(X)

class CoxnetWrapper(BaseSurvivalWrapper):
    def __init__(self, l1_ratio=0.9, alpha_min_ratio=0.01, n_alphas=100):
        self.l1_ratio, self.alpha_min_ratio, self.n_alphas = l1_ratio, alpha_min_ratio, n_alphas
    def fit(self, X, y):
        self.model_ = CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio,
                                             alpha_min_ratio=self.alpha_min_ratio,
                                             n_alphas=self.n_alphas,
                                             fit_baseline_model=True)
        try:
             self.model_.fit(X, y)
        except Exception as e:
             print(f"   ERROR fitting Coxnet: {e}"); self.model_ = None
        return self
    def predict(self, X):
        if self.model_ is None: return np.zeros(X.shape[0])
        return self.model_.predict(X)
    def predict_survival_function(self, X):
        if self.model_ is None or not hasattr(self.model_, 'predict_survival_function'):
             return super().predict_survival_function(X)
        return self.model_.predict_survival_function(X)

class GBSAWrapper(BaseSurvivalWrapper):
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42):
        self.n_estimators, self.learning_rate, self.max_depth, self.random_state = n_estimators, learning_rate, max_depth, random_state
    def fit(self, X, y):
        self.model_ = GradientBoostingSurvivalAnalysis(n_estimators=self.n_estimators,
                                                      learning_rate=self.learning_rate,
                                                      max_depth=self.max_depth,
                                                      random_state=self.random_state)
        try:
            self.model_.fit(X, y)
        except Exception as e:
            print(f"   ERROR fitting GBSA: {e}"); self.model_ = None
        return self
    def predict(self, X):
        if self.model_ is None: return np.zeros(X.shape[0])
        return self.model_.predict(X)

class RSFWrapper(BaseSurvivalWrapper):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=6, min_samples_leaf=3, random_state=42):
        self.n_estimators, self.max_depth, self.min_samples_split, self.min_samples_leaf, self.random_state = n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state
    def fit(self, X, y):
        self.model_ = RandomSurvivalForest(n_estimators=self.n_estimators,
                                           max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           random_state=self.random_state,
                                           n_jobs=-1)
        try:
            self.model_.fit(X, y)
        except Exception as e:
            print(f"   ERROR fitting RSF: {e}"); self.model_ = None
        return self
    def predict(self, X):
        if self.model_ is None: return np.zeros(X.shape[0])
        return self.model_.predict(X)
    def predict_survival_function(self, X):
         if self.model_ is None or not hasattr(self.model_, 'predict_survival_function'):
             return super().predict_survival_function(X)
         step_funcs = self.model_.predict_survival_function(X, return_array=False)
         return [SurvFunc(sf.x, sf.y) for sf in step_funcs]

class FastSVMWrapper(BaseSurvivalWrapper):
    def __init__(self, alpha=1, rank_ratio=0.8, max_iter=20, random_state=42):
        self.alpha, self.rank_ratio, self.max_iter, self.random_state = alpha, rank_ratio, max_iter, random_state
    def fit(self, X, y):
        self.model_ = FastSurvivalSVM(alpha=self.alpha,
                                      rank_ratio=self.rank_ratio,
                                      max_iter=self.max_iter,
                                      random_state=self.random_state)
        try:
             self.model_.fit(X, y)
        except Exception as e:
             print(f"   ERROR fitting FastSVM: {e}"); self.model_ = None
        return self
    def predict(self, X):
        if self.model_ is None: return np.zeros(X.shape[0])
        return self.model_.predict(X)

class ESTWrapper(BaseSurvivalWrapper):
    def __init__(self, n_estimators=50, random_state=42, n_jobs=-1, **kwargs):
        self.n_estimators, self.random_state, self.n_jobs, self.kwargs = n_estimators, random_state, n_jobs, kwargs
    def fit(self, X, y):
        self.model_ = ExtraTreesRegressor(n_estimators=self.n_estimators,
                                          random_state=self.random_state,
                                          n_jobs=self.n_jobs,
                                          **self.kwargs)
        try:
             self.model_.fit(X, y["time"])
        except Exception as e:
             print(f"   ERROR fitting EST: {e}"); self.model_ = None
        return self
    def predict(self, X):
        if self.model_ is None: return np.zeros(X.shape[0])
        return -self.model_.predict(X)

class XGBoostCoxWrapper(BaseSurvivalWrapper):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, n_jobs=-1, **kwargs):
        self.n_estimators, self.learning_rate, self.max_depth, self.random_state, self.n_jobs, self.kwargs = n_estimators, learning_rate, max_depth, random_state, n_jobs, kwargs
    def fit(self, X, y):
        self.model_ = xgb.XGBRegressor(objective='survival:cox',
                                     n_estimators=self.n_estimators,
                                     learning_rate=self.learning_rate,
                                     max_depth=self.max_depth,
                                     random_state=self.random_state,
                                     n_jobs=self.n_jobs,
                                     **self.kwargs)
        y_xgb = np.where(y["event"], y["time"], -y["time"])
        try:
             self.model_.fit(X, y_xgb)
        except Exception as e:
             print(f"   ERROR fitting XGBoostCox: {e}"); self.model_ = None
        return self
    def predict(self, X):
        if self.model_ is None: return np.zeros(X.shape[0])
        return self.model_.predict(X)

class LightGBMRegressionWrapper(BaseSurvivalWrapper):
     def __init__(self, n_estimators=50, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, **kwargs):
        self.n_estimators, self.learning_rate, self.num_leaves, self.random_state, self.n_jobs, self.kwargs = n_estimators, learning_rate, num_leaves, random_state, n_jobs, kwargs
     def fit(self, X, y):
        self.model_ = lgb.LGBMRegressor(objective='regression_l1',
                                      n_estimators=self.n_estimators,
                                      learning_rate=self.learning_rate,
                                      num_leaves=self.num_leaves,
                                      random_state=self.random_state,
                                      n_jobs=self.n_jobs,
                                      verbose=-1, # Suppress LightGBM verbosity
                                      **self.kwargs)
        try:
            self.model_.fit(X, y["time"])
        except Exception as e:
            print(f"   ERROR fitting LightGBM Regression: {e}"); self.model_ = None
        return self
     def predict(self, X):
        if self.model_ is None: return np.zeros(X.shape[0])
        return -self.model_.predict(X)

print("✅ Model wrappers defined.")

# ==================================
# 7. Model Comparison: Cross-Validation
# ==================================
print("\n--- Step 7: Starting 8-model comparison (Cross-Validation)... ---")
# Using the (faster) SimpleImputer pipeline for this comparison
# If you want the more complex one, change 'preprocess' here.
# We will use the SIMPLE (median/mode) imputer for model comparison,
# as IterativeImputer is too slow for 10-fold CV * 8 models.
print("NOTE: Using SimpleImputer (median/mode) for this comparison for speed.")
print("The 'IterativeImputer' pipeline will be used for the final model.")

# --- Define SIMPLE Preprocessing Pipeline (for CV) ---
num_pipeline_simple = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])
cat_pipeline_simple = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
])
preprocess_simple = ColumnTransformer(
    transformers=[
        ("num", num_pipeline_simple, num_features), # Apply to all num_features
        ("cat", cat_pipeline_simple, cat_features)
    ],
    remainder='drop'
)
print("✅ SIMPLE preprocessing pipeline (for CV) defined.")

# --- Define Model Dictionary ---
models = {
    "CoxPH": CoxPHWrapper(),
   # "ElasticNet-CoxPH": CoxnetWrapper(l1_ratio=0.9),
    "GBSA": GBSAWrapper(n_estimators=50, random_state=42),
   # "RSF": RSFWrapper(n_estimators=100, random_state=42),
    "EST": ESTWrapper(n_estimators=50, random_state=42, n_jobs=-1),
   # "XGBoost-Cox": XGBoostCoxWrapper(n_estimators=100, random_state=42, n_jobs=-1),
    "LightGBM-Reg": LightGBMRegressionWrapper(n_estimators=50, random_state=42, n_jobs=-1, verbosity=-1),
}
print(f"✅ Defined {len(models)} models for comparison.")

results = {name: {"c_index": [], "cd_auc": [], "ibs": [], "rmse": []} for name in models}
N_SPLITS = 5
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
successful_folds = {name: 0 for name in models}

for model_name, base_model in models.items():
    print(f"\n=== Training Model: {model_name} ===")
    model_results = {"c_index": [], "cd_auc": [], "ibs": [], "rmse": []}

    for fold, (train_idx_inner, test_idx_inner) in enumerate(kf.split(X_train_full, y_train_full['event']), start=1):
        print(f"  Fold {fold}/{N_SPLITS}")
        X_train, X_test = X_train_full.iloc[train_idx_inner], X_train_full.iloc[test_idx_inner]
        y_train, y_test = y_train_full[train_idx_inner], y_train_full[test_idx_inner]

        # --- Build Pipeline with SIMPLE preprocess ---
        pipeline = Pipeline([
            ('preprocess', clone(preprocess_simple)), # Use the SIMPLE preprocessor
            ('model', clone(base_model))
        ])

        ci, mean_auc, ibs, rmse = np.nan, np.nan, np.nan, np.nan

        try:
            if y_test["event"].sum() < 2:
                print(f"   ⚠️ Too few events in test set (<2), skipping fold.")
                continue

            pipeline.fit(X_train, y_train)

            if hasattr(pipeline.named_steps['model'], 'model_') and pipeline.named_steps['model'].model_ is None:
                 print(f"   ❌ Model fitting failed (internal wrapper error), skipping fold.")
                 continue

            risk_scores = pipeline.predict(X_test)

            event_mask_test = y_test["event"]
            times_test = y_test["time"]

            if event_mask_test.sum() >= 2:
                lower_bound = np.percentile(times_test[event_mask_test], 10)
                upper_bound = np.percentile(times_test[event_mask_test], 90)

                event_mask_train = y_train["event"]
                times_train = y_train["time"]
                if event_mask_train.sum() > 0:
                     tau_train = times_train[event_mask_train].max()
                     upper_bound = min(upper_bound, tau_train)

                if lower_bound < upper_bound:
                    times_eval = np.linspace(lower_bound, upper_bound, num=50)

                    try:
                        ci = concordance_index_ipcw(y_train, y_test, risk_scores, tau=upper_bound)[0]
                    except Exception as e:
                        print(f"   ⚠️ C-index calculation error: {e}")

                    try:
                        aucs, _ = cumulative_dynamic_auc(y_train, y_test, risk_scores, times_eval)
                        mean_auc = np.nanmean(aucs)
                    except Exception as e:
                        print(f"   ⚠️ CD-AUC calculation error: {e}")

                    try:
                        if hasattr(pipeline, 'predict_survival_function'):
                            surv_funcs = pipeline.predict_survival_function(X_test)
                            if hasattr(surv_funcs, '__len__') and len(surv_funcs) > 0 and len(surv_funcs) == X_test.shape[0]:
                                surv_matrix = np.row_stack([fn(times_eval) for fn in surv_funcs])
                                if surv_matrix.shape == (X_test.shape[0], len(times_eval)):
                                    try:
                                        ibs = integrated_brier_score(y_train, y_test, surv_matrix, times_eval)
                                    except Exception as e_ibs:
                                        print(f"   ⚠️ IBS Calculation Error: {e_ibs}")
                                    try:
                                        km_t, km_p = kaplan_meier_estimator(y_test["event"], y_test["time"])
                                        valid_times_mask = (times_eval >= km_t.min()) & (times_eval <= km_t.max())
                                        valid_times_eval = times_eval[valid_times_mask]
                                        if len(valid_times_eval) > 1:
                                            km_interp = np.interp(valid_times_eval, km_t, km_p)
                                            valid_surv_matrix = surv_matrix[:, valid_times_mask]
                                            mean_pred = np.nanmean(valid_surv_matrix, axis=0)
                                            valid_comparison_mask = ~np.isnan(mean_pred) & ~np.isnan(km_interp)
                                            if np.sum(valid_comparison_mask) > 0:
                                                rmse = np.sqrt(np.mean((mean_pred[valid_comparison_mask] - km_interp[valid_comparison_mask]) ** 2))
                                            else: rmse = np.nan
                                        else: rmse = np.nan
                                    except Exception as e_rmse:
                                        print(f"   ⚠️ RMSE Calculation Error: {e_rmse}")
                                else: print(f"   ⚠️ IBS/RMSE skipped: Survival matrix shape mismatch.")
                            else: print(f"   ⚠️ IBS/RMSE skipped: predict_survival_function returned invalid data.")
                        else: print(f"   ⚠️ IBS/RMSE skipped: Model lacks predict_survival_function method.")
                    except Exception as e_outer:
                        print(f"   ⚠️ Unexpected Error during IBS/RMSE: {e_outer}")

                    if not np.isnan(ci):
                        print(f"   ✅ C-index={ci:.4f}, CD-AUC={mean_auc:.4f}, IBS={ibs:.4f}, RMSE={rmse:.4f}")
                        model_results["c_index"].append(ci)
                        model_results["cd_auc"].append(mean_auc)
                        model_results["ibs"].append(ibs)
                        model_results["rmse"].append(rmse)
                        successful_folds[model_name] += 1
                    else:
                         print(f"   ❌ Fold {fold}: C-index calculation failed, fold results not recorded.")
                else:
                    print(f"   ⚠️ Invalid time range for evaluation, skipping fold.")
            else:
                 print(f"   ⚠️ Not enough events in test set to define evaluation range.")
        except Exception as e:
            print(f"   ❌ Fold {fold} failed with critical error: {e}")
            traceback.print_exc()

    for metric, values in model_results.items():
        results[model_name][metric].extend(values)

print("✅ Model comparison complete.")

# ==================================
# 8. Summarize CV Results
# ==================================
print("\n--- Step 8: Summarizing cross-validation results... ---")

# --- Check for any valid results ---
valid_results_exist = any(
    any(isinstance(val, (int, float)) and not np.isnan(val) for val in metric_list)
    for model_results_dict in results.values()
    for metric_list in model_results_dict.values()
)

if not valid_results_exist:
    print("❌ No valid model evaluation results were recorded. Cannot generate summary.")
else:
    display_data = {}
    successful_folds_counts = {}
    metric_map = {
        "c_index": {"display": "C-index (95% CI)", "higher_better": True},
        "cd_auc": {"display": "CD-AUC (95% CI)", "higher_better": True},
        "ibs": {"display": "IBS (95% CI)", "higher_better": False},
        "rmse": {"display": "RMSE (95% CI)", "higher_better": False}
    }
    all_models = list(results.keys())
    mean_metrics_for_ranking = {model: {} for model in all_models}

    for model in all_models:
        display_data[model] = {}
        c_index_values = [v for v in results.get(model, {}).get("c_index", []) if not np.isnan(v)]
        n_success = len(c_index_values)
        successful_folds_counts[model] = n_success

        if n_success < 2:
            t_critical = np.nan
            if n_success == 0:
                 print(f"⚠️ Model '{model}': No successful folds based on C-index.")
            else:
                 print(f"⚠️ Model '{model}': Not enough successful folds ({n_success}) to calculate 95% CI.")
        else:
            t_critical = t.ppf(0.975, n_success - 1)

        for metric_key, props in metric_map.items():
            values = [v for v in results.get(model, {}).get(metric_key, []) if not np.isnan(v)]
            current_n_success = len(values)
            mean = np.mean(values) if current_n_success > 0 else np.nan
            sd = np.std(values) if current_n_success > 1 else 0
            mean_metrics_for_ranking[model][f"{metric_key}_mean"] = mean

            if n_success < 2 or np.isnan(mean):
                ci_str = f"{mean:.4f} (N/A)" if not np.isnan(mean) else "N/A"
            else:
                se = sd / np.sqrt(n_success) if n_success > 0 else 0
                ci_lower, ci_upper = mean - t_critical * se, mean + t_critical * se
                ci_str = f"{mean:.4f} ({ci_lower:.4f}-{ci_upper:.4f})"
            display_data[model][props["display"]] = ci_str

    display_df = pd.DataFrame(display_data).T

    # --- Mean Rank Calculation ---
    try:
        print("\n--- Calculating ranks (Rank 1 = Best) ---")
        df_means_for_ranking = pd.DataFrame(mean_metrics_for_ranking).T
        df_ranks = pd.DataFrame(index=df_means_for_ranking.index)

        for metric_key, props in metric_map.items():
            mean_col = f"{metric_key}_mean"
            if mean_col in df_means_for_ranking.columns and not df_means_for_ranking[mean_col].isnull().all():
                 df_ranks[f'{mean_col}_rank'] = df_means_for_ranking[mean_col].rank(
                     method='min', ascending=(not props["higher_better"]), na_option='bottom')
                 print(f"  - Ranked '{mean_col}' ({'Higher is better' if props['higher_better'] else 'Lower is better'}).")

        rank_cols = [col for col in df_ranks.columns if col.endswith('_rank')]
        if rank_cols:
             display_df['Mean Rank'] = df_ranks[rank_cols].mean(axis=1)
             display_df = display_df.sort_values(by='Mean Rank', ascending=True)
             if 'Mean Rank' in display_df.columns and pd.api.types.is_numeric_dtype(display_df['Mean Rank']):
                 display_df['Mean Rank'] = display_df['Mean Rank'].round(2)
             print("✅ Mean rank calculation successful.")
        else:
             print("⚠️ No metrics could be ranked.")
             display_df['Mean Rank'] = "N/A"
    except Exception as e:
        print(f"❌ Error during mean rank calculation: {e}"); traceback.print_exc()
        display_df['Mean Rank'] = "Error"

    print("\n📊 Model Performance Comparison (Sorted by Mean Rank):")
    cols_to_display = ["C-index (95% CI)", "CD-AUC (95% CI)", "IBS (95% CI)","RMSE (95% CI)", "Mean Rank"]
    cols_to_display = [col for col in cols_to_display if col in display_df.columns]
    print(display_df[cols_to_display])

    # Save summary table
    summary_path = OUTPUT_DIR / "model_comparison_summary.xlsx"
    display_df.to_excel(summary_path)
    print(f"\n📂 Summary table saved to: {summary_path}")

    # --- Heatmap Generation ---
    try:
        plt.figure(figsize=(10, 6))
        metrics_for_heatmap = ['c_index_mean', 'cd_auc_mean', 'ibs_mean', 'rmse_mean']
        heatmap_data = df_means_for_ranking[[col for col in metrics_for_heatmap if col in df_means_for_ranking.columns]].dropna(how='all')
        heatmap_display_names = {
            'c_index_mean': 'C-index Mean', 'cd_auc_mean': 'CD-AUC Mean',
            'ibs_mean': 'IBS Mean', 'rmse_mean': 'RMSE Mean'
        }
        heatmap_data.rename(columns=heatmap_display_names, inplace=True)
        if not heatmap_data.empty:
             sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap="viridis", linewidths=.5)
             plt.title("Survival Model Comparison (Mean CV Performance)")
             plt.yticks(rotation=0); plt.tight_layout()
             heatmap_path = OUTPUT_DIR / "Model_Comparison_Heatmap.png"
             plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
             print(f"\n✅ Heatmap saved to: {heatmap_path}"); plt.show()
        else:
             print("\n⚠️ Heatmap not generated, no valid data.")
    except Exception as e:
        print(f"\n❌ Error generating heatmap: {e}"); traceback.print_exc()

print("\n🏁 Model comparison script finished.")

# %%
# ==================================
# 9. Final Cox Model: Training, Evaluation, and Parameter Extraction
# ==================================
print("\n" + "="*60)
print("--- Step 9: Final Cox Model Training, Evaluation, and Parameter Extraction ---")
print("="*60)

# --- 9.1 Define Final Model and Preprocessing (using COMPLEX IterativeImputer) ---
print("\n--- 9.1: Defining FINAL pipeline with IterativeImputer ---")
# This uses the 'preprocess' object defined at the top (lines 90-101)
# which contains IterativeImputer.
final_cox_model_wrapper = CoxPHWrapper()
final_pipeline = Pipeline([
    ('preprocess', clone(preprocess)), # Use the COMPLEX preprocessor
    ('model', clone(final_cox_model_wrapper))
])

# --- 9.2 Fit Final Model on 80% Training Data ---
print(f"\n--- 9.2: Fitting final model on 80% Training Set ({X_train_full.shape[0]} samples)... ---")
try:
    final_pipeline.fit(X_train_full, y_train_full)
    print("✅ Final model fitted successfully.")
    final_model_fitted = True
except Exception as e:
    print(f"❌ Failed to fit final model: {e}"); traceback.print_exc()
    final_model_fitted = False

# --- 9.3 Evaluate on 20% Test Set (with Bootstrap CI) ---
if final_model_fitted:
    print("\n--- 9.3: Evaluating on 20% Test Set (with Bootstrap CI)... ---")
    
    X_eval = X_test_final 
    y_eval = y_test_final
    y_train_ref = y_train_full 

    if y_eval["event"].sum() < 2:
        print("❌ Too few events in test set (<2). Skipping evaluation.")
    else:
        # --- 9.3.1: Calculate point estimates ---
        print("Calculating point estimates on test set...")
        c_index_orig, mean_auc_orig, ibs_orig, rmse_orig = np.nan, np.nan, np.nan, np.nan
        eval_times = []
        eval_upper_bound = np.nan
        
        # Define tau_train_full here
        event_mask_train = y_train_ref["event"]
        times_train = y_train_ref["time"]
        if event_mask_train.sum() > 0:
            tau_train_full = times_train[event_mask_train].max()
        else:
            tau_train_full = times_train.max()
            
        try:
            risk_scores_orig = final_pipeline.predict(X_eval)
            event_mask_eval = y_eval["event"]
            times_eval = y_eval["time"]
            
            lower_bound = np.percentile(times_eval[event_mask_eval], 10)
            upper_bound = np.percentile(times_eval[event_mask_eval], 90)
            
            if event_mask_train.sum() > 0: 
                eval_upper_bound = min(upper_bound, tau_train_full)
            else: 
                eval_upper_bound = upper_bound
                
            if lower_bound >= eval_upper_bound: 
                raise ValueError("Invalid evaluation time range")
                
            eval_times = np.linspace(lower_bound, eval_upper_bound, 100)
            
            c_index_orig = concordance_index_ipcw(y_train_ref, y_eval, risk_scores_orig, tau=eval_upper_bound)[0]
            aucs_orig, _ = cumulative_dynamic_auc(y_train_ref, y_eval, risk_scores_orig, eval_times)
            mean_auc_orig = np.nanmean(aucs_orig)
            
            if hasattr(final_pipeline, 'predict_survival_function'):
                surv_funcs_orig = final_pipeline.predict_survival_function(X_eval)
                if hasattr(surv_funcs_orig, '__len__') and len(surv_funcs_orig) == X_eval.shape[0]:
                    surv_matrix_orig = np.vstack([fn(eval_times) for fn in surv_funcs_orig])
                    if surv_matrix_orig.shape == (X_eval.shape[0], len(eval_times)):
                        ibs_orig = integrated_brier_score(y_train_ref, y_eval, surv_matrix_orig, eval_times)
                        km_t, km_p = kaplan_meier_estimator(y_eval["event"], y_eval["time"])
                        mask = (eval_times >= km_t.min()) & (eval_times <= km_t.max())
                        if mask.sum() > 1:
                            km_interp = np.interp(eval_times[mask], km_t, km_p)
                            mean_pred = np.nanmean(surv_matrix_orig[:, mask], axis=0)
                            mask_comp = ~np.isnan(mean_pred) & ~np.isnan(km_interp)
                            if mask_comp.sum() > 0:
                                rmse_orig = np.sqrt(np.mean((mean_pred[mask_comp] - km_interp[mask_comp]) ** 2))
        except Exception as e:
            print(f"❌ Error calculating point estimates: {e}")
            traceback.print_exc()

        # --- 9.3.2: Bootstrap CI ---
        n_bootstraps = 1000
        boot_individual_results = [] 
        print(f"\nStarting {n_bootstraps} bootstrap samples on test set (this may take time)...")
        test_indices = np.arange(len(X_eval))
        successful_bootstraps = 0
        
        for i in range(n_bootstraps):
            c_boot, mean_auc_boot, ibs_boot, rmse_boot = np.nan, np.nan, np.nan, np.nan
            boot_valid = False
            
            try:
                boot_idx = resample(test_indices)
                X_boot, y_boot = X_eval.iloc[boot_idx].copy(), y_eval[boot_idx]
                
                if y_boot["event"].sum() < 2: 
                    continue
                    
                risk_scores_boot = final_pipeline.predict(X_boot)
                
                # Calculate C-index first
                try:
                    c_boot = concordance_index_ipcw(y_train_ref, y_boot, risk_scores_boot, tau=eval_upper_bound)[0]
                    if not np.isnan(c_boot): 
                        boot_valid = True
                except Exception: 
                    pass
                
                # Calculate other metrics only if C-index was successful
                if boot_valid:
                    try:
                        event_mask_boot = y_boot["event"]
                        times_boot = y_boot["time"]
                        
                        # Use consistent evaluation times for all bootstrap samples
                        lower_b_boot = np.percentile(times_boot[event_mask_boot], 10)
                        upper_b_boot = np.percentile(times_boot[event_mask_boot], 90)
                        eval_upper_boot = min(upper_b_boot, tau_train_full)
                        
                        if lower_b_boot < eval_upper_boot:
                            eval_times_boot = np.linspace(lower_b_boot, eval_upper_boot, 50)
                            
                            # CD-AUC
                            try:
                                aucs_boot, _ = cumulative_dynamic_auc(y_train_ref, y_boot, risk_scores_boot, eval_times_boot)
                                mean_auc_boot = np.nanmean(aucs_boot)
                            except Exception:
                                pass
                            
                            # IBS and RMSE
                            try:
                                if hasattr(final_pipeline, 'predict_survival_function'):
                                    surv_funcs_boot = final_pipeline.predict_survival_function(X_boot)
                                    if hasattr(surv_funcs_boot, '__len__') and len(surv_funcs_boot) == X_boot.shape[0]:
                                        surv_matrix_boot = np.vstack([fn(eval_times_boot) for fn in surv_funcs_boot])
                                        if surv_matrix_boot.shape == (X_boot.shape[0], len(eval_times_boot)):
                                            # IBS
                                            try: 
                                                ibs_boot = integrated_brier_score(y_train_ref, y_boot, surv_matrix_boot, eval_times_boot)
                                            except Exception: 
                                                pass
                                            
                                            # RMSE
                                            try:
                                                km_t_b, km_p_b = kaplan_meier_estimator(y_boot["event"], y_boot["time"])
                                                mask_b = (eval_times_boot >= km_t_b.min()) & (eval_times_boot <= km_t_b.max())
                                                if mask_b.sum() > 1:
                                                    km_interp_b = np.interp(eval_times_boot[mask_b], km_t_b, km_p_b)
                                                    mean_pred_b = np.nanmean(surv_matrix_boot[:, mask_b], axis=0)
                                                    mask_comp_b = ~np.isnan(mean_pred_b) & ~np.isnan(km_interp_b)
                                                    if mask_comp_b.sum() > 0:
                                                        rmse_boot = np.sqrt(np.mean((mean_pred_b[mask_comp_b] - km_interp_b[mask_comp_b]) ** 2))
                                            except Exception: 
                                                pass
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                if boot_valid: 
                    boot_individual_results.append({
                        "c_index": c_boot, 
                        "cd_auc": mean_auc_boot, 
                        "ibs": ibs_boot, 
                        "rmse": rmse_boot
                    })
                    successful_bootstraps += 1
                    
                if (i + 1) % 100 == 0: 
                    print(f"  Bootstrap {i + 1}/{n_bootstraps} complete... (Success rate: {successful_bootstraps / (i + 1) * 100:.1f}%)")
                    
            except Exception as e_main_boot:
                if (i + 1) % 100 == 0:  # Only print occasional errors to avoid spam
                    print(f"  Bootstrap {i + 1} failed: {e_main_boot}")
                 
        print(f"\n✅ Bootstrap complete! {successful_bootstraps}/{n_bootstraps} samples succeeded.")
        # --- 9.3.3: Calculate and Print CIs ---
        results_final = {}
        metrics_orig = {"C-index": c_index_orig, "CD-AUC": mean_auc_orig, "IBS": ibs_orig, "RMSE": rmse_orig}
        metric_map_ci = {"c_index": "C-index", "cd_auc": "CD-AUC", "ibs": "IBS", "rmse": "RMSE"}
        final_boot_values = {key: [d[key] for d in boot_individual_results if key in d and not np.isnan(d[key])] 
                             for key in metric_map_ci.keys()}
        
        for key, display_name in metric_map_ci.items():
            boot_values = final_boot_values[key]
            point_estimate = metrics_orig[display_name]
            if len(boot_values) < 20: 
                print(f"⚠️ Insufficient samples ({len(boot_values)}) to calculate CI for: {display_name}")
                results_final[display_name] = f"{point_estimate:.4f} (N/A)" if not np.isnan(point_estimate) else "N/A"
            else:
                ci_lower, ci_upper = np.percentile(boot_values, [2.5, 97.5])
                results_final[display_name] = f"{point_estimate:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})"
        
        print("\n" + "="*60)
        print("📊 **Final Model Performance on 20% Test Set (95% CI):**")
        print("="*60)
        for name, result_str in results_final.items():
             print(f"{name:<28}: {result_str}")
        print("="*60)
            
       # except Exception as e:
           # print(f"\n❌ Error during test set evaluation: {e}"); traceback.print_exc()

# --- 9.4 Extract sksurv HR (based on 80% train) ---
if final_model_fitted:
    print("\n--- 9.4: Extracting final model HR (from 80% train set)... ---")
    try:
        preprocess_step = final_pipeline.named_steps['preprocess']
        feature_names_processed = preprocess_step.get_feature_names_out()
        coefs = final_pipeline.named_steps['model'].model_.coef_ # Access model inside wrapper
        hr_df_final = pd.DataFrame({"Variable": feature_names_processed, "Coefficient (logHR)": coefs})
        hr_df_final["Hazard Ratio (HR)"] = np.exp(hr_df_final["Coefficient (logHR)"])
        hr_df_final = hr_df_final.reindex(hr_df_final['Coefficient (logHR)'].abs().sort_values(ascending=False).index)
        excel_path_final = OUTPUT_DIR / "Final_CoxPH_Model_HR_Results_80Train_IterImp.xlsx"
        hr_df_final.to_excel(excel_path_final, index=False)
        print(f"📂 Final model HR table saved to: {excel_path_final}")
        print("--- Final Model HR Preview ---"); print(hr_df_final.round(3))
    except Exception as e:
        print(f"❌ Error extracting HRs: {e}"); traceback.print_exc()
else:
     print("\n⚠️ Skipping HR extraction (final model did not fit).")

# --- 9.5 Extract lifelines Detailed Stats (based on 80% train) ---
print("\n--- 9.5: Extracting detailed stats with lifelines (from 80% train set)... ---")
if final_model_fitted:
    try:
        preprocess_step = final_pipeline.named_steps['preprocess']
        # We need to use the .transform from the *fitted* pipeline
        X_train_full_processed = preprocess_step.transform(X_train_full)
        feature_names_processed = preprocess_step.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_train_full_processed, columns=feature_names_processed, index=X_train_full.index)
        
        df_lifelines = X_processed_df.copy()
        df_lifelines['time'] = y_train_full['time']
        df_lifelines['event'] = y_train_full['event']

        print("Fitting equivalent model in lifelines (penalizer=0)...")
        from lifelines import CoxPHFitter
        cph_lifelines = CoxPHFitter(penalizer=0) # Match alpha=0
        try:
            cph_lifelines.fit(df_lifelines, duration_col='time', event_col='event')
            print("✅ lifelines model fitted successfully.")
            
            print("\n--- Final Model Detailed Statistics (from lifelines) ---")
            final_summary = cph_lifelines.summary
            final_summary['-log10(p)'] = -np.log10(final_summary['p'].clip(lower=1e-300))
            cols_ordered = ['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%', 
                            'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p', '-log10(p)']
            cols_to_show = [col for col in cols_ordered if col in final_summary.columns]
            final_summary_display = final_summary[cols_to_show].sort_values(by='p')
            print(final_summary_display.round(4))
            
            excel_path_lifelines = OUTPUT_DIR / "Final_CoxPH_Model_Detailed_Summary_lifelines_80Train_IterImp.xlsx"
            final_summary_display.to_excel(excel_path_lifelines, index=True) 
            print(f"\n📂 Detailed stats table saved to: {excel_path_lifelines}")
        except Exception as e_ll: 
            print(f"❌ Failed to fit or summarize with lifelines: {e_ll}"); traceback.print_exc()
    except Exception as e_prep: 
        print(f"❌ Error preparing data for lifelines: {e_prep}"); traceback.print_exc()
else:
     print("\n⚠️ Skipping lifelines analysis (final model did not fit).")


# %%
# ==================================
# 10. Training Set KM Curve
# ==================================
print("\n" + "="*60)
print("--- Step 10: Generating Training Set KM Curve ---")
print("="*60)

if final_model_fitted:
    try:
        print("Loading lifelines summary to build scorecard...")
        # Note: This assumes the lifelines summary file from Step 9.5 was successfully created.
        # A more robust way would be to pass 'final_summary_display' directly.
        # For simplicity, we re-load the file.
        lifelines_summary_path = OUTPUT_DIR / "Final_CoxPH_Model_Detailed_Summary_lifelines_80Train_IterImp.xlsx"
        lifelines_summary = pd.read_excel(lifelines_summary_path, index_col=0)
        
        print("Calculating scores and thresholds on 80% training data...")
        significant_vars_summary = lifelines_summary[lifelines_summary['p'] < 0.05].copy()
        if significant_vars_summary.empty: 
            print("❌ No significant variables (p < 0.05) found. Cannot create scorecard for KM plot.")
        else:
            MAX_POINTS = 100
            max_abs_coef = significant_vars_summary['coef'].abs().max()
            if max_abs_coef == 0: raise ValueError("Max coefficient is zero.")
            
            significant_vars_summary['Points'] = (significant_vars_summary['coef'] / max_abs_coef * MAX_POINTS).round().astype(int)
            risk_scorecard_points = significant_vars_summary[['Points']]
            scorecard_lifelines_names = risk_scorecard_points.index.tolist()
            
            # --- We need the processed training data (X_train_full_processed) and original df_train_source ---
            # Let's re-create df_train_source (since it was redefined in a previous cell)
            df_train_source_km = df.iloc[train_idx].copy() # Get the 80% data again
            df_train_source_km['time'] = df_train_source_km[TIME_COL]
            df_train_source_km['event'] = df_train_source_km[EVENT_COL].astype(bool)

            # We already have X_train_full_processed (as X_processed_df) and actual_feature_names from Step 9.5
            # Let's ensure we use the correct processed data and feature names
            X_train_processed_df = df_lifelines.drop(columns=['time', 'event'])
            actual_feature_names = X_train_processed_df.columns.tolist()

            # --- Create Name Mapping ---
            print("Creating feature name mapping...")
            actual_feature_names_mapping = {}
            missing_in_actual = []
            for ll_name in scorecard_lifelines_names:
                if ll_name in actual_feature_names:
                    actual_feature_names_mapping[ll_name] = ll_name
                else:
                    base_name_ll = ll_name.split('__')[-1]
                    matched_actual_name = [actual for actual in actual_feature_names if actual.endswith(base_name_ll)]
                    if len(matched_actual_name) == 1:
                        actual_feature_names_mapping[ll_name] = matched_actual_name[0]
                    else:
                         missing_in_actual.append(ll_name)
            if missing_in_actual:
                 raise ValueError(f"Scorecard mapping failed: {missing_in_actual}")
            else:
                 print("✅ Feature name mapping successful.")

            # --- Calculate Scores & Thresholds ---
            score_features_actual = [actual_feature_names_mapping[ll_name] for ll_name in scorecard_lifelines_names]
            X_train_significant = X_train_processed_df[score_features_actual]
            points_vector = risk_scorecard_points.loc[scorecard_lifelines_names, 'Points'].values
            train_total_risk_scores = X_train_significant.dot(points_vector) 
            
            df_train_source_km['Total_Risk_Score'] = train_total_risk_scores
            
            q1, q2 = train_total_risk_scores.quantile([0.33, 0.66]).tolist()
            print(f"✅ Risk thresholds (q1, q2) calculated: {q1:.2f}, {q2:.2f}")

            def assign_risk_group(score, q1_thresh, q2_thresh):
                if pd.isna(score): return 'Unknown'
                if score <= q1_thresh: return 'Low Risk'
                elif score <= q2_thresh: return 'Medium Risk'
                else: return 'High Risk'
            df_train_source_km['Risk_Group'] = df_train_source_km['Total_Risk_Score'].apply(lambda x: assign_risk_group(x, q1, q2))
            print("✅ Risk groups assigned to training data.")
            print(df_train_source_km['Risk_Group'].value_counts())

            # --- Plot KM Curve ---
            print("\nPlotting Training Set KM Curve...")
            T_train_km = df_train_source_km['time']
            E_train_km = df_train_source_km['event']
            groups_train_km = df_train_source_km['Risk_Group']
            valid_groups_train = groups_train_km.unique()
            
            if len(valid_groups_train) < 2:
                print("❌ Not enough risk groups to plot KM curve.")
            else:
                kmf_dict = {} 
                color_map = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"}
                risk_group_order = ['Low Risk', 'Medium Risk', 'High Risk'] 
                
                for group_name in risk_group_order:
                    if group_name in valid_groups_train:
                        mask = (groups_train_km == group_name)
                        if mask.sum() > 0:
                            kmf = KaplanMeierFitter()
                            kmf.fit(T_train_km[mask], event_observed=E_train_km[mask], label=f"{group_name} (n={mask.sum()})")
                            kmf_dict[group_name] = kmf
                
                plt.figure(figsize=(8, 10))
                ax = plt.subplot(111)
                
                if 'Low Risk' in kmf_dict: kmf_dict['Low Risk'].plot(ax=ax, ci_show=True, color=color_map["Low Risk"])
                if 'Medium Risk' in kmf_dict: kmf_dict['Medium Risk'].plot(ax=ax, ci_show=True, color=color_map["Medium Risk"])
                if 'High Risk' in kmf_dict: kmf_dict['High Risk'].plot(ax=ax, ci_show=True, color=color_map["High Risk"])
                
                groups_to_test = [g for g in risk_group_order if g in kmf_dict]
                p_value_logrank = np.nan
                if len(groups_to_test) >= 2:
                     try:
                         df_for_test = df_train_source_km[df_train_source_km['Risk_Group'].isin(groups_to_test)]
                         logrank_result = multivariate_logrank_test(df_for_test['time'], df_for_test['Risk_Group'], df_for_test['event'])
                         p_value_logrank = logrank_result.p_value
                         print(f"\n✅ Log-rank test p-value: {p_value_logrank:.4f}")
                     except Exception as e_lr:
                          print(f"⚠️ Log-rank test failed: {e_lr}"); traceback.print_exc()
                
                plt.title("Kaplan-Meier Curves by Predicted Risk Groups (80% Training Set)", fontsize=14)
                plt.xlabel("Follow-up Time (Years)", fontsize=12)
                plt.ylabel("Estimated Survival Probability", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend(title="Risk Group", loc="lower left", fontsize=10)
                plt.xlim(0, 4.5); plt.xticks(ticks=[0, 2, 4])
                plt.ylim(0.2, 1.05); plt.yticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
                
                if not pd.isna(p_value_logrank):
                    p_text = f"Log-rank test, p < 0.001" if p_value_logrank < 0.001 else f"Log-rank test, p = {p_value_logrank:.3f}"
                    plt.text(0.95, 0.25, p_text, ha='right', va='bottom', transform=ax.transAxes, fontsize=10)
                
                try:
                    kmfs_for_table = [kmf_dict.get(group) for group in risk_group_order if group in kmf_dict]
                    if len(kmfs_for_table) >= 2:
                        add_at_risk_counts(*kmfs_for_table, ax=ax, fontsize=10)
                except Exception as e_at:
                    print(f"⚠️ Error adding at-risk counts: {e_at}")
                
                plt.tight_layout(pad=1.5, rect=[0, 0.15, 1, 0.95]) 
                km_output_path = OUTPUT_DIR / "CHARLS_TrainingSet_KM_Curve.png"
                plt.savefig(km_output_path, dpi=300)
                plt.show()
                print(f"✅ Training Set KM curve saved to: {km_output_path}")

    except Exception as e:
        print(f"❌ Error generating Training Set KM curve: {e}"); traceback.print_exc()
else:
     print("\n⚠️ Skipping Training Set KM Curve (final model did not fit).")


# %%
# ==================================
# 11. Training Set DCA
# ==================================
print("\n" + "="*60)
print("--- Step 11: Generating Training Set DCA ---")
print("="*60)

if final_model_fitted:
    try:
        print("Calculating risk scores for DCA...")
        TIME_HORIZON_DCA = 4.0  # Use 4-year for CHARLS

        # Use the full 80% training features (X_train_full) so lengths/index align with df_train_source_km
        if hasattr(final_pipeline, 'predict_survival_function'):
            surv_funcs_dca = final_pipeline.predict_survival_function(X_train_full)
            survival_probs_dca = np.array([fn(TIME_HORIZON_DCA) for fn in surv_funcs_dca])
            risk_probs_dca = 1 - survival_probs_dca

            # Align by index to avoid length mismatch
            risk_series = pd.Series(risk_probs_dca, index=X_train_full.index)

            # If df_train_source_km was created from the same train_idx, reindex will align correctly
            df_train_source_km['model_risk_dca'] = risk_series.reindex(df_train_source_km.index)

            # Warn if any values are missing after alignment
            if df_train_source_km['model_risk_dca'].isna().any():
                print("⚠️ Some model_risk_dca values are missing after alignment. Check train indices and X_train_full.")
        else:
            # Fallback: try pipeline.predict (risk score) and scale to probability-like score if needed
            print("⚠️ final_pipeline has no predict_survival_function, falling back to risk-score based DCA.")
            risk_scores = final_pipeline.predict(X_train_full)
            # convert risk score to pseudo-probability by min-max (only for plotting/DCA; document this)
            ps = (risk_scores - np.nanmin(risk_scores)) / (np.nanmax(risk_scores) - np.nanmin(risk_scores) + 1e-12)
            df_train_source_km['model_risk_dca'] = pd.Series(ps, index=X_train_full.index).reindex(df_train_source_km.index)

        print("Running dcurves...")
        dca_results = dca(
            data=df_train_source_km[['outcome_dca', 'model_risk_dca']],
            outcome='outcome_dca',
            modelnames=['model_risk_dca'],
            thresholds=np.arange(0.01, 0.61, 0.01)
        )
        
        print("Plotting DCA...")
        df_m = dca_results[dca_results['model'] == 'model_risk_dca']
        df_all = dca_results[dca_results['model'] == 'all']
        df_none = dca_results[dca_results['model'] == 'none']
        pt = df_m['threshold'].values
        nb_m = df_m['net_benefit'].values
        nb_all = df_all['net_benefit'].values
        nb_none = df_none['net_benefit'].values
        
        nb_m_s = lowess(nb_m, pt, frac=0.25, it=0)[:,1] # Smoothed
        
        plt.figure(figsize=(7,6))
        plt.plot(pt, nb_m_s, lw=2.5, color='#0072B2', label='CoxPH Model (Training)')
        plt.plot(pt, nb_all, lw=2, color='#E69F00', label='Treat All')
        plt.axhline(0, color='gray', lw=1.3, ls='--', label='Treat None')
        
        plt.title(f'Decision Curve Analysis (80% Training Set, {TIME_HORIZON_DCA}-Year)', fontsize=14)
        plt.xlabel('Threshold Probability', fontsize=12); plt.ylabel('Net Benefit', fontsize=12)
        plt.xlim(0.0, 0.60); plt.ylim(-0.1, 0.20)
        plt.xticks(ticks=np.arange(0.0, 0.61, 0.1)); plt.yticks(ticks=np.arange(-0.1, 0.21, 0.1))
        plt.grid(True, ls='--', alpha=0.5); plt.legend(loc='upper right', fontsize=11, frameon=False)
        
        prev = df_train_source_km['outcome_dca'].mean()
        if prev > 0 and prev < plt.xlim()[1]:
            plt.axvline(prev, ls=':', lw=1.5, color='gray')
            plt.text(prev, plt.ylim()[0] + 0.005, f'Prevalence ≈ {prev:.3f}\n(Treat-all NB=0)', ha='left', va='bottom', fontsize=9)
            print(f"ℹ️ Training set {TIME_HORIZON_DCA}-year prevalence: {prev:.4f}")
        
        # Calculate and plot Zero-NB and Highlight
        pt_start, pt_end = np.nan, np.nan
        diff_model_vs_all = nb_m_s - nb_all
        sign_change_all = np.where(np.sign(diff_model_vs_all[:-1]) != np.sign(diff_model_vs_all[1:]))[0]
        start_idx_list = [i for i in sign_change_all if diff_model_vs_all[i] < 0 and diff_model_vs_all[i+1] > 0]
        if len(start_idx_list) > 0:
            i = start_idx_list[0]; x0, x1 = pt[i], pt[i+1]; y0, y1 = diff_model_vs_all[i], diff_model_vs_all[i+1]
            pt_start = x0 - y0 * (x1 - x0) / (y1 - y0)
        elif diff_model_vs_all[0] > 0:
            pt_start = pt[0]
            
        diff_model_vs_none = nb_m_s
        sign_change_none = np.where(np.sign(diff_model_vs_none[:-1]) != np.sign(diff_model_vs_none[1:]))[0]
        end_idx_list = [i for i in sign_change_none if diff_model_vs_none[i] > 0 and diff_model_vs_none[i+1] < 0]
        if len(end_idx_list) > 0:
            i = end_idx_list[0]; x0, x1 = pt[i], pt[i+1]; y0, y1 = diff_model_vs_none[i], diff_model_vs_none[i+1]
            pt_end = x0 - y0 * (x1 - x0) / (y1 - y0)
            if pt_end < plt.xlim()[1]:
                plt.scatter([pt_end],[0], s=50, color='#0072B2')
                plt.text(pt_end + 0.002, 0.001, f'Zero-NB ≈ {pt_end:.2f}', ha='left', va='bottom', fontsize=9, color='#0072B2')
        elif (nb_m_s > 0).all():
            pt_end = pt[-1]
            
        if not np.isnan(pt_start) and not np.isnan(pt_end) and pt_start < pt_end:
            plt.axvspan(pt_start, pt_end, color='#0072B214', label='Model Provides Benefit')
            plt.legend(loc='upper right', fontsize=11, frameon=False)
            print(f"ℹ️ Model benefit range (Training): {pt_start:.3f} - {pt_end:.3f}")
        else:
            print("⚠️ No clear benefit range found for the model on the training set.")
            
        out_path = OUTPUT_DIR / "CHARLS_DCA_TrainingSet.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✅ Training Set DCA plot saved to: {out_path}")
    except Exception as e:
        print(f"❌ Error generating Training Set DCA: {e}"); traceback.print_exc()
else:
     print("\n⚠️ Skipping Training Set DCA (final model did not fit).")

# %%
# ==================================
# 12. Training Set ROC
# ======= Training Set ROC (cleaned) =======
print("\n" + "="*60)
print("--- Step 12: Generating Training Set ROC Curve ---")
print("="*60)

if final_model_fitted:
    try:
        # Ensure we have df_train_source_km (80% train source)
        if 'df_train_source_km' not in locals():
            df_train_source_km = df.iloc[train_idx].copy()
            df_train_source_km['time'] = df_train_source_km[TIME_COL]
            df_train_source_km['event'] = df_train_source_km[EVENT_COL].astype(bool)

        # Prepare processed training matrix and feature names
        preprocess_step = final_pipeline.named_steps['preprocess']
        X_train_full_processed = preprocess_step.transform(X_train_full)
        feature_names_processed = preprocess_step.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_train_full_processed,
                                      columns=feature_names_processed,
                                      index=X_train_full.index)

        # Build lifelines summary if not present
        if 'lifelines_summary' not in locals():
            df_lifelines_tmp = X_processed_df.copy()
            df_lifelines_tmp['time'] = y_train_full['time']
            df_lifelines_tmp['event'] = y_train_full['event']
            cph_tmp = CoxPHFitter(penalizer=0)
            cph_tmp.fit(df_lifelines_tmp, duration_col='time', event_col='event')
            lifelines_summary = cph_tmp.summary

        significant_vars_summary = lifelines_summary[lifelines_summary['p'] < 0.05].copy()
        if significant_vars_summary.empty:
            raise ValueError("No significant variables (p < 0.05) found for ROC scorecard.")

        risk_scorecard_coef = significant_vars_summary[['coef']]
        scorecard_lifelines_names = risk_scorecard_coef.index.tolist()

        # map lifelines names to processed feature names
        processed_names = list(feature_names_processed)
        actual_feature_names_mapping = {}
        missing_in_actual = []
        for ll_name in scorecard_lifelines_names:
            if ll_name in processed_names:
                actual_feature_names_mapping[ll_name] = ll_name
            else:
                base = ll_name.split('__')[-1]
                matched = [n for n in processed_names if n.endswith(base)]
                if len(matched) == 1:
                    actual_feature_names_mapping[ll_name] = matched[0]
                else:
                    missing_in_actual.append(ll_name)
        if missing_in_actual:
            raise ValueError(f"ROC Scorecard mapping failed: {missing_in_actual}")

        score_features_actual = [actual_feature_names_mapping[n] for n in scorecard_lifelines_names]
        X_train_significant = X_processed_df[score_features_actual]
        coef_vector = risk_scorecard_coef.loc[scorecard_lifelines_names, 'coef'].values

        # compute linear predictor and align indices
        train_total_risk_scores = X_train_significant.dot(coef_vector)
        if not isinstance(train_total_risk_scores, pd.Series):
            train_total_risk_scores = pd.Series(train_total_risk_scores, index=X_train_full.index)

        # align to df_for_analysis (prefer df_train_source_km)
        df_for_analysis = df_train_source_km.copy()
        df_for_analysis['Total_Risk_Score_Coef'] = train_total_risk_scores.reindex(df_for_analysis.index)

        # ensure time/event aligned
        time_series = pd.Series(y_train_full['time'], index=X_train_full.index)
        event_series = pd.Series(y_train_full['event'].astype(bool), index=X_train_full.index)
        df_for_analysis['time'] = time_series.reindex(df_for_analysis.index).astype(float)
        df_for_analysis['event'] = event_series.reindex(df_for_analysis.index).astype(bool)

        print("✅ ROC data prepared and aligned.")

        TIME_HORIZON_ROC = 4.0  # 4-year
        y_true = (df_for_analysis['time'] <= TIME_HORIZON_ROC) & (df_for_analysis['event'])
        y_score = df_for_analysis['Total_Risk_Score_Coef']

        valid_mask = y_score.notna() & df_for_analysis['time'].notna() & df_for_analysis['event'].notna()
        if valid_mask.sum() == 0:
            raise ValueError("No valid samples with both score and outcome for ROC.")

        if y_true[valid_mask].sum() < 1:
            print(f"❌ No events at {TIME_HORIZON_ROC} years. Cannot calculate ROC.")
        else:
            fpr, tpr, thresholds = roc_curve(y_true[valid_mask].astype(int), y_score[valid_mask])
            roc_auc = auc(fpr, tpr)
            j_statistic = tpr - fpr
            best_idx = np.nanargmax(j_statistic)
            best_cutoff = thresholds[best_idx]
            sensitivity = tpr[best_idx]
            specificity = 1 - fpr[best_idx]
            print(f"✅ ROC Analysis (Training Set): AUC = {roc_auc:.4f}")
            print(f"   Best Cutoff (logHR): {best_cutoff:.4f} -> Sens={sensitivity:.2f}, Spec={specificity:.2f}")

            # plot ROC
            palette = sns.color_palette("colorblind")
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color=palette[0], lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
            plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color=palette[3], s=120, zorder=3,
                        label=f'Best Cutoff = {best_cutoff:.2f}\n(Sens={sensitivity:.2f}, Spec={specificity:.2f})')
            plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
            plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=14)
            plt.ylabel('Sensitivity (True Positive Rate)', fontsize=14)
            plt.title(f'ROC Curve for {TIME_HORIZON_ROC:.1f}-Year Risk (80% Training Set)', fontsize=14, weight='bold')
            plt.legend(loc="lower right", fontsize=12); plt.grid(True, linestyle='--', alpha=0.6); sns.despine()

            output_path = OUTPUT_DIR / "CHARLS_Training_Set_ROC_Curve.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Training Set ROC curve saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error generating Training Set ROC: {e}")
        traceback.print_exc()
else:
    print("\n⚠️ Skipping Training Set ROC (final model did not fit).")
# %%
#Calibration_Curve
print("\n" + "="*60)
print("--- Step 13: Training-set Calibration Curve (4-year) ---")
print("="*60)

try:
    if not final_model_fitted:
        print("⚠️ Skipping calibration (final model not fitted).")
    else:
        TIME_HORIZON_CAL = 4.0

        # Prepare true outcomes on training set (80% used for final fit)
        # y_train_full is a structured array (sksurv). Use same logic as other sections.
        y_train_times = y_train_full["time"]
        y_train_events = y_train_full["event"].astype(bool)
        y_true_train = (y_train_times <= TIME_HORIZON_CAL) & (y_train_events)

        # Obtain predicted probability of event by TIME_HORIZON_CAL:
        # Prefer predict_survival_function; fallback to score -> min-max to pseudo-prob.
        try:
            if hasattr(final_pipeline, "predict_survival_function"):
                surv_funcs_train = final_pipeline.predict_survival_function(X_train_full)
                # Some wrappers may return callables; evaluate at TIME_HORIZON_CAL
                survival_probs_train = np.array([fn(TIME_HORIZON_CAL) for fn in surv_funcs_train])
                prob_event_train = 1.0 - survival_probs_train
            else:
                raise AttributeError("final_pipeline lacks predict_survival_function")
        except Exception:
            # fallback: use risk score and map to [0,1] for calibration plotting (documented approximation)
            scores_train = final_pipeline.predict(X_train_full)
            min_s, max_s = np.nanmin(scores_train), np.nanmax(scores_train)
            denom = (max_s - min_s) if (max_s - min_s) != 0 else 1.0
            prob_event_train = (scores_train - min_s) / denom
            print("⚠️ Used min-max scaled risk scores as pseudo-probabilities for calibration (fallback).")

        # Align lengths / mask invalids
        idx_train = np.arange(len(y_true_train))
        mask_valid = (~np.isnan(prob_event_train)) & (~np.isnan(y_train_times))
        if mask_valid.sum() == 0:
            raise ValueError("No valid samples for calibration after alignment.")

        y_true_valid = y_true_train[mask_valid].astype(int)
        prob_pred_valid = np.array(prob_event_train)[mask_valid]

        # Compute calibration curve (binning by quantiles -> stable for unbalanced events)
        try:
            prob_true_bins, prob_pred_bins = calibration_curve(y_true_valid, prob_pred_valid,
                                                               n_bins=10, strategy='quantile')
        except Exception as e_cal:
            # fallback to uniform bins
            prob_true_bins, prob_pred_bins = calibration_curve(y_true_valid, prob_pred_valid,
                                                               n_bins=10, strategy='uniform')
            print(f"⚠️ calibration_curve quantile failed, used uniform bins: {e_cal}")

        # Calibration slope & intercept (linear regression of observed ~ predicted)
        cal_slope, cal_intercept = (np.nan, np.nan)
        try:
            if len(prob_pred_bins) > 1:
                lr = LinearRegression()
                lr.fit(prob_pred_bins.reshape(-1, 1), prob_true_bins)
                cal_slope = lr.coef_[0]
                cal_intercept = lr.intercept_
            else:
                cal_slope, cal_intercept = (np.nan, np.nan)
        except Exception as e_lr:
            print(f"⚠️ Calibration slope/intercept calculation failed: {e_lr}")

        # Print summary
        print(f"\nCalibration (Training set, {TIME_HORIZON_CAL:.1f} years):")
        print(f"  Samples used for calibration: {mask_valid.sum()}")
        print(f"  Observed event rate at {TIME_HORIZON_CAL:.1f}y: {y_true_valid.mean():.4f}")
        if not np.isnan(cal_intercept):
            print(f"  Calibration intercept = {cal_intercept:.4f} (ideal 0)")
        if not np.isnan(cal_slope):
            print(f"  Calibration slope     = {cal_slope:.4f} (ideal 1)")

        # Plot calibration curve
        plt.figure(figsize=(7.5, 7.5))
        plt.plot(prob_pred_bins, prob_true_bins, marker='o', linestyle='-', label='Observed vs Predicted')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
        # Optionally plot regression line if available
        try:
            if not np.isnan(cal_slope):
                x_line = np.linspace(0, 1, 100)
                y_line = cal_intercept + cal_slope * x_line
                plt.plot(x_line, y_line, color='C1', linestyle=':', label=f'Calibration fit (slope={cal_slope:.2f})')
        except Exception:
            pass

        plt.xlabel(f"Predicted probability of event by {TIME_HORIZON_CAL:.1f} years", fontsize=12)
        plt.ylabel("Observed event rate (in bin)", fontsize=12)
        plt.title(f"Calibration Curve (Training set, {TIME_HORIZON_CAL:.1f}-Year)", fontsize=14)
        plt.grid(True, ls='--', alpha=0.6)
        plt.legend(loc='upper left')
        plt.xlim(0, min(1.0, max(np.max(prob_pred_bins), np.max(prob_true_bins), 0.4)))
        plt.ylim(0, min(1.0, max(np.max(prob_pred_bins), np.max(prob_true_bins), 0.4)))

        # Save figure
        cal_path = OUTPUT_DIR / f"CHARLS_TrainingSet_Calibration_{int(TIME_HORIZON_CAL)}y.png"
        plt.savefig(cal_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Calibration curve saved to: {cal_path}")

except Exception as e:
    print(f"❌ Error generating calibration curve: {e}")
    traceback.print_exc()
# %%
