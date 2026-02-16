#!/usr/bin/env python
# coding: utf-8

"""
Model Validation Script (External JSTAR Cohort)

This script performs a comprehensive validation of a pre-trained CoxPH model 
on the external JSTAR dataset.
It executes four main analyses:
1.  Calculates survival performance metrics (C-Index, CD-AUC, IBS).
2.  Performs risk stratification (scorecard + MaxStat) and plots a 
    2-year cumulative incidence bar chart.
3.  Performs and plots Decision Curve Analysis (DCA) for 2-year risk.
"""

import os
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dcurves import dca
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import t
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
from statsmodels.nonparametric.smoothers_lowess import lowess

# --- Warnings Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="The objective` survival:cox` is deprecated")

# --- Relative Path Configuration ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Path Constants ---
TRAIN_PATH = DATA_DIR / "charls_model_ready.csv"
VALID_PATH = DATA_DIR / "sample_jstar_validate_ready.csv" 

LIFELINES_SUMMARY_PATH = (
    OUTPUT_DIR / "table3_Final_CoxPH_Model_Detailed_Summary_lifelines_80Train_IterImp.xlsx"
)

# --- Global Helper Function ---
def load_csv_auto(path):
    """Tries to load a CSV with multiple common encodings."""
    encodings = ["utf-8-sig", "utf-8", "cp932", "gbk", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise FileNotFoundError(f"Could not read file with any encoding: {path}")


# ==================================
# ANALYSIS 1: PERFORMANCE METRICS (C-Index, CD-AUC, IBS)
# ==================================
def run_external_validation_metrics():
    """
    Fits the sksurv CoxPH pipeline and calculates
    C-Index, time-dependent AUC, and IBS for JSTAR.
    """
    print("\n--- 1. Running Performance Metrics (C-Index, CD-AUC, IBS) [JSTAR] ---")

    # --- 1.1 Define Variables & Preprocessing ---
    num_features = ["orient", "memory", "grip_max"]
    cat_features = ["sex", "edu_3cat", "age_group2_safe"]
    all_features = num_features + cat_features
    TIME_COL, EVENT_COL = "surv_time", "event"

    high_missing_continuous = ["grip_max"]
    low_missing_continuous = [x for x in num_features if x not in high_missing_continuous]

    low_miss_num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    high_miss_num_pipeline = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    estimator=BayesianRidge(), random_state=42, max_iter=10
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("low_miss_num", low_miss_num_pipeline, low_missing_continuous),
            ("high_miss_num", high_miss_num_pipeline, high_missing_continuous),
            ("cat", cat_pipeline, cat_features),
        ]
    )

    # --- 1.2 Load Training Data ---
    try:
        df_train_source = load_csv_auto(TRAIN_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    df_train_source = df_train_source.dropna(subset=[TIME_COL, EVENT_COL])
    X_train = df_train_source[all_features].copy()
    y_train = Surv.from_arrays(
        event=df_train_source[EVENT_COL].astype(bool), time=df_train_source[TIME_COL]
    )

    # --- 1.3 Fit Final Pipeline ---
    print("Fitting pipeline for performance metrics...")
    final_cox_model = CoxPHSurvivalAnalysis(alpha=0.01, ties="efron")
    final_pipeline = Pipeline([("preprocess", preprocess), ("model", final_cox_model)])
    final_pipeline.fit(X_train, y_train)
    print(" Pipeline fitting complete.")

    # --- 1.4 Load External Validation Set (JSTAR) ---
    try:
        df_validation_source = load_csv_auto(VALID_PATH)
    except FileNotFoundError as e:
        print(e)
        return
    
    # Check for new categories
    print("\n🔍 Checking for new categories in JSTAR:")
    for c in cat_features:
        new_cats = set(df_validation_source[c].dropna().unique()) - set(df_train_source[c].dropna().unique())
        if new_cats:
            print(f"{c} has new categories in validation: {new_cats}")

    df_validation_source = df_validation_source.dropna(subset=[TIME_COL, EVENT_COL])
    X_validation = df_validation_source[all_features].copy()
    y_validation = Surv.from_arrays(
        event=df_validation_source[EVENT_COL].astype(bool),
        time=df_validation_source[TIME_COL],
    )

    # --- 1.5 External Validation Performance ---
    print("Calculating external validation performance...")
    try:
        risk_scores_orig = final_pipeline.predict(X_validation)
        event_mask_val = y_validation["event"]
        times_val = y_validation["time"]
        lower_b, upper_b = np.percentile(times_val[event_mask_val], [10, 90])
        eval_upper_bound = min(upper_b, y_train["time"].max())
        eval_times = np.linspace(lower_b, eval_upper_bound, 100)

        try:
            c_index_orig = concordance_index_ipcw(
                y_train, y_validation, risk_scores_orig, tau=eval_upper_bound
            )[0]
        except Exception:
            print("IPCW C-index failed, using Harrell's C-index.")
            c_index_orig = concordance_index_censored(
                y_validation["event"], y_validation["time"], -risk_scores_orig
            )[0]

        aucs_orig, _ = cumulative_dynamic_auc(
            y_train, y_validation, risk_scores_orig, eval_times
        )
        mean_auc_orig = np.nanmean(aucs_orig)

        X_val_transformed = final_pipeline.named_steps["preprocess"].transform(
            X_validation
        )
        surv_funcs = final_pipeline.named_steps["model"].predict_survival_function(
            X_val_transformed
        )
        surv_matrix = np.vstack([fn(eval_times) for fn in surv_funcs])
        ibs_orig = integrated_brier_score(
            y_train, y_validation, surv_matrix, eval_times
        )

        km_t, km_p = kaplan_meier_estimator(y_validation["event"], y_validation["time"])
        km_interp = np.interp(eval_times, km_t, km_p)
        mean_pred = surv_matrix.mean(axis=0)
        rmse_orig = np.sqrt(np.mean((mean_pred - km_interp) ** 2))

    except Exception as e:
        print(f"Error during external validation calculation: {e}")
        traceback.print_exc()
        c_index_orig, mean_auc_orig, ibs_orig, rmse_orig = [np.nan] * 4

    # --- 1.6 Bootstrap CI ---
    print("Calculating Bootstrap CIs...")
    n_bootstraps = 300
    boot_results = []
    val_indices = np.arange(len(X_validation))
    for i in range(n_bootstraps):
        boot_idx = resample(val_indices)
        X_boot = X_validation.iloc[boot_idx]
        y_boot = y_validation[boot_idx]
        if y_boot["event"].sum() < 2:
            continue
        try:
            risk_boot = final_pipeline.predict(X_boot)
            c_boot = concordance_index_ipcw(
                y_train, y_boot, risk_boot, tau=eval_upper_bound
            )[0]
            aucs, _ = cumulative_dynamic_auc(
                y_train, y_boot, risk_boot, eval_times
            )
            auc_boot = np.nanmean(aucs)
            Xb_t = final_pipeline.named_steps["preprocess"].transform(X_boot)
            survf = final_pipeline.named_steps["model"].predict_survival_function(Xb_t)
            survm = np.vstack([fn(eval_times) for fn in survf])
            ibs_boot = integrated_brier_score(y_train, y_boot, survm, eval_times)
            boot_results.append(
                {"c_index": c_boot, "cd_auc": auc_boot, "ibs": ibs_boot}
            )
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"Completed {i+1}/{n_bootstraps} bootstrap samples...")

    def get_ci(values):
        if len(values) < 20:
            return ("N/A", "N/A")
        return tuple(np.percentile(values, [2.5, 97.5]))

    results_final = {
        "C-index": f"{c_index_orig:.4f} (95% CI: {get_ci([x['c_index'] for x in boot_results])[0]} - {get_ci([x['c_index'] for x in boot_results])[1]})",
        "CD-AUC": f"{mean_auc_orig:.4f} (95% CI: {get_ci([x['cd_auc'] for x in boot_results])[0]} - {get_ci([x['cd_auc'] for x in boot_results])[1]})",
        "IBS": f"{ibs_orig:.4f} (95% CI: {get_ci([x['ibs'] for x in boot_results])[0]} - {get_ci([x['ibs'] for x in boot_results])[1]})",
        "RMSE": f"{rmse_orig:.4f} (CI not computed)",
    }

    print("\n" + "=" * 60)
    print("Final Model Performance on External Validation Set (JSTAR):")
    print("=" * 60)
    for k, v in results_final.items():
        print(f"{k:<25}: {v}")
    print("=" * 60)

    # Save results
    out_path = OUTPUT_DIR / "jstar_validation_results.xlsx"
    pd.DataFrame([results_final]).to_excel(out_path, index=False)
    print(f"Performance metrics saved to: {out_path.resolve()}")

# ==================================
# ANALYSIS 2: RISK STRATIFICATION & CUMULATIVE INCIDENCE PLOT
# ==================================
def run_risk_stratification_plots():
    """
    Loads coefficients, builds scorecard, finds MaxStat groups,
    and plots cumulative incidence bar chart for JSTAR.
    """
    print("\n--- 2. Running Risk Stratification & Incidence Plot [JSTAR] ---")

    # --- 2.1 Define Variables & Preprocessing ---
    num_features = ["orient", "memory", "grip_max"]
    cat_features = ["sex", "edu_3cat", "age_group2_safe"]
    all_features = num_features + cat_features
    TIME_COL, EVENT_COL = "surv_time", "event"

    high_missing_continuous = ["grip_max"]
    low_missing_continuous = [
        col for col in num_features if col not in high_missing_continuous
    ]

    low_miss_num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    high_miss_num_pipeline = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=10,
                    random_state=42,
                    imputation_order="ascending",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("low_miss_num", low_miss_num_pipeline, low_missing_continuous),
            ("high_miss_num", high_miss_num_pipeline, high_missing_continuous),
            ("cat", cat_pipeline, cat_features),
        ],
        remainder="drop",
    )

    # --- 2.2 Load Lifelines Model Results (Dependency) ---
    print(f"Loading lifelines model coefficients from: {LIFELINES_SUMMARY_PATH}")
    try:
        lifelines_summary = pd.read_excel(LIFELINES_SUMMARY_PATH, index_col=0)
    except FileNotFoundError:
        print(f"ERROR: Cannot find required input file: {LIFELINES_SUMMARY_PATH}")
        print("Please run the main training script first to generate this file.")
        return
    except Exception as e:
        print(f"Error reading lifelines summary file: {e}")
        return

    # --- 2.3 Dynamically Calculate Scorecard ---
    print("Calculating risk scorecard from significant variables (p < 0.05)...")
    significant_vars_summary = lifelines_summary[
        lifelines_summary["p"] < 0.05
    ].copy()

    if significant_vars_summary.empty:
        print("ERROR: No significant variables found (p < 0.05). Cannot create scorecard.")
        return
    
    MAX_POINTS = 100
    max_abs_coef = significant_vars_summary["coef"].abs().max()
    if max_abs_coef == 0:
        print("ERROR: Max coefficient is zero. Cannot create score system.")
        return

    significant_vars_summary["Points"] = (
        significant_vars_summary["coef"] / max_abs_coef * MAX_POINTS
    ).round().astype(int)
    risk_scorecard_points = significant_vars_summary[["Points"]]

    rename_fixes = {
        "low__memory": "low_miss_num__memory",
        "low__orient": "low_miss_num__orient",
        "high__grip_max": "high_miss_num__grip_max",
    }
    risk_scorecard_points.index = risk_scorecard_points.index.map(
        lambda x: rename_fixes.get(x, x)
    )

    # --- 2.4 Calculate Risk Thresholds on Training Set ---
    print("Reloading training set to calculate risk thresholds...")
    T_high_risk = np.nan
    try:
        df_train_source = load_csv_auto(TRAIN_PATH)
        X_train = df_train_source[all_features].copy()

        print("Fitting preprocessor on training data...")
        preprocess.fit(X_train)
        X_train_processed = preprocess.transform(X_train)
        feature_names_train_processed = preprocess.get_feature_names_out()
        X_train_processed_df = pd.DataFrame(
            X_train_processed, columns=feature_names_train_processed, index=X_train.index
        )

        available_features = feature_names_train_processed
        rename_map = {}
        for old_name in risk_scorecard_points.index:
            matched = [f for f in available_features if old_name.split("__")[-1] in f]
            if len(matched) == 1:
                rename_map[old_name] = matched[0]
            elif len(matched) > 1:
                prefix = old_name.split("__")[0]
                prefixed_matches = [m for m in matched if prefix in m]
                if prefixed_matches:
                    rename_map[old_name] = prefixed_matches[0]
                else:
                    rename_map[old_name] = matched[0]
            else:
                print(f"Could not map scorecard feature: {old_name}")

        risk_scorecard_points.index = risk_scorecard_points.index.map(
            lambda x: rename_map.get(x, x)
        )
        score_features = risk_scorecard_points.index.tolist()
        print("Final Risk Scorecard (Dynamic):")
        print(risk_scorecard_points)

        missing_score_cols_train = [
            col for col in score_features if col not in X_train_processed_df.columns
        ]
        if missing_score_cols_train:
            raise ValueError(
                f"Training data missing scorecard features: {missing_score_cols_train}"
            )

        X_train_significant = X_train_processed_df[score_features]
        train_total_risk_scores = X_train_significant.dot(
            risk_scorecard_points["Points"]
        )

        q_quantile = 0.80
        T_high_risk = train_total_risk_scores.quantile(q_quantile)
        print(f"High-risk threshold (80th percentile) calculated: {T_high_risk:.2f}")

    except Exception as e:
        print(f"Error calculating risk thresholds: {e}")
        traceback.print_exc()
        return

    # --- 2.5 Load & Process External Validation Set (JSTAR) ---
    print("Loading and processing external validation set (JSTAR)...")
    try:
        df_validation_source = load_csv_auto(VALID_PATH)
        df_validation_source = df_validation_source.dropna(
            subset=[TIME_COL, EVENT_COL]
        ).copy()
        X_validation = df_validation_source[all_features].copy()
        df_validation_source["time"] = df_validation_source[TIME_COL]
        df_validation_source["event"] = df_validation_source[EVENT_COL].astype(bool)
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return

    # --- 2.6 Apply Scorecard & Find MaxStat Cutpoint ---
    print("Applying preprocessor to validation set...")
    try:
        X_validation_processed = preprocess.transform(X_validation)
        feature_names_val_processed = preprocess.get_feature_names_out()
        X_validation_processed_df = pd.DataFrame(
            X_validation_processed,
            columns=feature_names_val_processed,
            index=X_validation.index,
        )
    except Exception as e:
        print(f"Error transforming validation data: {e}")
        traceback.print_exc()
        return

    missing_score_cols_val = [
        col for col in score_features if col not in X_validation_processed_df.columns
    ]
    if missing_score_cols_val:
        print(f"Validation data missing scorecard features: {missing_score_cols_val}")
        return

    X_val_significant = X_validation_processed_df[score_features]
    validation_total_risk_scores = X_val_significant.dot(
        risk_scorecard_points["Points"]
    )
    df_validation_source["Total_Risk_Score"] = validation_total_risk_scores
    print("Risk scores calculated for validation set.")

    # --- MaxStat optimal two-group search ---
    scores_val = df_validation_source["Total_Risk_Score"]
    time_val = df_validation_source["time"]
    event_val = df_validation_source["event"]
    best_p_value = 1.0
    optimal_T = np.nan
    optimal_split_perc = np.nan

    print("[ MaxStat ] Searching for optimal log-rank threshold...")
    test_quantiles = np.linspace(0.10, 0.90, 81)
    thresholds = scores_val.quantile(test_quantiles).unique()

    for T in thresholds:
        if pd.isna(T) or np.isinf(T):
            continue
        group_high_count = (scores_val > T).sum()
        group_low_count = (scores_val <= T).sum()
        if (
            group_high_count < 0.05 * len(df_validation_source)
            or group_low_count < 0.05 * len(df_validation_source)
        ):
            continue

        results = multivariate_logrank_test(
            time_val, (scores_val > T).replace({True: "High", False: "Low"}), event_val
        )
        current_p_value = results.p_value
        if current_p_value < best_p_value:
            best_p_value = current_p_value
            optimal_T = T
            optimal_split_perc = group_high_count / len(df_validation_source)

    if pd.isna(optimal_T):
        print("ERROR: MaxStat could not find a suitable threshold. Using Q80 fallback.")
        optimal_T = T_high_risk # Fallback to Q80
        results = multivariate_logrank_test(
            time_val,  
            (scores_val > optimal_T).replace({True: 'High', False: 'Low'}), 
            event_val
        )
        best_p_value = results.p_value
        optimal_split_perc = (scores_val > optimal_T).sum() / len(df_validation_source)
        
    print(f"Final Optimal T = {optimal_T:.2f}")
    print(f"Final Split (High Risk) = {optimal_split_perc*100:.1f}%")
    print(f"Final Log-rank P = {best_p_value:.5f}")

    # --- Apply two-group (Low/High) stratification ---
    def assign_risk_group_optimal(score, T_thresh):
        if pd.isna(score): return 'Unknown'
        return 'High Risk' if score > T_thresh else 'Low Risk'

    df_validation_source['Risk_Group'] = df_validation_source['Total_Risk_Score'].apply(
        lambda x: assign_risk_group_optimal(x, optimal_T)
    )
    print("Validation set risk stratification complete (2 groups):")
    print(df_validation_source['Risk_Group'].value_counts())

    # --- 2.7 Calculate Log-Rank and KMF objects ---
    print("\n--- Calculating KMF objects and Log-Rank P-value ---")
    
    DATASET_NAME = "JSTAR"
    group_suffix = "2grps"
    
    risk_group_order = ['Low Risk', 'High Risk']
    df_validation_source['Risk_Group'] = pd.Categorical(
        df_validation_source['Risk_Group'], categories=risk_group_order, ordered=True
    )
    
    T_val = df_validation_source['time'].astype(float)
    E_val = df_validation_source['event'].astype(bool)
    groups_val = df_validation_source['Risk_Group']

    valid_groups = [g for g in risk_group_order if (groups_val == g).sum() > 0]
    color_map = {"Low Risk": "green", "High Risk": "red"} 
    kmf_dict = {}

    if len(valid_groups) < 2:
        print("ERROR: Fewer than 2 risk groups. Skipping Bar plot.")
        return
    else:
        # Create KMF objects for bar plot
        for g in risk_group_order:
            if g not in valid_groups:
                continue
            mask = (groups_val == g)
            n_g = int(mask.sum())
            if n_g == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(T_val[mask], event_observed=E_val[mask], label=f"{g} (n={n_g})")
            kmf_dict[g] = kmf

    # --- 2.8 Calculate Cumulative Incidence & Plot Bar Chart ---
    print("\n--- Plotting cumulative incidence bar chart ---")
    end_time = 2.0 # JSTAR uses 2-year endpoint

    incidence_data = {}
    group_n_counts = df_validation_source['Risk_Group'].value_counts()
    total_n = len(df_validation_source)

    for g in valid_groups:
        if g not in kmf_dict: # Make sure kmf was created
            continue
        kmf = kmf_dict[g]
        
        try:
            try:
                survival_df = kmf.survival_function_at_times(end_time)
                ci_df = kmf.confidence_interval_survival_function_at_times(end_time)
            except AttributeError:
                survival_df = kmf.survival_function_
                ci_df = kmf.confidence_interval_

            try:
                S_t = survival_df.loc[survival_df.index >= end_time].iloc[0, 0]
                CI_lower = ci_df.loc[ci_df.index >= end_time].iloc[0, 0]
                CI_upper = ci_df.loc[ci_df.index >= end_time].iloc[0, 1]
            except IndexError:
                S_t = survival_df.iloc[-1, 0]
                CI_lower = ci_df.iloc[-1, 0]
                CI_upper = ci_df.iloc[-1, 1]
            
            if pd.isna(S_t): S_t = survival_df.iloc[-1, 0]
            if pd.isna(CI_lower): CI_lower = ci_df.iloc[-1, 0]
            if pd.isna(CI_upper): CI_upper = ci_df.iloc[-1, 1]

            I_t = 1 - S_t
            I_CI_lower = 1 - CI_upper
            I_CI_upper = 1 - CI_lower
            n_count = group_n_counts.get(g, 0)
            n_perc = (n_count / total_n) if total_n > 0 else 0
            
            incidence_data[g] = {
                'Incidence': I_t,
                'CI_Lower': I_CI_lower,
                'CI_Upper': I_CI_upper,
                'Error_Min': np.clip(I_t - I_CI_lower, 0, None),
                'Error_Max': np.clip(I_CI_upper - I_t, 0, None),
                'N': n_count, 
                'N_Perc': n_perc
            }
            
            print(f" {g} Group {end_time}yr Incidence: {I_t*100:.1f}% (95% CI: {I_CI_lower*100:.1f}-{I_CI_upper*100:.1f}) (n={n_count}, {n_perc*100:.1f}%)")
        except Exception as e:
            print(f" Failed to calculate incidence for {g} group: {e}")
            traceback.print_exc()
            continue

    if incidence_data:
        df_incidence = pd.DataFrame(incidence_data).T
        df_incidence = df_incidence.reindex(risk_group_order)
        
        fig_bar, ax_bar = plt.subplots(figsize=(6, 7))
        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig_bar.patch.set_facecolor('white')
        ax_bar.set_facecolor('white')
        
        errors = df_incidence[['Error_Min', 'Error_Max']].T.values
        bar_colors = [color_map[g] for g in df_incidence.index]
        
        bars = ax_bar.bar(
            df_incidence.index, 
            df_incidence['Incidence'], 
            yerr=errors, 
            capsize=6,
            color=bar_colors,
            ecolor='black',
            width=0.7,
            alpha=0.85
        )
        
        ax_bar.set_ylabel(f"{int(end_time)}-year cumulative incidence (%)", fontsize=12, labelpad=10)
        ax_bar.set_xlabel("Risk Group", fontsize=12) 
        ax_bar.set_title(f"{int(end_time)}-year cumulative incidence by risk group ({DATASET_NAME} validation cohort)", fontsize=14, pad=15)
        
        max_incidence_ci_upper = min(df_incidence['CI_Upper'].max(), 1.0)
        y_upper_limit_base = np.ceil(max_incidence_ci_upper * 10) / 10
        if y_upper_limit_base < 0.20: y_upper_limit_base = 0.20
        y_upper_limit = y_upper_limit_base * 1.20
        y_upper_limit = min(y_upper_limit, 1.0) 

        ax_bar.set_ylim(0, y_upper_limit)
        step = 0.10 if y_upper_limit >= 0.3 else 0.05
        ax_bar.set_yticks(np.arange(0, y_upper_limit * 1.05, step))
        ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.5, color='gray')

        new_x_labels = []
        y_offset_bar = y_upper_limit * 0.015
        
        for i, bar in enumerate(bars):
            I_t = df_incidence['Incidence'].iloc[i]
            ci_upper = df_incidence['CI_Upper'].iloc[i]
            n_count = int(df_incidence['N'].iloc[i]) 
            n_perc = df_incidence['N_Perc'].iloc[i]
            
            label_incidence = f'{I_t*100:.1f}%'
            y_pos_inc = min(ci_upper + y_offset_bar, y_upper_limit * 0.98)
            
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2, 
                y_pos_inc, 
                label_incidence,
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black'
            )
            
            new_x_labels.append(f'{df_incidence.index[i]}\n(n={n_count}, {n_perc*100:.1f}%)')
        
        ax_bar.set_xticklabels(new_x_labels, fontsize=11)
        plt.tight_layout()
        
        filename_bar = OUTPUT_DIR / f"Fig4B_{DATASET_NAME}_Incidence_BarPlot_{int(end_time)}yr_{group_suffix}.png"
        plt.savefig(filename_bar, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_bar) 
        print(f"Final Cumulative Incidence Bar Plot saved to: {filename_bar}")
    else:
        print(" No sufficient data to plot cumulative incidence bar plot.")
    
    print("Bar Plot process finished")

# ==================================
# ANALYSIS 3: DECISION CURVE ANALYSIS (DCA)
# ==================================
# ==============================================================================
# 1. Variables & Preprocessing (Aligned with Main Analysis)
# ==============================================================================
TIME_COL, EVENT_COL = "surv_time", "event"
TIME_HORIZON = 2.0  # JSTAR uses 2-year horizon

num_lowmiss = ["memory", "orient"]
num_highmiss = ["grip_max"]
cat_features = ["sex", "edu_3cat", "age_group2_safe"]
all_features = num_lowmiss + num_highmiss + cat_features

preprocess = ColumnTransformer([
    ("low_miss_num",  Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_lowmiss),
    ("high_miss_num", Pipeline([("imp", IterativeImputer(random_state=42)), ("sc", StandardScaler())]), num_highmiss),
    ("cat",           Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))]), cat_features)
])

# %%
# ==============================================================================
# 2. Model Training & Prediction
# ==============================================================================
print("\n--- Step 1: Training CoxPH model on CHARLS training set ---")
df_train = pd.read_csv(TRAIN_PATH)
y_train = Surv.from_arrays(event=df_train[EVENT_COL].astype(bool), time=df_train[TIME_COL])

final_pipeline = make_pipeline(preprocess, CoxPHSurvivalAnalysis(ties="efron"))
final_pipeline.fit(df_train[all_features], y_train)

print(f"\n--- Step 2: Predicting 2-year risk on JSTAR (n={pd.read_csv(VALID_PATH).shape[0]}) ---")
df_val = pd.read_csv(VALID_PATH)
risk_probs = 1 - np.array([fn(TIME_HORIZON) for fn in final_pipeline.predict_survival_function(df_val[all_features])])

# %%
# ==============================================================================
# 3. Landmark Filtering (Strictly Aligned with Code C)
# ==============================================================================
print("\n--- Step 3: Applying Landmark filtering ---")
df_dca = df_val.copy()
df_dca["model_risk"] = risk_probs

# Landmark Definition
df_dca["is_case"]    = (df_dca[TIME_COL] <= TIME_HORIZON) & (df_dca[EVENT_COL] == 1)
df_dca["is_control"] = (df_dca[TIME_COL] > TIME_HORIZON)

# Exclude uninformative censored cases
valid_mask = df_dca["is_case"] | df_dca["is_control"]
df_dca_filtered = df_dca[valid_mask].copy()
df_dca_filtered["outcome"] = df_dca_filtered["is_case"].astype(int)

true_prevalence = df_dca_filtered["outcome"].mean()
print(f"Original: {len(df_dca)}, After Landmark: {len(df_dca_filtered)}")
print(f"2-year event prevalence: {true_prevalence:.4f}")

# %%
# ==============================================================================
# 4. DCA Calculation & Smoothing
# ==============================================================================
print("\n--- Step 4: Running Decision Curve Analysis ---")
thresholds = np.arange(0.005, 0.065, 0.005) 

dca_results = dca(
    data=df_dca_filtered[["outcome", "model_risk"]],
    outcome="outcome",
    modelnames=["model_risk"],
    thresholds=thresholds
)

df_m = dca_results[dca_results["model"] == "model_risk"]
df_all = dca_results[dca_results["model"] == "all"]

pt = df_m["threshold"].values
nb_m = df_m["net_benefit"].values
nb_all = df_all["net_benefit"].values

# Smooth curves for visualization
smooth_frac = 0.5 
nb_m_s = lowess(nb_m, pt, frac=smooth_frac, it=0)[:, 1]
nb_all_s = lowess(nb_all, pt, frac=smooth_frac, it=0)[:, 1]

# %%
# ==============================================================================
# 5. Visualization
# ==============================================================================
plt.figure(figsize=(8, 8))

plt.plot(pt, nb_m_s,    lw=3,  color="#0072B2", label="CoxPH Model (Validation)")
plt.plot(pt, nb_all_s, lw=2,  color="#E69F00", label="Treat All")
plt.plot(pt, [0]*len(pt), lw=1.5, color="gray", linestyle=":", label="Treat None")

# Layout parameters matching Code C
plt.xlim(0.0, 0.06)
plt.ylim(-0.005, 0.02)
plt.tick_params(labelsize=15)
plt.xlabel("Threshold Probability", fontsize=20)
plt.ylabel("Net Benefit", fontsize=20)
plt.title("External Validation (JSTAR 2-Year)", fontsize=22)

# Prevalence Line
plt.axvline(true_prevalence, ls=":", lw=1.5, color="gray")
plt.text(true_prevalence + 0.001, 0.005, 
         f"Prevalence ≈ {true_prevalence:.3f}\n(Treat-all NB=0)", 
         ha="left", fontsize=12, color="gray")

# Zero-NB Calculation & Scatter
def calc_zero_nb(pt_arr, nb_arr):
    idx = np.where(np.sign(nb_arr[:-1]) != np.sign(nb_arr[1:]))[0]
    if len(idx) == 0: return None
    i = idx[-1]
    return pt_arr[i] - nb_arr[i] * (pt_arr[i+1] - pt_arr[i]) / (nb_arr[i+1] - nb_arr[i])

zero_nb = calc_zero_nb(pt, nb_m_s)
if zero_nb:
    plt.scatter([zero_nb], [0], s=40, color="#0072B2")
    plt.text(zero_nb + 0.001, 0.0015, f"Zero-NB ≈ {zero_nb:.2f}", fontsize=9, color="#0072B2")

# Legend and Borders
plt.legend(loc="upper right", fontsize=11, frameon=True, edgecolor="black")
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)

out_path = OUTPUT_DIR / "FigS3B_JSTAR_DCA_ValidationSet_2yr.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n✅ JSTAR DCA Figure saved to: {out_path}")


# ==================================
# MAIN EXECUTION
# ==================================
def main():
    """
    Run all validation analyses sequentially for JSTAR.
    """
    run_external_validation_metrics()
    run_risk_stratification_plots()
    print("All JSTAR validation analyses complete")


if __name__ == "__main__":
    main()