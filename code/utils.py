"""
Utility Functions for AI-Based Economic Forecast
Across German States

Authors    : Rohith Boggula, Rakesh Adepu
Course     : Advanced AI-based Application Systems – Business Information Systems
Institution: University of Potsdam
"""

from pathlib import Path

import pandas as pd
import numpy as np


# ======================================================
# DIRECTORY MANAGEMENT
# ======================================================
def create_output_directories():
    """
    Create required output directories for models and visualizations.

    Returns:
        Path: Models directory
        Path: Learning base (visualizations) directory
    """

    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)

    learningbase_dir = Path("../learningBase")
    learningbase_dir.mkdir(parents=True, exist_ok=True)

    return models_dir, learningbase_dir


# ======================================================
# DATA LOADING
# ======================================================
def load_data():
    """
    Load preprocessed training and test datasets.

    Returns:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
    """

    train_df = pd.read_csv("../data/processed/training_data.csv")
    test_df = pd.read_csv("../data/processed/test_data.csv")

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples    : {len(test_df)}")

    feature_cols = ["population", "employment", "year"]
    target_col = "gdp"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print("\nData configuration:")
    print(f"  Features : {feature_cols}")
    print(f"  Target   : {target_col}")
    print(f"  X_train  : {X_train.shape}")
    print(f"  X_test   : {X_test.shape}")

    return X_train, y_train, X_test, y_test


# ======================================================
# MODEL COMPARISON
# ======================================================
def print_comparison(metrics_ann, metrics_ols):
    """
    Print a comparison table for ANN and OLS models.

    Args:
        metrics_ann (dict): Performance metrics of ANN model
        metrics_ols (dict): Performance metrics of OLS model
    """

    print("\nMODEL COMPARISON")
    print("-" * 60)

    comparison_df = pd.DataFrame(
        {
            "Model": ["ANN", "OLS"],
            "Test MSE": [
                metrics_ann["test"]["mse"],
                metrics_ols["test"]["mse"],
            ],
            "Test MAE": [
                metrics_ann["test"]["mae"],
                metrics_ols["test"]["mae"],
            ],
            "Test R²": [
                metrics_ann["test"]["r2"],
                metrics_ols["test"]["r2"],
            ],
        }
    )

    print(comparison_df.to_string(index=False))

    # Determine better-performing model
    ann_r2 = metrics_ann["test"]["r2"]
    ols_r2 = metrics_ols["test"]["r2"]

    if ols_r2 > ann_r2:
        diff = ols_r2 - ann_r2
        print(f"\nOLS outperforms ANN by {diff:.4f} R² points")
        print("→ Indicates predominantly linear relationships")
    else:
        diff = ann_r2 - ols_r2
        print(f"\nANN outperforms OLS by {diff:.4f} R² points")
        print("→ Indicates presence of non-linear patterns")


# ======================================================
# MODULE TESTING
# ======================================================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print("\nUtility module test completed successfully")