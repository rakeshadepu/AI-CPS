#!/usr/bin/env python3
"""
OLS Activation Inference Script

Loads the trained OLS model (currentOlsSolution.pkl) and
reads a single-entry activation_data.csv to produce a prediction.
Also prints the actual target value for easy comparison.

Authors     : Rohith Boggula, Rakesh Adepu
Course      : Advanced AI-based Application Systems - Business Information Systems
Institution : University of Potsdam
"""

# ======================================================
# IMPORTS
# ======================================================

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ======================================================
# GLOBAL SETTINGS
# ======================================================

script_dir = Path(__file__).parent.resolve()


# ======================================================
# OLS INFERENCE
# ======================================================

def load_ols_model(model_path: Path):
    """
    Load the trained OLS model from disk.

    Args:
        model_path (Path): Path to the saved OLS model (.pkl file).

    Returns:
        Trained OLS model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        ols_model = pickle.load(f)
    
    # print(f"Loaded OLS model from: {model_path}")
    return ols_model


def load_activation_data(activation_path: Path):
    """
    Load activation data from CSV file.

    Args:
        activation_path (Path): Path to activation_data.csv

    Returns:
        pd.DataFrame: Activation data
        np.ndarray or None: Actual target values if present
    """
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation data not found: {activation_path}")

    activation_df = pd.read_csv(activation_path)
    # print(f"\nLoaded activation data from: {activation_path}")
    print(f"Data shape: {activation_df.shape}")
    
    # Check for target column (GDP)
    target_col = "GDP"
    if target_col in activation_df.columns:
        X_activation = activation_df.drop(columns=[target_col])
        y_actual = activation_df[target_col].values
        # print(f"Found target column '{target_col}' - will compare predictions with actual values")
    else:
        X_activation = activation_df
        y_actual = None
        print(f"No target column '{target_col}' found - will only generate predictions")
    
    return X_activation, y_actual


def make_ols_prediction(
    ols_model,
    X_activation: pd.DataFrame,
    y_actual: np.ndarray = None
):
    """
    Make predictions using the OLS model.

    Args:
        ols_model: Trained OLS model
        X_activation (pd.DataFrame): Input features for prediction
        y_actual (np.ndarray, optional): Actual target values for comparison

    Returns:
        np.ndarray: Predictions
    """
    # Add constant to match OLS training (constant is added during training)

    # Features used during training
    trained_features = ols_model.model.exog_names
    trained_features_no_const = [f for f in trained_features if f != "const"]

    # Drop unused columns like 'state'
    X_activation_aligned = X_activation[trained_features_no_const]

    # Add constant exactly once
    X_activation_ols = sm.add_constant(
        X_activation_aligned,
        prepend=True,
        has_constant="add"
    )
    """ The code below is commented out as it is intended only for debugging purposes and is not required at the moment; however, it can be used if any issues arise in the future."""
    # print("\nDEBUG INFORMATION")
    # print("-" * 40)
    # print("Model expects columns:")
    # print(ols_model.model.exog_names)

    # print("\nActivation columns BEFORE constant:")
    # print(X_activation.columns.tolist())

    # print("\nActivation columns AFTER constant:")
    # print(X_activation_ols.columns.tolist())

    # print("\nShapes:")
    # print("X_activation_ols:", X_activation_ols.shape)
    # print("Model params     :", ols_model.params.shape)
    # print("-" * 40)


    # Make prediction
    y_pred_ols = ols_model.predict(X_activation_ols)
    
    print("OLS PREDICTION RESULTS")
    print("-" * 60)
    
    print(f"\nPredicted GDP values:\n{y_pred_ols.values}")
    
    if y_actual is not None:
        print(f"\nActual GDP values:\n{y_actual}")
        
        # Calculate simple metrics
        error = y_actual - y_pred_ols.values
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))
        
        print(f"\nPrediction Error Metrics:")
        print(f"  MAE  : {mae:.6f}")
        print(f"  RMSE : {rmse:.6f}")
    
    return y_pred_ols


# ======================================================
# MAIN EXECUTION
# ======================================================

def main():
    """
    Main execution function for OLS inference.
    """
    print("OLS ACTIVATION INFERENCE")
    print("-" * 80)
    
    # Define paths
    model_path = script_dir.parent / "images" / "knowledgeBase_economic_forecast_across_german_states" / "currentOlsSolution.pkl"
    activation_path = script_dir.parent / "images"  / "activationBase_economic_forecast_across_german_states" / "activation_data.csv"
    
    # print(f"\nPaths:")
    # print(f"  Script directory  : {script_dir}")
    # print(f"  Model path        : {model_path}")
    # print(f"  Activation path   : {activation_path}")
    
    # Step 1: Load the OLS model
    ols_model = load_ols_model(model_path)
    
    # Step 2: Load activation data
    X_activation, y_actual = load_activation_data(activation_path)
    
    # Step 3: Make predictions
    y_pred = make_ols_prediction(ols_model, X_activation, y_actual)
    
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("-" * 80 + "\n")
    
    return y_pred


# ======================================================
# MODULE TESTING
# ======================================================

if __name__ == "__main__":
    main()