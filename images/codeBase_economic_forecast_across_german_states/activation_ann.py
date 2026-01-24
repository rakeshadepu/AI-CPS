#!/usr/bin/env python3
"""
ANN Activation Inference Script

Loads the trained TensorFlow ANN model (currentAiSolution.h5) and
reads activation_data.csv to produce GDP predictions.
Also prints the actual target value for easy comparison.

Authors     : Rohith Boggula, Rakesh Adepu
Course      : Advanced AI-based Application Systems - Business Information Systems
Institution : University of Potsdam
"""

# ======================================================
# IMPORTS
# ======================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ======================================================
# GLOBAL SETTINGS
# ======================================================

warnings.filterwarnings("ignore")

script_dir = Path(__file__).parent.resolve()


# ======================================================
# ANN INFERENCE
# ======================================================

def load_ann_model(model_path: Path):
    """
    Load the trained ANN model from disk.

    Args:
        model_path (Path): Path to the saved ANN model (.h5 file).

    Returns:
        tensorflow.keras.Model: Loaded ANN model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    ann_model = load_model(model_path)
    # print(f"Loaded ANN model from: {model_path}")
    
    return ann_model


def load_activation_data(activation_path: Path):
    """
    Load activation data from CSV file and apply training-consistent preprocessing.
    """

    if not activation_path.exists():
        raise FileNotFoundError(f"Activation data not found: {activation_path}")

    activation_df = pd.read_csv(activation_path)
    # print(f"\nLoaded activation data from: {activation_path}")
    print(f"Data shape: {activation_df.shape}")

    target_col = "GDP"

    # Separate target
    if target_col in activation_df.columns:
        y_actual = activation_df[target_col].values
        X_df = activation_df.drop(columns=[target_col])
        # print(f"Found target column '{target_col}' - will compare predictions with actual values")
    else:
        y_actual = None
        X_df = activation_df
        print(f"No target column '{target_col}' found - will only generate predictions")

    #  MATCH TRAINING FEATURES (CRITICAL)
    # ANN was trained on exactly 3 numeric features
    FEATURE_COLUMNS = ["population", "employment", "year"]


    # Keep only features used during training
    X_df = X_df[FEATURE_COLUMNS]

    # Convert to NumPy (same format as training)
    X_activation = X_df.astype(float).to_numpy()

    return X_activation, y_actual

def make_ann_prediction(
    ann_model,
    X_activation: np.ndarray,
    y_actual: np.ndarray = None
):
    """
    Make predictions using the ANN model.

    Args:
        ann_model: Trained ANN model
        X_activation (np.ndarray): Input features for prediction
        y_actual (np.ndarray, optional): Actual target values for comparison

    Returns:
        np.ndarray: Predictions
    """
    
    # Safety check: feature consistency
    assert X_activation.shape[1] == ann_model.input_shape[1], (
        f"Feature mismatch: model expects {ann_model.input_shape[1]}, "
        f"got {X_activation.shape[1]}"
    )

    # Make prediction
    y_pred_ann = ann_model.predict(X_activation, verbose=0).flatten()
    
    print("ANN PREDICTION RESULTS")
    print("-" * 60)
    
    print(f"\nPredicted GDP values:\n{y_pred_ann}")
    
    if y_actual is not None:
        print(f"\nActual GDP values:\n{y_actual}")
        
        # Calculate prediction metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_actual, y_pred_ann)
        mse = mean_squared_error(y_actual, y_pred_ann)
        rmse = np.sqrt(mse)
        
        print(f"\nPrediction Error Metrics:")
        print(f"  MAE  : {mae:.6f}")
        print(f"  MSE  : {mse:.6f}")
        print(f"  RMSE : {rmse:.6f}")
        
        # Only calculate R2 if we have more than 1 sample
        # if len(y_actual) > 1:
        #     r2 = r2_score(y_actual, y_pred_ann)
        #     print(f"  R²   : {r2:.6f}")
        # else:
        #     print(f"  R²   : N/A (Requires >1 sample)")
    
    return y_pred_ann


# ======================================================
# MAIN EXECUTION
# ======================================================

def main():
    """
    Main execution function for ANN inference.
    """
    print("ANN ACTIVATION INFERENCE")
    print("-" * 80)
    
    # Define paths
    from pathlib import Path

    model_path = Path("/tmp/knowledgeBase/currentAiSolution.keras")
    activation_path = Path("/tmp/activationBase/activation_data.csv")

    
    # print(f"\nPaths:")
    # print(f"  Script directory  : {script_dir}")
    # print(f"  Model path        : {model_path}")
    # print(f"  Activation path   : {activation_path}")
    
    # Step 1: Load the ANN model
    ann_model = load_ann_model(model_path)
    
    # Print model summary
    print("\nANN MODEL ARCHITECTURE")
    ann_model.summary()
    
    # Step 2: Load activation data
    X_activation, y_actual = load_activation_data(activation_path)
    
    print(f"\nActivation data features: {X_activation.shape[1]}")
    print(f"Number of samples: {X_activation.shape[0]}")
    
    # Step 3: Make predictions
    y_pred = make_ann_prediction(ann_model, X_activation, y_actual)
    
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("-" * 80 + "\n")
    
    return y_pred


# ======================================================
# MODULE TESTING
# ======================================================

if __name__ == "__main__":
    main()