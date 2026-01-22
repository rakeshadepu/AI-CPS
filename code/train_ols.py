"""
OLS Model Training for GDP Prediction
AI-Based Economic Forecast Across German States

Authors    : Rohith Boggula, Rakesh Adepu
Course     : Advanced AI-based Application Systems - Business Information Systems
Institution: University of Potsdam
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ======================================================
# MODEL TRAINING
# ======================================================

def train_ols_model(X_train, y_train):
    """
    Train an Ordinary Least Squares (OLS) regression model.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target.

    Returns:
        tuple:
            - statsmodels.regression.linear_model.RegressionResults: Fitted OLS model.
            - pd.DataFrame or np.ndarray: Training features with constant term.
    """
    print("\nTRAINING OLS MODEL")

    # Add intercept term
    X_train_ols = sm.add_constant(X_train)

    # Fit OLS model
    ols_model = sm.OLS(y_train, X_train_ols).fit()

    print("\nOLS MODEL SUMMARY")
    print("-" * 60)
    print(ols_model.summary())

    return ols_model, X_train_ols


# ======================================================
# MODEL EVALUATION
# ======================================================

def evaluate_ols_model(ols_model, X_train, y_train, X_test, y_test):
    """
    Evaluate OLS model performance.

    Args:
        ols_model: Fitted OLS model.
        X_train, y_train: Training data.
        X_test, y_test: Test data.

    Returns:
        tuple:
            - dict: Performance metrics.
            - np.ndarray: Training predictions.
            - np.ndarray: Test predictions.
            - pd.DataFrame or np.ndarray: Test features with constant term.
    """
    print("\nEVALUATING OLS MODEL")

    # Add constant term
    X_train_ols = sm.add_constant(X_train)
    X_test_ols = sm.add_constant(X_test)

    # Predictions
    y_train_pred = ols_model.predict(X_train_ols)
    y_test_pred = ols_model.predict(X_test_ols)

    # Training metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Test metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nTraining Set Metrics:")
    print(f"  MSE : {train_mse:.6f}")
    print(f"  MAE : {train_mae:.6f}")
    print(f"  R²  : {train_r2:.6f}")

    print("\nTest Set Metrics:")
    print(f"  MSE : {test_mse:.6f}")
    print(f"  MAE : {test_mae:.6f}")
    print(f"  R²  : {test_r2:.6f}")

    metrics = {
        "train": {"mse": train_mse, "mae": train_mae, "r2": train_r2},
        "test": {"mse": test_mse, "mae": test_mae, "r2": test_r2},
    }

    return metrics, y_train_pred, y_test_pred, X_test_ols


# ======================================================
# MODEL SAVING
# ======================================================

def save_ols_model(ols_model, ols_dir):
    """
    Save the trained OLS model and its summary.

    Args:
        ols_model: Fitted OLS model.
        ols_dir (Path): Directory to save the model.
    """
    print("\nSAVING OLS MODEL")
    ols_dir = Path(ols_dir)

    # Save model (pickle)
    model_path = ols_dir / "currentOlsSolution.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(ols_model, file)
    print(f"  Model saved to: {model_path}")

    # Save summary
    summary_path = ols_dir / "currentOlsSolution_summary.txt"
    with open(summary_path, "w") as file:
        file.write(ols_model.summary().as_text())
    print(f"  Summary saved to: {summary_path}")


# ======================================================
# COMPLETE TRAINING PIPELINE
# ======================================================

def train_and_evaluate_ols(X_train, y_train, X_test, y_test, ols_dir):
    """
    Complete pipeline for OLS model training, evaluation, and visualization.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        models_dir: Directory to save the model.

    Returns:
        tuple:
            - ols_model: Trained OLS model.
            - metrics: Dictionary of performance metrics.
            - predictions: Dictionary containing all predictions.
    """
    print("\nOLS MODEL TRAINING PIPELINE")

    # 1. Train the model
    ols_model, X_train_ols = train_ols_model(X_train, y_train)

    # 2. Evaluate the model
    metrics, y_train_pred, y_test_pred, X_test_ols = evaluate_ols_model(
        ols_model, X_train, y_train, X_test, y_test
    )

    # 3. Save the model
    save_ols_model(ols_model, ols_dir)

    # 4. Visualizations
    try:
        from visualization_ols import create_all_visualizations

        create_all_visualizations(
            ols_model=ols_model,
            metrics=metrics,
            y_train=y_train,
            y_train_pred=y_train_pred,
            y_test=y_test,
            y_test_pred=y_test_pred,
            X_train_ols=X_train_ols,
            output_dir=ols_dir,
        )
    except ImportError:
        print("\nWARNING: visualization_ols module not found.")
        print("Visualizations will be skipped.")
        print("Please ensure visualization_ols.py is in the same directory.")

    # Package predictions
    predictions = {
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "X_train_ols": X_train_ols,
        "X_test_ols": X_test_ols,
    }

    return ols_model, metrics, predictions


# ======================================================
# EXAMPLE USAGE
# ======================================================

if __name__ == "__main__":
    """
    Example usage of the complete OLS training pipeline.
    Demonstrates the end-to-end workflow for OLS model training.
    """
    from utils import load_data, create_output_directories
    from visualization_ols import create_all_visualizations

    # Create and get all directories ONCE
    documentation_dir, ann_dir, ols_dir = create_output_directories()

    X_train, y_train, X_test, y_test = load_data()

    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Train and evaluate OLS model
    ols_model, metrics, predictions = train_and_evaluate_ols(
        X_train, y_train, X_test, y_test, ols_dir=ols_dir
    )

    create_all_visualizations(
        ols_model=ols_model,
        metrics=metrics,
        y_train=y_train,
        y_train_pred=predictions["y_train_pred"],
        y_test=y_test,
        y_test_pred=predictions["y_test_pred"],
        X_train_ols=predictions["X_train_ols"],
        output_dir=ols_dir,
    )

    save_ols_model(ols_model, ols_dir=ols_dir) 

    # Display final results
    print("\nTraining Performance:")
    print(f"  MSE: {metrics['train']['mse']:.6f}")
    print(f"  MAE: {metrics['train']['mae']:.6f}")
    print(f"  R²:  {metrics['train']['r2']:.6f}")

    print("\nTest Performance:")
    print(f"  MSE: {metrics['test']['mse']:.6f}")
    print(f"  MAE: {metrics['test']['mae']:.6f}")
    print(f"  R²:  {metrics['test']['r2']:.6f}")
