"""
ANN Model Training for GDP Prediction with Complete Visualizations
AI-Based Economic Forecast Across German States

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
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ======================================================
# GLOBAL SETTINGS
# ======================================================

warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)


# ======================================================
# MODEL CREATION
# ======================================================

def create_ann_model(input_dim: int) -> tf.keras.Model:
    """
    Create and compile the ANN architecture.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tensorflow.keras.Model: Compiled ANN model.
    """

    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,),kernel_regularizer=l2(0.001)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1)  # Linear output for regression
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    print("\nANN MODEL ARCHITECTURE")
    model.summary()

    return model


# ======================================================
# MODEL TRAINING
# ======================================================

def train_ann_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 500,
    batch_size: int = 16,
):
    """
    Train the ANN model.

    Args:
        model: Compiled Keras model.
        X_train, y_train: Training data.
        X_test, y_test: Validation data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.

    Returns:
        tensorflow.keras.callbacks.History: Training history.
    """

    print("\n" + "=" * 60)
    print("TRAINING ANN MODEL")
    print("=" * 60)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1,
    )

    print("\nTraining completed")
    print(f"  Epochs                : {len(history.history['loss'])}")
    print(f"  Final training loss   : {history.history['loss'][-1]:.6f}")
    print(f"  Final validation loss : {history.history['val_loss'][-1]:.6f}")

    return history


# ======================================================
# MODEL EVALUATION
# ======================================================

def evaluate_ann_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Evaluate ANN model performance.

    Args:
        model: Trained Keras model.
        X_train, y_train: Training data.
        X_test, y_test: Test data.

    Returns:
        dict: Performance metrics.
        np.ndarray: Training predictions.
        np.ndarray: Test predictions.
    """

    print("\nEVALUATING ANN MODEL")

    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_test_pred = model.predict(X_test, verbose=0).flatten()

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    print("\nTraining Set Metrics:")
    print(f"  MSE : {train_mse:.6f}")
    print(f"  MAE : {train_mae:.6f}")
    print(f"  R²  : {train_r2:.6f}")

    print("\nTest Set Metrics:")
    print(f"  MSE  : {test_mse:.6f}")
    print(f"  MAE  : {test_mae:.6f}")
    print(f"  R²   : {test_r2:.6f}")
    print(f"  RMSE : {test_rmse:.6f}")

    metrics = {
        "train": {
            "mse": train_mse,
            "mae": train_mae,
            "r2": train_r2,
        },
        "test": {
            "mse": test_mse,
            "mae": test_mae,
            "r2": test_r2,
            "rmse": test_rmse,
        },
    }

    return metrics, y_train_pred, y_test_pred


# ======================================================
# MODEL SAVING
# ======================================================

def save_ann_model(model: tf.keras.Model, models_dir: Path) -> None:
    """
    Save the trained ANN model to disk.

    Args:
        model: Trained Keras model.
        models_dir (Path): Directory to save the model.
    """

    print("\nSAVING ANN MODEL")

    model_path = models_dir / "currentAiSolution.h5"
    model.save(model_path)

    print(f"  Model saved to: {model_path}\n")


# ======================================================
# COMPLETE TRAINING PIPELINE
# ======================================================

def complete_training_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 500,
    batch_size: int = 16,
    learning_base_dir: str = "learningBase",
):
    """
    Complete ANN training pipeline including evaluation and visualization.

    Returns:
        model   : Trained ANN model
        history : Training history
        metrics : Performance metrics
    """
    from utils import create_output_directories

    documentation_dir, ann_dir, ols_dir = create_output_directories()

    print("\nCOMPLETE ANN TRAINING PIPELINE")
    print("=" * 80)

    models_dir = ann_dir

    print(f"\nOutput directory: {models_dir.absolute()}")

    # Step 1: Model creation
    model = create_ann_model(input_dim=X_train.shape[1])

    # Step 2: Model training
    history = train_ann_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Step 3: Model evaluation
    metrics, _, y_test_pred = evaluate_ann_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Step 4: Save model
    save_ann_model(model, models_dir)

    # Step 5: Generate visualizations
    generate_all_visualizations(
        history=history,
        y_test=y_test,
        y_test_pred=y_test_pred,
        metrics=metrics,
        output_dir=models_dir,
    )

    print("\nTRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print(f"All outputs saved to: {models_dir.absolute()}")

    return model, history, metrics


# ======================================================
# MODULE TESTING
# ======================================================

if __name__ == "__main__":

    # Custom visualization utilities
    from visualization_ann import generate_all_visualizations
    from utils import load_data, create_output_directories

    documentation_dir, ann_dir, ols_dir = create_output_directories()

    print("\nTESTING ANN TRAINING MODULE")
    print("=" * 80)

    # Load dataset
    X_train, y_train, X_test, y_test = load_data()

    print("\nData loaded successfully:")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Features         : {X_train.shape[1]}")

    # Execute training pipeline
    model, history, metrics = complete_training_pipeline(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=500,
        batch_size=16,
        learning_base_dir=ann_dir,
    )

    print("\nMODULE TEST COMPLETED SUCCESSFULLY")
