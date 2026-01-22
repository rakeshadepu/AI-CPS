"""
OLS Model Visualization and Testing
AI-Based Economic Forecast Across German States

Authors    : Rohith Boggula, Rakesh Adepu
Course     : Advanced AI-based Application Systems – Business Information Systems
Institution: University of Potsdam
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ------------------------------------------------------
# DIAGNOSTIC PLOTS
# ------------------------------------------------------
def plot_ols_diagnostics(ols_model, y_train, y_train_pred, X_train_ols, output_dir):
    """
    Create comprehensive diagnostic plots for an OLS model.

    Args:
        ols_model: Fitted OLS model.
        y_train: Actual training values.
        y_train_pred: Predicted training values.
        X_train_ols: Training features with constant term.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate residuals
    residuals = y_train - y_train_pred
    standardized_residuals = residuals / np.std(residuals)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("OLS Regression Diagnostics", fontsize=16, fontweight="bold")

    # 1. Residuals vs Fitted Values
    axes[0, 0].scatter(y_train_pred, residuals, alpha=0.6, edgecolors="k", s=50)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Fitted Values", fontsize=12)
    axes[0, 0].set_ylabel("Residuals", fontsize=12)
    axes[0, 0].set_title("Residuals vs Fitted", fontsize=14, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    # Add smoothed line
    sorted_indices = np.argsort(y_train_pred)
    z = np.polyfit(y_train_pred[sorted_indices], residuals[sorted_indices], 3)
    p = np.poly1d(z)
    axes[0, 0].plot(
        y_train_pred[sorted_indices],
        p(y_train_pred[sorted_indices]),
        "b-",
        linewidth=2,
        label="Trend",
    )
    axes[0, 0].legend()

    # 2. Q-Q Plot (Normal Probability Plot)
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q Plot", fontsize=14, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Scale-Location Plot (Spread-Location)
    sqrt_abs_std_residuals = np.sqrt(np.abs(standardized_residuals))
    axes[1, 0].scatter(
        y_train_pred,
        sqrt_abs_std_residuals,
        alpha=0.6,
        edgecolors="k",
        s=50,
    )
    axes[1, 0].set_xlabel("Fitted Values", fontsize=12)
    axes[1, 0].set_ylabel("√|Standardized Residuals|", fontsize=12)
    axes[1, 0].set_title("Scale-Location Plot", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # Add smoothed line
    z = np.polyfit(
        y_train_pred[sorted_indices],
        sqrt_abs_std_residuals[sorted_indices],
        3,
    )
    p = np.poly1d(z)
    axes[1, 0].plot(
        y_train_pred[sorted_indices],
        p(y_train_pred[sorted_indices]),
        "r-",
        linewidth=2,
        label="Trend",
    )
    axes[1, 0].legend()

    # 4. Residuals vs Leverage
    influence = ols_model.get_influence()
    leverage = influence.hat_matrix_diag

    axes[1, 1].scatter(
        leverage,
        standardized_residuals,
        alpha=0.6,
        edgecolors="k",
        s=50,
    )
    axes[1, 1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel("Leverage", fontsize=12)
    axes[1, 1].set_ylabel("Standardized Residuals", fontsize=12)
    axes[1, 1].set_title("Residuals vs Leverage", fontsize=14, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    # Add Cook's distance contours
    x_range = np.linspace(0.001, max(leverage), 100)
    for d in [0.5, 1.0]:
        y_pos = np.sqrt(d * len(residuals) * (1 - x_range) / x_range)
        y_neg = -y_pos
        axes[1, 1].plot(x_range, y_pos, "r--", alpha=0.5, linewidth=1)
        axes[1, 1].plot(x_range, y_neg, "r--", alpha=0.5, linewidth=1)
        axes[1, 1].text(
            max(leverage) * 0.9,
            y_pos[-1],
            f"Cook's d={d}",
            fontsize=9,
            color="red",
        )

    plt.tight_layout()

    # Save figure
    diagnostic_path = output_dir / "ols_diagnostic_plots.png"
    plt.savefig(diagnostic_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------
# SCATTER PLOTS (ACTUAL VS PREDICTED)
# ------------------------------------------------------
def plot_predictions_scatter(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    train_r2,
    test_r2,
    output_dir,
):
    """
    Create scatter plots comparing actual vs predicted values.

    Args:
        y_train: Actual training values.
        y_train_pred: Predicted training values.
        y_test: Actual test values.
        y_test_pred: Predicted test values.
        train_r2: R² score for training set.
        test_r2: R² score for test set.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "OLS Model: Actual vs Predicted Values",
        fontsize=16,
        fontweight="bold",
    )

    # Training Set Scatter Plot
    axes[0].scatter(
        y_train,
        y_train_pred,
        alpha=0.6,
        edgecolors="k",
        s=80,
        label="Data Points",
    )

    # Perfect prediction line
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    axes[0].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )

    # Add regression line
    z = np.polyfit(y_train, y_train_pred, 1)
    p = np.poly1d(z)
    axes[0].plot(
        np.sort(y_train),
        p(np.sort(y_train)),
        "b-",
        linewidth=2,
        label="Fitted Line",
    )

    axes[0].set_xlabel("Actual GDP", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Predicted GDP", fontsize=12, fontweight="bold")
    axes[0].set_title(
        f"Training Set (R² = {train_r2:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Test Set Scatter Plot
    axes[1].scatter(
        y_test,
        y_test_pred,
        alpha=0.6,
        edgecolors="k",
        s=80,
        color="orange",
        label="Data Points",
    )

    # Perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    axes[1].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )

    # Add regression line
    z = np.polyfit(y_test, y_test_pred, 1)
    p = np.poly1d(z)
    axes[1].plot(
        np.sort(y_test),
        p(np.sort(y_test)),
        "b-",
        linewidth=2,
        label="Fitted Line",
    )

    axes[1].set_xlabel("Actual GDP", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Predicted GDP", fontsize=12, fontweight="bold")
    axes[1].set_title(
        f"Test Set (R² = {test_r2:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    scatter_path = output_dir / "ols_predictions_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------
# RESIDUAL DISTRIBUTION PLOT
# ------------------------------------------------------
def plot_residual_distribution(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    output_dir,
):
    """
    Create residual distribution plots for training and test sets.

    Args:
        y_train: Actual training values.
        y_train_pred: Predicted training values.
        y_test: Actual test values.
        y_test_pred: Predicted test values.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Residual Distribution Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Training residuals histogram
    axes[0].hist(
        train_residuals,
        bins=30,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        density=True,
    )

    # Fit normal distribution
    mu, std = stats.norm.fit(train_residuals)
    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[0].plot(x, p, "r-", linewidth=2, label="Normal Fit")

    axes[0].axvline(
        x=0,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Zero Line",
    )
    axes[0].set_xlabel("Residuals", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Density", fontsize=12, fontweight="bold")
    axes[0].set_title(
        f"Training Set (μ={mu:.2f}, σ={std:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Test residuals histogram
    axes[1].hist(
        test_residuals,
        bins=30,
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
        density=True,
    )

    # Fit normal distribution
    mu, std = stats.norm.fit(test_residuals)
    xmin, xmax = axes[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[1].plot(x, p, "r-", linewidth=2, label="Normal Fit")

    axes[1].axvline(
        x=0,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Zero Line",
    )
    axes[1].set_xlabel("Residuals", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Density", fontsize=12, fontweight="bold")
    axes[1].set_title(
        f"Test Set (μ={mu:.2f}, σ={std:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    residual_path = output_dir / "ols_residual_distribution.png"
    plt.savefig(residual_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------
# PERFORMANCE METRICS COMPARISON
# ------------------------------------------------------
def save_performance_metrics(metrics, output_dir):
    """
    Save performance metrics to CSV and create visualization.

    Args:
        metrics: Dictionary containing performance metrics.
        output_dir: Directory to save files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    metrics_df = pd.DataFrame(
        {
            "Dataset": ["Training", "Test"],
            "MSE": [metrics["train"]["mse"], metrics["test"]["mse"]],
            "MAE": [metrics["train"]["mae"], metrics["test"]["mae"]],
            "R²": [metrics["train"]["r2"], metrics["test"]["r2"]],
        }
    )

    # Save to CSV
    metrics_path = output_dir / "ols_performance_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "OLS Model Performance Metrics",
        fontsize=16,
        fontweight="bold",
    )

    metrics_names = ["MSE", "MAE", "R²"]
    colors = ["skyblue", "lightcoral"]

    for idx, metric in enumerate(metrics_names):
        values = metrics_df[metric].values
        bars = axes[idx].bar(
            ["Training", "Test"],
            values,
            color=colors,
            edgecolor="black",
            linewidth=2,
        )
        axes[idx].set_ylabel(metric, fontsize=12, fontweight="bold")
        axes[idx].set_title(
            f"{metric} Comparison",
            fontsize=14,
            fontweight="bold",
        )
        axes[idx].grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()

    # Save figure
    metrics_viz_path = output_dir / "ols_performance_metrics.png"
    plt.savefig(metrics_viz_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------
# MAIN VISUALIZATION FUNCTION
# ------------------------------------------------------
def create_all_visualizations(
    ols_model,
    metrics,
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    X_train_ols,
    output_dir="learningbase",
):
    """
    Create all visualizations for OLS model evaluation.

    Args:
        ols_model: Fitted OLS model.
        metrics: Dictionary containing performance metrics.
        y_train: Actual training values.
        y_train_pred: Predicted training values.
        y_test: Actual test values.
        y_test_pred: Predicted test values.
        X_train_ols: Training features with constant term.
        output_dir: Directory to save all plots.
    """
    # 1. Diagnostic Plots
    plot_ols_diagnostics(
        ols_model,
        y_train,
        y_train_pred,
        X_train_ols,
        output_dir,
    )

    # 2. Scatter Plots
    plot_predictions_scatter(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        metrics["train"]["r2"],
        metrics["test"]["r2"],
        output_dir,
    )

    # 3. Residual Distribution
    plot_residual_distribution(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        output_dir,
    )

    # 4. Performance Metrics
    save_performance_metrics(metrics, output_dir)


# ------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------
if __name__ == "__main__":
    """
    Example usage of visualization functions.
    This assumes you have already trained your OLS model.
    """
