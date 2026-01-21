"""
Complete Visualization Module for ANN Model Training
AI-Based Economic Forecast Across German States

This module creates all required visualizations:
1. Training and validation curves
2. Diagnostic plots (4-panel)
3. Scatter plots (Actual vs Predicted)

Authors: Rakesh Adepu, Rohith Boggula
Course: Advanced AI-based Application Systems - Business Information Systems
Institution: University of Potsdam
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path


def create_training_validation_curves(history, output_dir):
    """
    VISUALIZATION 1: Training and Validation Curves
    
    Creates a 2-panel plot showing:
    - Left panel: Training vs Validation Loss (MSE)
    - Right panel: Training vs Validation MAE
    
    Args:
        history: Keras training history object
        output_dir: Directory to save the plot (Path object)
    """

    print("CREATING TRAINING AND VALIDATION CURVES")

    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history.history["loss"]) + 1)
    
    # ===== Panel 1: Loss Curves =====
    axes[0].plot(epochs, history.history["loss"], 
                 label="Training Loss", linewidth=2.5, 
                 color='#2E86AB', marker='o', markersize=3, alpha=0.8)
    axes[0].plot(epochs, history.history["val_loss"], 
                 label="Validation Loss", linewidth=2.5, 
                 color='#A23B72', marker='s', markersize=3, alpha=0.8)
    
    axes[0].set_xlabel("Epoch", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Loss (MSE)", fontsize=13, fontweight='bold')
    axes[0].set_title("Training vs Validation Loss", 
                      fontsize=15, fontweight='bold', pad=15)
    axes[0].legend(fontsize=11, loc='best', frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].tick_params(labelsize=10)
    
    # Add final values annotation
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    axes[0].text(0.98, 0.98, 
                 f'Final Training Loss: {final_train_loss:.6f}\nFinal Validation Loss: {final_val_loss:.6f}',
                 transform=axes[0].transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ===== Panel 2: MAE Curves =====
    axes[1].plot(epochs, history.history["mae"], 
                 label="Training MAE", linewidth=2.5, 
                 color='#2E86AB', marker='o', markersize=3, alpha=0.8)
    axes[1].plot(epochs, history.history["val_mae"], 
                 label="Validation MAE", linewidth=2.5, 
                 color='#A23B72', marker='s', markersize=3, alpha=0.8)
    
    axes[1].set_xlabel("Epoch", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("MAE", fontsize=13, fontweight='bold')
    axes[1].set_title("Training vs Validation MAE", 
                      fontsize=15, fontweight='bold', pad=15)
    axes[1].legend(fontsize=11, loc='best', frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].tick_params(labelsize=10)
    
    # Add final values annotation
    final_train_mae = history.history["mae"][-1]
    final_val_mae = history.history["val_mae"][-1]
    axes[1].text(0.98, 0.98, 
                 f'Final Training MAE: {final_train_mae:.6f}\nFinal Validation MAE: {final_val_mae:.6f}',
                 transform=axes[1].transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    save_path = output_dir / "1_training_validation_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Saved: {save_path}")
    print(f"  Total Epochs: {len(epochs)}")
    print(f"  Final Training Loss: {final_train_loss:.6f}")
    print(f"  Final Validation Loss: {final_val_loss:.6f}")
    print("\n")


def create_diagnostic_plots(y_actual, y_predicted, output_dir, model_name="ANN"):
    """
    VISUALIZATION 2: Diagnostic Plots (4-Panel)
    
    Creates a 2x2 grid of diagnostic plots:
    - Top-left: Residuals vs Predicted Values
    - Top-right: Histogram of Residuals
    - Bottom-left: Normal Q-Q Plot
    - Bottom-right: Scale-Location Plot
    
    Args:
        y_actual: Actual test values (numpy array)
        y_predicted: Predicted test values (numpy array)
        output_dir: Directory to save the plot (Path object)
        model_name: Name of the model (for title)
    """

    print(f"CREATING DIAGNOSTIC PLOTS - {model_name}")

    
    residuals = y_actual - y_predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{model_name} Model: Diagnostic Plots', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # ===== Plot 1: Residuals vs Predicted =====
    axes[0, 0].scatter(y_predicted, residuals, 
                       alpha=0.6, s=50, 
                       edgecolors='black', linewidths=0.5, 
                       color='steelblue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2.5, label='Zero Line')
    
    # Add lowess smoothing line
    from scipy.signal import savgol_filter
    sorted_idx = np.argsort(y_predicted)
    try:
        smoothed = savgol_filter(residuals[sorted_idx], 
                                window_length=min(51, len(residuals)//3*2+1), 
                                polyorder=3)
        axes[0, 0].plot(y_predicted[sorted_idx], smoothed, 
                       color='orange', linewidth=2, label='Trend')
    except:
        pass
    
    axes[0, 0].set_xlabel('Predicted GDP (Normalized)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Residuals', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # ===== Plot 2: Histogram of Residuals =====
    n, bins, patches = axes[0, 1].hist(residuals, bins=30, 
                                        edgecolor='black', 
                                        alpha=0.7, 
                                        color='skyblue',
                                        density=True)
    
    # Overlay normal distribution curve
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 
                    'r-', linewidth=2.5, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
    axes[0, 1].axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero')
    
    axes[0, 1].set_xlabel('Residuals', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # ===== Plot 3: Normal Q-Q Plot =====
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q Plot', fontsize=14, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Enhance Q-Q plot appearance
    line = axes[1, 0].get_lines()[0]
    line.set_markerfacecolor('steelblue')
    line.set_markeredgecolor('black')
    line.set_markersize(6)
    line.set_alpha(0.6)
    
    # ===== Plot 4: Scale-Location Plot =====
    standardized_residuals = residuals / np.std(residuals)
    sqrt_abs_std_residuals = np.sqrt(np.abs(standardized_residuals))
    
    axes[1, 1].scatter(y_predicted, sqrt_abs_std_residuals, 
                       alpha=0.6, s=50, 
                       edgecolors='black', linewidths=0.5, 
                       color='coral')
    
    # Add trend line
    sorted_idx = np.argsort(y_predicted)
    try:
        smoothed = savgol_filter(sqrt_abs_std_residuals[sorted_idx], 
                                window_length=min(51, len(residuals)//3*2+1), 
                                polyorder=3)
        axes[1, 1].plot(y_predicted[sorted_idx], smoothed, 
                       color='blue', linewidth=2, label='Trend')
    except:
        pass
    
    axes[1, 1].set_xlabel('Predicted GDP (Normalized)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('√|Standardized Residuals|', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Scale-Location Plot', fontsize=14, fontweight='bold', pad=10)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = output_dir / f"2_diagnostic_plots_{model_name.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Saved: {save_path}")
    print(f"  Residual Statistics:")
    print(f"    Mean: {residuals.mean():.6f}")
    print(f"    Std:  {residuals.std():.6f}")
    print(f"    Min:  {residuals.min():.6f}")
    print(f"    Max:  {residuals.max():.6f}")
    print("\n")


def create_scatter_plot(y_actual, y_predicted, metrics, output_dir, model_name="ANN"):
    """
    VISUALIZATION 3: Scatter Plot (Actual vs Predicted)
    
    Creates a scatter plot comparing actual vs predicted values
    with perfect prediction line and performance metrics
    
    Args:
        y_actual: Actual test values (numpy array)
        y_predicted: Predicted test values (numpy array)
        metrics: Dictionary containing performance metrics
        output_dir: Directory to save the plot (Path object)
        model_name: Name of the model (for title)
    """

    print(f"CREATING SCATTER PLOT - {model_name}")

    
    fig, ax = plt.subplots(figsize=(11, 11))
    
    # Scatter plot
    ax.scatter(y_actual, y_predicted, 
               alpha=0.6, s=80, 
               edgecolors='black', linewidths=0.7, 
               color='steelblue', 
               label='Predictions')
    
    # Perfect prediction line (45-degree line)
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    margin = (max_val - min_val) * 0.05
    
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'r--', linewidth=3, label='Perfect Prediction', alpha=0.8)
    
    ax.set_xlabel('Actual GDP (Normalized)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted GDP (Normalized)', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} Model: Actual vs Predicted GDP', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    # Set equal aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    # Add performance metrics text box
    r2 = metrics['test']['r2']
    mse = metrics['test']['mse']
    mae = metrics['test']['mae']
    rmse = metrics['test']['rmse']
    
    metrics_text = f"Performance Metrics:\n"
    metrics_text += f"{'='*30}\n"
    metrics_text += f"R² Score    = {r2:.6f}\n"
    metrics_text += f"MSE         = {mse:.6f}\n"
    metrics_text += f"RMSE        = {rmse:.6f}\n"
    metrics_text += f"MAE         = {mae:.6f}\n"
    metrics_text += f"{'='*30}\n"
    metrics_text += f"N samples   = {len(y_actual)}"
    
    ax.text(0.97, 0.03, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8),
            family='monospace')
    
    plt.tight_layout()
    save_path = output_dir / f"3_scatter_plot_{model_name.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Saved: {save_path}")
    print(f"  R² Score: {r2:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print("\n")


def save_training_summary(history, metrics, output_dir):
    """
    Save comprehensive training summary to text file
    
    Args:
        history: Keras training history object
        metrics: Dictionary containing performance metrics
        output_dir: Directory to save the file (Path object)
    """

    print("SAVING TRAINING SUMMARY")

    
    summary_path = output_dir / "training_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AI-BASED ECONOMIC FORECAST ACROSS GERMAN STATES\n")
        f.write("ANN Model Training Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Training Information
        f.write("TRAINING INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Training Iterations (Epochs): {len(history.history['loss'])}\n")
        f.write(f"Final Training Loss (MSE):          {history.history['loss'][-1]:.8f}\n")
        f.write(f"Final Validation Loss (MSE):        {history.history['val_loss'][-1]:.8f}\n")
        f.write(f"Final Training MAE:                 {history.history['mae'][-1]:.8f}\n")
        f.write(f"Final Validation MAE:               {history.history['val_mae'][-1]:.8f}\n\n")
        
        # Best Epoch Information
        best_epoch = np.argmin(history.history['val_loss']) + 1
        f.write(f"Best Validation Loss at Epoch:      {best_epoch}\n")
        f.write(f"Best Validation Loss Value:         {min(history.history['val_loss']):.8f}\n\n")
        
        # Training Set Performance
        f.write("=" * 80 + "\n")
        f.write("TRAINING SET PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Squared Error (MSE):           {metrics['train']['mse']:.8f}\n")
        f.write(f"Mean Absolute Error (MAE):          {metrics['train']['mae']:.8f}\n")
        f.write(f"R² Score:                           {metrics['train']['r2']:.8f}\n\n")
        
        # Test Set Performance
        f.write("=" * 80 + "\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Squared Error (MSE):           {metrics['test']['mse']:.8f}\n")
        f.write(f"Root Mean Squared Error (RMSE):     {metrics['test']['rmse']:.8f}\n")
        f.write(f"Mean Absolute Error (MAE):          {metrics['test']['mae']:.8f}\n")
        f.write(f"R² Score:                           {metrics['test']['r2']:.8f}\n\n")
        
        # Model Assessment
        f.write("=" * 80 + "\n")
        f.write("MODEL ASSESSMENT\n")
        f.write("-" * 80 + "\n")
        
        # Check for overfitting/underfitting
        train_r2 = metrics['train']['r2']
        test_r2 = metrics['test']['r2']
        r2_diff = train_r2 - test_r2
        
        if r2_diff > 0.1:
            f.write("Status: OVERFITTING DETECTED\n")
            f.write(f"  Training R² ({train_r2:.4f}) significantly exceeds Test R² ({test_r2:.4f})\n")
            f.write(f"  Difference: {r2_diff:.4f}\n")
        elif test_r2 > 0.8:
            f.write("Status: EXCELLENT FIT\n")
            f.write(f"  Test R² of {test_r2:.4f} indicates strong predictive performance\n")
        elif test_r2 > 0.6:
            f.write("Status: GOOD FIT\n")
            f.write(f"  Test R² of {test_r2:.4f} indicates reasonable predictive performance\n")
        else:
            f.write("Status: MODERATE FIT\n")
            f.write(f"  Test R² of {test_r2:.4f} suggests room for improvement\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f" Saved: {summary_path}")
    print(f"  Total Epochs: {len(history.history['loss'])}")
    print(f"  Final Validation Loss: {history.history['val_loss'][-1]:.6f}")
    print(f"  Test R² Score: {metrics['test']['r2']:.6f}")


def generate_all_visualizations(history, y_test, y_test_pred, metrics, output_dir):
    """
    Main function to generate all required visualizations
    
    Args:
        history: Keras training history object
        y_test: Actual test values
        y_test_pred: Predicted test values
        metrics: Dictionary containing performance metrics
        output_dir: Directory to save all outputs (Path object)
    """

    print("GENERATING ALL REQUIRED VISUALIZATIONS")

    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all three required visualizations
    create_training_validation_curves(history, output_dir)
    create_diagnostic_plots(y_test, y_test_pred, output_dir, model_name="ANN")
    create_scatter_plot(y_test, y_test_pred, metrics, output_dir, model_name="ANN")
    
    # Save training summary
    save_training_summary(history, metrics, output_dir)



if __name__ == "__main__":
    print("Visualization module ready - import 'generate_all_visualizations' to use")