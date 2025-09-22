import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(y_true, y_pred, average='weighted'):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        average (str): Averaging method for multi-class metrics.
        
    Returns:
        metrics_dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics_dict

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), cmap='Blues', output_dir=None, filename=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names (list): List of class names.
        figsize (tuple): Figure size.
        cmap (str): Colormap for the plot.
        output_dir (str): Directory to save the plot. If None, saves in current directory.
        filename (str): Filename for the saved plot. If None, uses default name.
    """
    # Save confusion matrix data for offline inspection (secure path)
    cm = confusion_matrix(y_true, y_pred)
    debug_dir = "debug_output"
    os.makedirs(debug_dir, exist_ok=True)
    np.save(os.path.join(debug_dir, "confusion_matrix.npy"), cm)
    
    # Create figure
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save to specified output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if filename:
            save_path = os.path.join(output_dir, filename)
        else:
            save_path = os.path.join(output_dir, 'confusion_matrix.png')
    else:
        save_path = filename if filename else 'confusion_matrix.png'
    
    try:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        import traceback
        print(f"Error saving confusion matrix: {str(e)}")
        with open("debug.log", "a") as f:
            f.write(f"\nError saving confusion matrix: {str(e)}\n")
            f.write(traceback.format_exc())
    finally:
        plt.close()

def print_metrics_report(metrics_dict, title="Model Performance Metrics"):
    """
    Print a formatted report of metrics.
    
    Args:
        metrics_dict: Dictionary containing metrics.
        title (str): Title for the report.
    """
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50)
    
    for metric, value in metrics_dict.items():
        print(f"{metric.replace('_', ' ').title():15}: {value:.4f}")
    
    print("=" * 50 + "\n")