import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_training_curves(clients, title_prefix="Client", figsize=(12, 10), output_dir=None):
    """
    Plot training and validation curves for multiple clients.
    
    Args:
        clients: List of Client objects.
        title_prefix (str): Prefix for plot titles.
        figsize (tuple): Figure size.
        output_dir (str): Directory to save the plot. If None, saves in current directory.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Plot training loss
    for client in clients:
        axs[0, 0].plot(client.train_losses, label=f"{title_prefix} {client.client_id}")
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot validation loss
    for client in clients:
        if client.val_losses:  # Check if validation was performed
            axs[0, 1].plot(client.val_losses, label=f"{title_prefix} {client.client_id}")
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot training accuracy
    for client in clients:
        axs[1, 0].plot(client.train_accuracies, label=f"{title_prefix} {client.client_id}")
    axs[1, 0].set_title('Training Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot validation accuracy
    for client in clients:
        if client.val_accuracies:  # Check if validation was performed
            axs[1, 1].plot(client.val_accuracies, label=f"{title_prefix} {client.client_id}")
    axs[1, 1].set_title('Validation Accuracy')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save to specified output directory if provided (secure path)
    if output_dir:
        # Validate output_dir to prevent path traversal
        output_dir = os.path.normpath(output_dir)
        if not output_dir.startswith('.') and '..' not in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'client_training_curves.png')
        else:
            save_path = os.path.join('results', 'client_training_curves.png')
            os.makedirs('results', exist_ok=True)
    else:
        save_path = os.path.join('results', 'client_training_curves.png')
        os.makedirs('results', exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()

def plot_global_performance(server, figsize=(12, 10), output_dir=None):
    """
    Plot global model performance metrics over communication rounds.
    
    Args:
        server: Server object with test metrics.
        figsize (tuple): Figure size.
        output_dir (str): Directory to save the plot. If None, saves in current directory.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Validate server attributes before accessing
    if not hasattr(server, 'test_losses') or not server.test_losses:
        print("No test losses available to plot.")
        return
    if not hasattr(server, 'test_accuracies') or not server.test_accuracies:
        print("No test accuracies available to plot.")
        return
        
    rounds = np.arange(1, len(server.test_losses) + 1)
    
    # Plot test loss
    axs[0, 0].plot(rounds, server.test_losses, 'o-', label='Test Loss')
    axs[0, 0].set_title('Global Model Test Loss')
    axs[0, 0].set_xlabel('Communication Round')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    
    # Plot test accuracy
    axs[0, 1].plot(rounds, server.test_accuracies, 'o-', label='Test Accuracy')
    axs[0, 1].set_title('Global Model Test Accuracy')
    axs[0, 1].set_xlabel('Communication Round')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].grid(True)
    
    # Plot validation loss if available
    if hasattr(server, 'val_losses') and server.val_losses:
        val_rounds = np.arange(1, len(server.val_losses) + 1)
        axs[1, 0].plot(val_rounds, server.val_losses, 'o-', color='orange', label='Validation Loss')
        axs[1, 0].set_title('Global Model Validation Loss')
        axs[1, 0].set_xlabel('Communication Round')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].grid(True)
    
    # Plot validation accuracy if available
    if hasattr(server, 'val_accuracies') and server.val_accuracies:
        val_acc_rounds = np.arange(1, len(server.val_accuracies) + 1)
        axs[1, 1].plot(val_acc_rounds, server.val_accuracies, 'o-', color='orange', label='Validation Accuracy')
        axs[1, 1].set_title('Global Model Validation Accuracy')
        axs[1, 1].set_xlabel('Communication Round')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save to specified output directory if provided (secure path)
    if output_dir:
        # Validate output_dir to prevent path traversal
        output_dir = os.path.normpath(output_dir)
        if not output_dir.startswith('.') and '..' not in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'global_model_performance.png')
        else:
            save_path = os.path.join('results', 'global_model_performance.png')
            os.makedirs('results', exist_ok=True)
    else:
        save_path = os.path.join('results', 'global_model_performance.png')
        os.makedirs('results', exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()

def plot_client_importance(client_ids, importance_scores, figsize=(10, 6), output_dir=None):
    """
    Plot client importance scores.
    
    Args:
        client_ids: List of client IDs.
        importance_scores: List of importance scores.
        figsize (tuple): Figure size.
        output_dir (str): Directory to save the plot. If None, saves in current directory.
    """
    plt.figure(figsize=figsize)
    bars = plt.bar(client_ids, importance_scores)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Client ID')
    plt.ylabel('Importance Score')
    plt.title('Client Importance Scores')
    plt.xticks(client_ids)
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Save to specified output directory if provided (secure path)
    if output_dir:
        # Validate output_dir to prevent path traversal
        output_dir = os.path.normpath(output_dir)
        if not output_dir.startswith('.') and '..' not in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'client_importance_scores.png')
        else:
            save_path = os.path.join('results', 'client_importance_scores.png')
            os.makedirs('results', exist_ok=True)
    else:
        save_path = os.path.join('results', 'client_importance_scores.png')
        os.makedirs('results', exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()


def plot_energy_metrics(energy_tracker, figsize=(14, 10), output_dir=None):
    """
    Plot energy consumption metrics over communication rounds.
    
    Args:
        energy_tracker: EnergyTracker object with energy metrics.
        figsize (tuple): Figure size.
        output_dir (str): Directory to save the plot. If None, saves in current directory.
    """
    if not hasattr(energy_tracker, 'global_metrics') or not energy_tracker.global_metrics:
        print("No energy metrics available to plot.")
        return
    
    # Extract metrics
    rounds = []
    energy_consumption = []
    accuracy = []
    energy_efficiency = []
    
    for round_num, metrics in energy_tracker.global_metrics.items():
        rounds.append(round_num)
        energy_consumption.append(metrics.get('energy_consumed', 0))
        accuracy.append(metrics.get('accuracy', 0))
        # Calculate energy efficiency (accuracy per joule)
        if metrics.get('energy_consumed', 0) > 0:
            efficiency = metrics.get('accuracy', 0) / metrics.get('energy_consumed', 1)
        else:
            efficiency = 0
        energy_efficiency.append(efficiency)
    
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Plot energy consumption
    axs[0, 0].plot(rounds, energy_consumption, 'o-', color='green', label='Energy Consumption')
    axs[0, 0].set_title('Energy Consumption per Round')
    axs[0, 0].set_xlabel('Communication Round')
    axs[0, 0].set_ylabel('Energy (Joules)')
    axs[0, 0].grid(True)
    
    # Plot accuracy
    axs[0, 1].plot(rounds, accuracy, 'o-', color='blue', label='Accuracy')
    axs[0, 1].set_title('Model Accuracy per Round')
    axs[0, 1].set_xlabel('Communication Round')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].grid(True)
    
    # Plot energy efficiency (accuracy per joule)
    axs[1, 0].plot(rounds, energy_efficiency, 'o-', color='purple', label='Energy Efficiency')
    axs[1, 0].set_title('Energy Efficiency (Accuracy per Joule)')
    axs[1, 0].set_xlabel('Communication Round')
    axs[1, 0].set_ylabel('Efficiency (Acc/J)')
    axs[1, 0].grid(True)
    
    # Plot accuracy vs energy consumption scatter
    axs[1, 1].scatter(energy_consumption, accuracy, c=rounds, cmap='viridis', alpha=0.8)
    axs[1, 1].set_title('Accuracy vs Energy Consumption')
    axs[1, 1].set_xlabel('Energy Consumption (Joules)')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].grid(True)
    
    # Add colorbar for round numbers
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axs[1, 1])
    cbar.set_label('Communication Round')
    
    plt.tight_layout()
    
    # Save to specified output directory if provided (secure path)
    if output_dir:
        # Validate output_dir to prevent path traversal
        output_dir = os.path.normpath(output_dir)
        if not output_dir.startswith('.') and '..' not in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'energy_metrics.png')
        else:
            save_path = os.path.join('results', 'energy_metrics.png')
            os.makedirs('results', exist_ok=True)
    else:
        save_path = os.path.join('results', 'energy_metrics.png')
        os.makedirs('results', exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()


def plot_client_energy_heatmap(clients, figsize=(12, 8), output_dir=None):
    """
    Plot heatmap of client energy metrics.
    
    Args:
        clients: List of Client objects with energy trackers.
        figsize (tuple): Figure size.
        output_dir (str): Directory to save the plot. If None, saves in current directory.
    """
    # Extract client energy metrics
    client_ids = []
    energy_consumption = []
    training_efficiency = []
    
    for client in clients:
        if hasattr(client, 'energy_tracker') and client.energy_tracker is not None:
            metrics = client.energy_tracker.get_metrics()
            if metrics:
                client_ids.append(f'Client {client.client_id}')
                energy_consumption.append(metrics.get('total_energy', 0))
                training_efficiency.append(metrics.get('training_efficiency', 0))
    
    if not client_ids:
        print("No client energy metrics available to plot.")
        return
    
    # Create DataFrame for heatmap
    data = {
        'Client ID': client_ids,
        'Energy Consumption (J)': energy_consumption,
        'Training Efficiency': training_efficiency
    }
    df = pd.DataFrame(data).set_index('Client ID')
    
    # Create custom colormap for better visualization
    energy_cmap = LinearSegmentedColormap.from_list('energy_cmap', ['#d4eeff', '#0068c9'])
    efficiency_cmap = LinearSegmentedColormap.from_list('efficiency_cmap', ['#ffedd4', '#f28e2b'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot energy consumption heatmap
    sns.heatmap(df[['Energy Consumption (J)']], annot=True, fmt='.2f', cmap=energy_cmap, ax=ax1)
    ax1.set_title('Client Energy Consumption')
    
    # Plot training efficiency heatmap
    sns.heatmap(df[['Training Efficiency']], annot=True, fmt='.4f', cmap=efficiency_cmap, ax=ax2)
    ax2.set_title('Client Training Efficiency (Acc/J)')
    
    plt.tight_layout()
    
    # Save to specified output directory if provided (secure path)
    if output_dir:
        # Validate output_dir to prevent path traversal
        output_dir = os.path.normpath(output_dir)
        if not output_dir.startswith('.') and '..' not in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'client_energy_heatmap.png')
        else:
            save_path = os.path.join('results', 'client_energy_heatmap.png')
            os.makedirs('results', exist_ok=True)
    else:
        save_path = os.path.join('results', 'client_energy_heatmap.png')
        os.makedirs('results', exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()


def plot_carbon_footprint(energy_tracker, figsize=(10, 6), output_dir=None):
    """
    Plot carbon footprint estimation based on energy consumption.
    
    Args:
        energy_tracker: EnergyTracker object with energy metrics.
        figsize (tuple): Figure size.
        output_dir (str): Directory to save the plot. If None, saves in current directory.
    """
    if not hasattr(energy_tracker, 'global_metrics') or not energy_tracker.global_metrics:
        print("No energy metrics available to plot carbon footprint.")
        return
    
    # Extract metrics
    rounds = []
    carbon_emissions = []
    
    # Carbon intensity factor (gCO2eq/kWh) - average for mixed energy sources
    # This is a simplified estimate and can be adjusted based on region/energy source
    carbon_intensity = 475  # global average carbon intensity
    
    for round_num, metrics in energy_tracker.global_metrics.items():
        rounds.append(round_num)
        # Convert Joules to kWh and calculate emissions
        energy_kwh = metrics.get('energy_consumed', 0) / 3600000  # J to kWh
        emissions = energy_kwh * carbon_intensity  # gCO2eq
        carbon_emissions.append(emissions)
    
    plt.figure(figsize=figsize)
    plt.bar(rounds, carbon_emissions, color='darkgreen', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(carbon_emissions):
        plt.text(rounds[i], v + 0.0001, f'{v:.5f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.xlabel('Communication Round')
    plt.ylabel('Carbon Emissions (gCO2eq)')
    plt.title('Estimated Carbon Footprint per Round')
    plt.xticks(rounds)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add total carbon footprint
    total_emissions = sum(carbon_emissions)
    plt.figtext(0.5, 0.01, f'Total Carbon Footprint: {total_emissions:.5f} gCO2eq', 
                ha='center', fontsize=10, bbox={'facecolor':'lightgreen', 'alpha':0.5, 'pad':5})
    
    plt.tight_layout()
    
    # Save to specified output directory if provided (secure path)
    if output_dir:
        # Validate output_dir to prevent path traversal
        output_dir = os.path.normpath(output_dir)
        if not output_dir.startswith('.') and '..' not in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'carbon_footprint.png')
        else:
            save_path = os.path.join('results', 'carbon_footprint.png')
            os.makedirs('results', exist_ok=True)
    else:
        save_path = os.path.join('results', 'carbon_footprint.png')
        os.makedirs('results', exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()