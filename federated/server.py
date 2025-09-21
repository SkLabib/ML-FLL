import torch
import numpy as np
from .clustering import k_hard_means_clustering
import copy
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comprehensive_energy_tracker import EnergyTracker

class Server:
    def __init__(self, global_model, test_loader=None, device=None, 
                 mu=0.01, selection_ratio=0.7, early_stopping_rounds=5,
                 energy_tracker=None):
        """
        Initialize the federated learning server.
        
        Args:
            global_model: The global neural network model.
            test_loader: DataLoader for test data.
            device: Device to run the model on (CPU).
        """
        # Validate and set device
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.global_model = global_model.to(self.device)
        self.test_loader = test_loader
        
        # FedProx hyperparameter (proximal term weight)
        self.mu = mu
        
        # Client selection parameters
        self.selection_ratio = selection_ratio  # Top percentage of clients to select
        
        # Early stopping parameters
        self.early_stopping_rounds = early_stopping_rounds
        self.best_accuracy = 0.0
        self.rounds_without_improvement = 0
        
        # Energy tracking
        self.energy_tracker = energy_tracker
        
        # Divergence monitoring
        self.divergence_log = []
        
        # For tracking global model performance
        self.test_accuracies = []
        self.test_losses = []
        self.val_accuracies = []
        self.val_losses = []
    
    def aggregate(self, client_weights, client_importances=None, client_energy_metrics=None):
        """
        Aggregate client model weights using FedProx and weighted averaging based on importance.
        
        Args:
            client_weights: List of client model weights.
            client_importances: List of client importance scores.
            client_energy_metrics: List of client energy efficiency metrics.
            
        Returns:
            global_weights: Aggregated global model weights.
        """
        # If no importance scores provided, use equal weights
        if client_importances is None:
            client_importances = np.ones(len(client_weights))
            
        # Get current global model weights for FedProx
        global_model_weights = self.global_model.get_weights()
        
        # Apply intelligent client selection if energy metrics are provided
        if client_energy_metrics is not None:
            selected_indices = self.select_clients(client_importances, client_energy_metrics)
            print(f"Selected {len(selected_indices)}/{len(client_weights)} clients for aggregation")
            
            # Validate selected clients
            if not selected_indices:
                raise ValueError("No clients selected for aggregation")
                
            # Filter weights and importances to only include selected clients
            selected_weights = [client_weights[i] for i in selected_indices]
            selected_importances = [client_importances[i] for i in selected_indices]
        else:
            selected_weights = client_weights
            selected_importances = client_importances
        
        # Apply smoothing to importance scores
        selected_importances = smooth_importance_scores(selected_importances)
        
        # Normalize importance scores
        total_importance = sum(selected_importances)
        normalized_importances = [imp / total_importance for imp in selected_importances]
        
        print(f"Smoothed importance weights: {normalized_importances}")
        
        # Initialize global weights with zeros like the first client's weights
        global_weights = [torch.zeros_like(w) for w in selected_weights[0]]
        
        # Weighted average of client weights with FedProx regularization
        for client_idx, weights in enumerate(selected_weights):
            importance = normalized_importances[client_idx]
            for i, w in enumerate(weights):
                # Apply FedProx regularization: w_i = w_i - mu * (w_i - w_global)
                if self.mu > 0:
                    regularized_w = w - self.mu * (w - global_model_weights[i])
                    global_weights[i] += importance * regularized_w
                else:
                    global_weights[i] += importance * w
        
        # Update global model with aggregated weights
        self.global_model.set_weights(global_weights)
        
        return global_weights
        
    def select_clients(self, client_importances, client_energy_metrics):
        """
        Select clients based on data quality (importance) and energy efficiency.
        
        Args:
            client_importances: List of client importance scores (data quality).
            client_energy_metrics: List of client energy efficiency metrics.
            
        Returns:
            selected_indices: Indices of selected clients.
        """
        # Calculate selection score = (data_quality × energy_efficiency)
        selection_scores = []
        
        for i, (importance, energy_metric) in enumerate(zip(client_importances, client_energy_metrics)):
            # Higher importance is better (data quality)
            # Higher training_efficiency is better (accuracy gain per energy unit)
            if isinstance(energy_metric, dict):
                # If energy_metric is a dictionary with training_efficiency
                efficiency = energy_metric.get('training_efficiency', 0)
                if efficiency < 0:  # Avoid negative efficiency values
                    efficiency = 0
            else:
                # If energy_metric is a direct value
                efficiency = max(0, energy_metric)
                
            # Combine metrics (both higher is better)
            score = importance * (1 + efficiency)  # Add 1 to ensure non-zero score even with zero efficiency
            selection_scores.append((i, score))
        
        # Sort by score in descending order
        selection_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top percentage of clients
        num_to_select = max(1, int(self.selection_ratio * len(client_importances)))
        selected_indices = [idx for idx, _ in selection_scores[:num_to_select]]
        
        return selected_indices
    
    def cluster_and_aggregate(self, client_weights, client_energy_metrics=None, max_k=4, random_state=42):
        """
        Cluster client weights using silhouette-based adaptive clustering and aggregate with importance weighting.
        
        Args:
            client_weights: List of client model weights.
            client_energy_metrics: List of client energy efficiency metrics.
            max_k (int): Maximum number of clusters to consider (default: 4).
            random_state (int): Random seed for reproducibility.
            
        Returns:
            global_weights: Aggregated global model weights.
            cluster_assignments: Cluster assignment for each client.
            importance_scores: Importance score for each client.
        """
        # Start energy tracking for aggregation
        if self.energy_tracker:
            self.energy_tracker.start_tracking()
        
        # Apply silhouette-based adaptive clustering with round-aware reclustering
        from .clustering import flatten_weights
        flattened_weights = flatten_weights(client_weights)
        unique_clients = len(np.unique(flattened_weights, axis=0))
        
        # Get current round number for adaptive reclustering
        current_round = len(self.test_accuracies)
        
        # Always try to maintain multiple clusters, prevent single cluster collapse
        min_clusters = 2  # Always maintain at least 2 clusters
        
        if unique_clients < 2:
            print(f"Only {unique_clients} unique clients detected - forcing minimum {min_clusters} clusters")
            cluster_assignments, optimal_k, importance_scores = k_hard_means_clustering(
                client_weights, max_k, random_state, force_k=min_clusters, 
                divergence_log=self.divergence_log, round_num=current_round)
        else:
            print(f"Found {unique_clients} unique clients - using silhouette-based adaptive clustering")
            cluster_assignments, optimal_k, importance_scores = k_hard_means_clustering(
                client_weights, max_k, random_state, force_k=None, 
                divergence_log=self.divergence_log, round_num=current_round)
            
            # CRITICAL: Ensure we never collapse to single cluster
            if optimal_k == 1 and len(client_weights) > 1:
                print(f"CRITICAL: Clustering collapsed to 1 cluster - forcing {min_clusters} clusters")
                cluster_assignments, optimal_k, importance_scores = k_hard_means_clustering(
                    client_weights, max_k, random_state, force_k=min_clusters, 
                    divergence_log=self.divergence_log, round_num=current_round)
        
        print(f"Round {current_round} clustering results: {optimal_k} clusters identified")
        print(f"Cluster assignments: {cluster_assignments}")
        print(f"Importance scores: {importance_scores}")
        
        # Validate clustering stability
        if optimal_k == 1:
            raise ValueError(f"CRITICAL ERROR: Single cluster detected in round {current_round}! This should never happen.")
        
        # Perform cluster-wise aggregation first
        cluster_weights = {}
        for cluster_id in range(optimal_k):
            # Get indices of clients in this cluster
            cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
            
            if not cluster_indices:  # Skip empty clusters
                continue
                
            print(f"Aggregating cluster {cluster_id} with {len(cluster_indices)} clients")
            
            # Get weights and importance scores for clients in this cluster
            cluster_client_weights = [client_weights[i] for i in cluster_indices]
            cluster_importance = [importance_scores[i] for i in cluster_indices]
            
            # Normalize importance scores within cluster
            total_importance = sum(cluster_importance)
            if total_importance > 0:
                normalized_importance = [imp / total_importance for imp in cluster_importance]
            else:
                normalized_importance = [1.0 / len(cluster_indices)] * len(cluster_indices)
            
            # Initialize cluster weights with zeros like the first client's weights
            cluster_weights[cluster_id] = [torch.zeros_like(w) for w in cluster_client_weights[0]]
            
            # Weighted average of client weights within cluster
            for client_idx, weights in enumerate(cluster_client_weights):
                importance = normalized_importance[client_idx]
                for i, w in enumerate(weights):
                    cluster_weights[cluster_id][i] += importance * w
        
        # Fed-CAM approach: Combine cluster weights with shared global component
        # Initialize global weights with zeros like the first client's weights
        global_weights = [torch.zeros_like(w) for w in client_weights[0]]
        
        # Calculate cluster weights based on cluster size and importance
        num_clusters = len(cluster_weights)
        if num_clusters > 0:
            # Fed-CAM: Create shared global component (average of all clients)
            shared_component = [torch.zeros_like(w) for w in client_weights[0]]
            total_clients = len(client_weights)
            
            for weights in client_weights:
                for i, w in enumerate(weights):
                    shared_component[i] += w / total_clients
            
            # Combine clusters with shared component (70% cluster-specific, 30% shared)
            cluster_ratio = 0.7
            shared_ratio = 0.3
            
            # Weight clusters by their size and average importance
            cluster_sizes = {}
            cluster_importance = {}
            
            for cluster_id in cluster_weights.keys():
                cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
                cluster_sizes[cluster_id] = len(cluster_indices)
                cluster_importance[cluster_id] = np.mean([importance_scores[i] for i in cluster_indices])
            
            # Normalize cluster weights by size and importance
            total_weighted_size = sum(cluster_sizes[cid] * cluster_importance[cid] for cid in cluster_weights.keys())
            
            for cluster_id, weights in cluster_weights.items():
                cluster_weight = (cluster_sizes[cluster_id] * cluster_importance[cluster_id]) / total_weighted_size
                
                for i, w in enumerate(weights):
                    # Fed-CAM: cluster-specific + shared component
                    cluster_component = cluster_ratio * w
                    shared_comp = shared_ratio * shared_component[i]
                    global_weights[i] += cluster_weight * (cluster_component + shared_comp)
        else:
            # Fallback to regular aggregation if no valid clusters
            global_weights = self.aggregate(client_weights, importance_scores, client_energy_metrics)
        
        # Stop energy tracking and log metrics
        if self.energy_tracker:
            round_num = len(self.test_accuracies)  # Current round number
            energy_metrics = self.energy_tracker.stop_tracking(round_num=round_num, operation_type="aggregation")
            print(f"Aggregation energy: {energy_metrics['total_energy_wh']:.6f} Wh")
        
        return global_weights, cluster_assignments, importance_scores
    
    def evaluate(self, criterion=None, val_loader=None):
        """
        Evaluate the global model on the test set and optionally on validation set.
        
        Args:
            criterion: Loss function (default: CrossEntropyLoss).
            val_loader: Optional validation data loader. If provided, will be used for evaluation.
            
        Returns:
            test_loss, test_accuracy: Test loss and accuracy.
            all_preds, all_targets: Predictions and targets for confusion matrix.
        """
        # Determine which loader to use for evaluation
        eval_loader = val_loader if val_loader is not None else self.test_loader
        
        if eval_loader is None:
            print("No data loader provided for evaluation.")
            return None, None, None, None
        
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        
        # Start energy tracking if available
        if self.energy_tracker:
            self.energy_tracker.start()
        
        # Evaluate on the selected loader
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        test_loss = test_loss / len(eval_loader.dataset)
        test_accuracy = correct / total
        
        # Store results based on loader type
        if val_loader is not None:
            # This is a validation evaluation
            self.val_losses.append(test_loss)
            self.val_accuracies.append(test_accuracy)
            
            # Check for early stopping
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                self.rounds_without_improvement = 0
                # Save best model weights
                self.best_model_weights = copy.deepcopy(self.global_model.get_weights())
            else:
                self.rounds_without_improvement += 1
                
            print(f"Global Model - Val Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
            print(f"Rounds without improvement: {self.rounds_without_improvement}/{self.early_stopping_rounds}")
        else:
            # This is a test evaluation
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)
        
        # Stop energy tracking and log metrics
        if self.energy_tracker:
            energy_consumed = self.energy_tracker.stop_tracking()
            self.energy_tracker.update_global_metrics(
                accuracy=test_accuracy,
                energy_consumed=energy_consumed,
                round_num=len(self.test_accuracies) if val_loader is None else len(self.val_accuracies)
            )
            if isinstance(energy_consumed, dict):
                if 'total_energy_wh' in energy_consumed:
                    print(f"Evaluation energy consumption: {energy_consumed['total_energy_wh']:.4f}Wh")
                else:
                    print(f"Evaluation energy consumption: {energy_consumed}")
            else:
                print(f"Evaluation energy consumption: {energy_consumed}")
        
        # Print appropriate message based on evaluation type
        if val_loader is not None:
            print(f"Global Model - Val Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        else:
            print(f"Global Model - Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy, np.array(all_preds), np.array(all_targets)
    
    def _evaluate_on_loader(self, data_loader, criterion):
        """
        Helper method to evaluate model on a specific data loader.
        
        Args:
            data_loader: DataLoader to evaluate on.
            criterion: Loss function.
            
        Returns:
            loss, accuracy: Evaluation metrics.
        """
        self.global_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        loss = running_loss / len(data_loader.dataset)
        accuracy = correct / total
        
        return loss, accuracy
        
    def should_early_stop(self):
        """
        Check if training should be stopped early due to plateau in validation accuracy.
        
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.rounds_without_improvement >= self.early_stopping_rounds:
            print(f"Early stopping triggered after {self.rounds_without_improvement} rounds without improvement")
            # Restore best model weights
            if hasattr(self, 'best_model_weights'):
                self.global_model.set_weights(self.best_model_weights)
                print("Restored best model weights")
            return True
        return False
        
    def async_aggregate(self, client_weights_list, client_importances_list, timestamps=None):
        """
        Perform asynchronous aggregation of client weights that arrive at different times.
        
        Args:
            client_weights_list: List of client model weights.
            client_importances_list: List of client importance scores.
            timestamps: List of timestamps when each client update was received.
            
        Returns:
            global_weights: Aggregated global model weights.
        """
        if timestamps is None:
            # If no timestamps provided, treat all updates as equally recent
            timestamps = [1.0] * len(client_weights_list)
            
        # Calculate staleness weights (more recent updates get higher weight)
        max_timestamp = max(timestamps)
        staleness_weights = [ts / max_timestamp for ts in timestamps]
        
        # Combine importance with staleness
        combined_weights = []
        for imp, stale in zip(client_importances_list, staleness_weights):
            combined_weights.append(imp * stale)
            
        # Use regular aggregation with combined weights
        return self.aggregate(client_weights_list, combined_weights)


def smooth_importance_scores(importance_scores, min_weight=0.05, max_weight=0.5):
    """
    Apply smoothing to importance scores to prevent a single client from dominating.
    
    Args:
        importance_scores: Raw importance scores for each client.
        min_weight: Minimum weight any client should have (as a fraction of total).
        max_weight: Maximum weight any client should have (as a fraction of total).
        
    Returns:
        smoothed_scores: Smoothed importance scores that satisfy constraints.
    """
    n_clients = len(importance_scores)
    
    # Ensure min_weight * n_clients <= 1 to allow all clients to have at least min_weight
    if min_weight * n_clients > 1:
        min_weight = 1 / n_clients
        print(f"Warning: min_weight adjusted to {min_weight:.4f} to accommodate all clients")
    
    # Convert to numpy array if not already
    scores = np.array(importance_scores)
    
    # Step 1: Apply softmax to maintain relative ordering while bounding values
    # Using temperature parameter to control the softness of the distribution
    temperature = 1.0
    exp_scores = np.exp(scores / temperature)
    softmax_scores = exp_scores / np.sum(exp_scores)
    
    # Step 2: Iteratively adjust scores to satisfy min and max constraints
    smoothed_scores = softmax_scores.copy()
    
    # First pass: Handle maximum weight constraint
    while np.max(smoothed_scores) > max_weight:
        # Find the client with the highest score
        max_idx = np.argmax(smoothed_scores)
        excess = smoothed_scores[max_idx] - max_weight
        
        # Redistribute excess to other clients proportionally to their current scores
        smoothed_scores[max_idx] = max_weight
        
        # Create a mask for redistribution (exclude the max client)
        redistribution_mask = np.ones(n_clients, dtype=bool)
        redistribution_mask[max_idx] = False
        
        # Redistribute excess proportionally
        if np.sum(smoothed_scores[redistribution_mask]) > 0:
            redistribution_weights = smoothed_scores[redistribution_mask] / np.sum(smoothed_scores[redistribution_mask])
            smoothed_scores[redistribution_mask] += excess * redistribution_weights
    
    # Second pass: Handle minimum weight constraint
    below_min = smoothed_scores < min_weight
    if np.any(below_min):
        # Calculate how much we need to add to bring below-min clients up to min_weight
        deficit = np.sum(min_weight - smoothed_scores[below_min])
        
        # Set all below-min clients to min_weight
        smoothed_scores[below_min] = min_weight
        
        # Create a mask for reduction (exclude below-min clients, respect max_weight)
        reduction_mask = ~below_min & (smoothed_scores > min_weight)
        
        # Calculate how much we can reduce from each client above min_weight
        if np.sum(smoothed_scores[reduction_mask]) > 0:
            # Calculate reduction proportionally to how much above min_weight each client is
            room_above_min = smoothed_scores[reduction_mask] - min_weight
            reduction_weights = room_above_min / np.sum(room_above_min)
            smoothed_scores[reduction_mask] -= deficit * reduction_weights
    
    # Final normalization to ensure sum is exactly 1.0
    smoothed_scores = smoothed_scores / np.sum(smoothed_scores)
    
    return smoothed_scores