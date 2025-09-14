import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch

def flatten_weights(weights_list):
    """
    Flatten a list of model weights into a single vector for clustering.
    
    Args:
        weights_list: List of model weight tensors.
        
    Returns:
        flattened_weights: Flattened numpy array of weights.
    """
    flattened = []
    for weights in weights_list:
        client_weights = []
        for w in weights:
            client_weights.append(w.cpu().numpy().flatten())
        flattened.append(np.concatenate(client_weights))
    return np.array(flattened)

def k_hard_means_clustering(client_weights, max_k=None, random_state=42, force_k=None):
    """
    Apply K-Hard Means clustering to client model weights with automatic or fixed k selection.
    
    Args:
        client_weights: List of client model weights.
        max_k (int): Maximum number of clusters to consider.
        random_state (int): Random seed for reproducibility.
        force_k (int): If provided, forces clustering to use this specific k value.
        
    Returns:
        cluster_assignments: Cluster assignment for each client.
        optimal_k: Optimal number of clusters determined.
        importance_scores: Importance score for each client based on clustering.
    """
    # Flatten weights for clustering
    flattened_weights = flatten_weights(client_weights)
    num_clients = flattened_weights.shape[0]
    
    # Set maximum k to consider
    if max_k is None:
        max_k = min(num_clients - 1, 5)  # Default: min(num_clients-1, 5)
    else:
        max_k = min(max_k, num_clients - 1)  # Ensure max_k is valid
    
    # If only 1 client, return single cluster
    if num_clients == 1:
        return np.array([0]), 1, np.array([1.0])
    
    # If force_k is provided and valid, use it directly
    if force_k is not None and 2 <= force_k <= num_clients:
        kmeans = KMeans(n_clusters=force_k, random_state=random_state, n_init=10)
        cluster_assignments = kmeans.fit_predict(flattened_weights)
        optimal_k = force_k
    else:
        # Try different values of k and select the best one using silhouette score
        silhouette_scores = []
        kmeans_models = []
        
        # Start from k=2 (minimum for silhouette score)
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(flattened_weights)
            
            # Skip if any cluster has only one sample (can't compute silhouette)
            if len(np.unique(cluster_labels)) < k:
                silhouette_scores.append(-1)  # Invalid score
            else:
                score = silhouette_score(flattened_weights, cluster_labels)
                silhouette_scores.append(score)
            
            kmeans_models.append(kmeans)
        
        # Select optimal k (if all scores are invalid, use k=2)
        if all(score == -1 for score in silhouette_scores):
            optimal_idx = 0  # k=2
        else:
            optimal_idx = np.argmax(silhouette_scores)
        
        optimal_k = optimal_idx + 2  # Convert index to k value
        optimal_kmeans = kmeans_models[optimal_idx]
        cluster_assignments = optimal_kmeans.predict(flattened_weights)
    
    # Calculate importance scores based on distance to cluster center
    # Clients closer to their cluster center are more important
    importance_scores = np.zeros(num_clients)
    
    # Get the kmeans model to use for importance calculation
    if force_k is not None and 2 <= force_k <= num_clients:
        kmeans_model = kmeans
    else:
        kmeans_model = optimal_kmeans
    
    for i in range(num_clients):
        # Get distance to assigned cluster center
        cluster_idx = cluster_assignments[i]
        center = kmeans_model.cluster_centers_[cluster_idx]
        distance = np.linalg.norm(flattened_weights[i] - center)
        
        # Convert distance to importance (inverse relationship)
        # Add small epsilon to avoid division by zero
        importance_scores[i] = 1.0 / (distance + 1e-10)
    
    # Normalize importance scores to sum to num_clients
    importance_scores = importance_scores * (num_clients / importance_scores.sum())
    
    return cluster_assignments, optimal_k, importance_scores