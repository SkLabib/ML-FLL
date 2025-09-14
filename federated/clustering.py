import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import os

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
            # Check if tensor is already on CPU to avoid unnecessary operations
            if w.device.type != 'cpu':
                w_cpu = w.cpu()
            else:
                w_cpu = w
            client_weights.append(w_cpu.numpy().flatten())
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
    
    # Save original features for debugging (secure path)
    debug_dir = "debug_output"
    os.makedirs(debug_dir, exist_ok=True)
    np.save(os.path.join(debug_dir, "cluster_features.npy"), flattened_weights)
    
    # Check for NaN/inf values
    nan_count = np.isnan(flattened_weights).sum()
    inf_count = np.isinf(flattened_weights).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"WARNING: Found {nan_count} NaN values and {inf_count} inf values in features")
        # Replace NaN/inf with zeros
        flattened_weights = np.nan_to_num(flattened_weights, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for duplicate points
    unique_rows = np.unique(flattened_weights, axis=0)
    n_unique = unique_rows.shape[0]
    print(f"Number of unique client weight vectors: {n_unique} out of {num_clients}")
    
    # Early return if all client weight vectors are identical or only one unique vector
    if n_unique <= 1:
        print("Only one unique client weight vector found, skipping clustering.")
        cluster_assignments = np.zeros(num_clients, dtype=int)
        importance_scores = np.ones(num_clients, dtype=float)
        return cluster_assignments, 1, importance_scores
    
    # Add differential noise to create meaningful weight divergence
    if n_unique < num_clients:
        print("Adding differential noise to create weight divergence for better clustering")
        np.random.seed(random_state)
        # Use larger noise for better divergence
        jitter = np.random.normal(0, 1e-4, flattened_weights.shape)
        # Add client-specific noise patterns
        for i in range(num_clients):
            client_noise = np.random.normal(0, 1e-4 * (i + 1), flattened_weights[i].shape)
            flattened_weights[i] += jitter[i] + client_noise
    else:
        # Even with unique weights, add small noise to improve clustering stability
        print("Adding stability noise to improve clustering")
        np.random.seed(random_state)
        stability_noise = np.random.normal(0, 1e-5, flattened_weights.shape)
        flattened_weights = flattened_weights + stability_noise
    
    # Normalize features using StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_weights = scaler.fit_transform(flattened_weights)
    
    # Apply dimensionality reduction if features are high-dimensional
    if scaled_weights.shape[1] > 50:
        from sklearn.decomposition import PCA
        # Ensure n_components doesn't exceed min(n_samples, n_features)
        n_components = min(scaled_weights.shape[0] - 1, scaled_weights.shape[1], 50)
        if n_components > 0:  # Ensure we have at least 1 component
            try:
                pca = PCA(n_components=n_components, random_state=random_state)
                scaled_weights = pca.fit_transform(scaled_weights)
                print(f"Reduced feature dimensions from {flattened_weights.shape[1]} to {scaled_weights.shape[1]}")
                print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_)*100:.2f}%")
            except ValueError as e:
                print(f"PCA failed: {e}. Skipping dimensionality reduction.")
        else:
            print("Skipping PCA: insufficient samples for dimensionality reduction")

    
    # Set maximum k to consider
    if max_k is None:
        max_k = min(num_clients - 1, 5)  # Default: min(num_clients-1, 5)
    else:
        max_k = min(max_k, num_clients - 1)  # Ensure max_k is valid
    
    # If only 1 client, return single cluster
    if num_clients == 1:
        return np.array([0]), 1, np.array([1.0])
    
    # Adjust n_clusters if needed based on unique points
    if n_unique < max_k:
        print(f"WARNING: Number of unique points ({n_unique}) is less than max_k ({max_k})")
        print(f"Adjusting max_k to {n_unique}")
        max_k = n_unique
        if force_k is not None and force_k > n_unique:
            print(f"Adjusting force_k from {force_k} to {n_unique}")
            force_k = n_unique
    
    # If force_k is provided and valid, use it directly
    if force_k is not None and 2 <= force_k <= num_clients:
        optimal_kmeans = KMeans(n_clusters=force_k, random_state=random_state, n_init=20)
        cluster_assignments = optimal_kmeans.fit_predict(scaled_weights)
        optimal_k = force_k
        
        # Log clustering metrics
        inertia = optimal_kmeans.inertia_
        print(f"KMeans with k={force_k}: inertia={inertia:.4f}")
        
        # Log cluster sizes
        unique_labels, counts = np.unique(cluster_assignments, return_counts=True)
        print("Cluster sizes:")
        for label, count in zip(unique_labels, counts):
            print(f"  Cluster {label}: {count} clients")
        
        # Calculate silhouette score if possible
        if len(np.unique(cluster_assignments)) > 1 and len(cluster_assignments) > len(np.unique(cluster_assignments)):
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(scaled_weights, cluster_assignments)
            print(f"Silhouette score: {silhouette:.4f}")
    else:
        # Try different values of k and select the best one using silhouette score
        silhouette_scores = []
        kmeans_models = []
        
        # Start from k=2 (minimum for silhouette score)
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
            cluster_labels = kmeans.fit_predict(scaled_weights)
            
            # Skip if any cluster has only one sample (can't compute silhouette)
            if len(np.unique(cluster_labels)) < k:
                silhouette_scores.append(-1)  # Invalid score
                print(f"KMeans with k={k}: Found only {len(np.unique(cluster_labels))} distinct clusters")
            else:
                from sklearn.metrics import silhouette_score
                score = silhouette_score(scaled_weights, cluster_labels)
                silhouette_scores.append(score)
                print(f"KMeans with k={k}: silhouette={score:.4f}, inertia={kmeans.inertia_:.4f}")
            
            kmeans_models.append(kmeans)
        
        # Select optimal k (if all scores are invalid, use k=2)
        if all(score == -1 for score in silhouette_scores) or len(silhouette_scores) == 0:
            # If no valid clustering found, create a simple 2-cluster solution
            optimal_k = min(2, num_clients)
            print(f"All silhouette scores are invalid, defaulting to k={optimal_k}")
            if optimal_k == 1:
                cluster_assignments = np.zeros(num_clients, dtype=int)
                # Create dummy kmeans for importance calculation
                optimal_kmeans = type('DummyKMeans', (), {})() 
                optimal_kmeans.cluster_centers_ = np.array([scaled_weights.mean(axis=0)])
            else:
                optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=20)
                cluster_assignments = optimal_kmeans.fit_predict(scaled_weights)
        else:
            # Filter out invalid scores
            valid_scores = [(i, score) for i, score in enumerate(silhouette_scores) if score != -1]
            if valid_scores:
                # Get index of best valid score
                best_idx, best_score = max(valid_scores, key=lambda x: x[1])
                print(f"Selected optimal k={best_idx+2} with silhouette={best_score:.4f}")
                optimal_k = best_idx + 2  # Convert index to k value
                optimal_kmeans = kmeans_models[best_idx]
                cluster_assignments = optimal_kmeans.predict(scaled_weights)
            else:
                # Fallback if somehow we have no valid scores (shouldn't happen due to earlier check)
                optimal_k = min(2, num_clients)
                print(f"No valid silhouette scores found, defaulting to k={optimal_k}")
                optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=20)
                cluster_assignments = optimal_kmeans.fit_predict(scaled_weights)
        
        # Log cluster sizes
        unique_labels, counts = np.unique(cluster_assignments, return_counts=True)
        print("Cluster sizes:")
        for label, count in zip(unique_labels, counts):
            print(f"  Cluster {label}: {count} clients")
    
    # Ensure we never have a single cluster - add artificial diversity if needed
    if len(np.unique(cluster_assignments)) == 1 and num_clients > 1:
        print("WARNING: KMeans assigned all clients to the same cluster. Adding artificial diversity.")
        
        # Force at least 2 clusters by splitting clients
        min_clusters = min(2, num_clients)
        
        # Try Agglomerative Clustering first
        from sklearn.cluster import AgglomerativeClustering
        try:
            agg_cluster = AgglomerativeClustering(n_clusters=min_clusters)
            agg_labels = agg_cluster.fit_predict(scaled_weights)
            
            if len(np.unique(agg_labels)) > 1:
                print(f"Agglomerative Clustering found {len(np.unique(agg_labels))} clusters")
                cluster_assignments = agg_labels
                optimal_k = len(np.unique(agg_labels))
                
                # Create a dummy KMeans model for importance score calculation
                from sklearn.neighbors import NearestCentroid
                centroid_classifier = NearestCentroid()
                centroid_classifier.fit(scaled_weights, agg_labels)
                optimal_kmeans = type('DummyKMeans', (), {})()
                optimal_kmeans.cluster_centers_ = centroid_classifier.centroids_
            else:
                raise ValueError("Agglomerative clustering also failed")
        except:
            # Fallback: manually assign clients to different clusters
            print("Forcing artificial cluster diversity")
            cluster_assignments = np.array([i % min_clusters for i in range(num_clients)])
            optimal_k = min_clusters
            
            # Create artificial cluster centers
            cluster_centers = []
            for k in range(optimal_k):
                cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == k]
                if cluster_indices:
                    center = np.mean(scaled_weights[cluster_indices], axis=0)
                else:
                    center = scaled_weights[k % num_clients]  # Fallback
                cluster_centers.append(center)
            
            optimal_kmeans = type('DummyKMeans', (), {})()
            optimal_kmeans.cluster_centers_ = np.array(cluster_centers)
    
    # Save cluster labels for debugging (secure path)
    np.save(os.path.join(debug_dir, "cluster_labels.npy"), cluster_assignments)
    
    # Calculate importance scores based on distance to cluster center
    # Clients closer to their cluster center are more important
    importance_scores = np.zeros(num_clients)
    
    for i in range(num_clients):
        # Get distance to assigned cluster center
        cluster_idx = cluster_assignments[i]
        center = optimal_kmeans.cluster_centers_[cluster_idx]
        distance = np.linalg.norm(scaled_weights[i] - center)
        
        # Convert distance to importance (inverse relationship)
        # Add small epsilon to avoid division by zero
        importance_scores[i] = 1.0 / (distance + 1e-10)
    
    # Normalize importance scores to sum to num_clients
    importance_scores = importance_scores * (num_clients / importance_scores.sum())
    
    return cluster_assignments, optimal_k, importance_scores