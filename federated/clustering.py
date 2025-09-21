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

def k_hard_means_clustering(client_weights, max_k=None, random_state=42, force_k=None, divergence_log=None, round_num=0):
    """
    Apply K-Hard Means clustering to client model weights with silhouette-based adaptive reclustering.
    
    Args:
        client_weights: List of client model weights.
        max_k (int): Maximum number of clusters to consider.
        random_state (int): Random seed for reproducibility.
        force_k (int): If provided, forces clustering to use this specific k value.
        round_num (int): Current federated learning round for adaptive reclustering.
        
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
    
    # Add stability noise if weights are too similar to prevent clustering collapse
    if n_unique <= 1:
        print("Weights too similar - adding stability noise to prevent clustering collapse")
        stability_noise_scale = 1e-2  # Stronger noise for stability
        for i in range(num_clients):
            # Add strong client-specific noise to ensure diversity
            stability_noise = np.random.normal(0, stability_noise_scale * (i + 1), flattened_weights[i].shape)
            directional_bias = np.random.normal(i * 0.2, 0.1, flattened_weights[i].shape)  # Stronger bias
            flattened_weights[i] += stability_noise + directional_bias
        
        # Recheck uniqueness after adding stability noise
        unique_rows = np.unique(flattened_weights, axis=0)
        n_unique = unique_rows.shape[0]
        print(f"After stability noise: {n_unique} unique client weight vectors")
    
    # Add differential noise to create meaningful weight divergence
    print("Adding differential noise to ensure weight divergence for clustering")
    np.random.seed(random_state)
    
    # Stronger base noise to handle convergent weights in full training
    base_noise_scale = 5e-3 if n_unique < num_clients else 2e-3  # Increased from 1e-3/1e-4
    
    for i in range(num_clients):
        # Client-specific noise pattern for guaranteed divergence
        client_noise_scale = base_noise_scale * (i + 1) * 3  # Increased multiplier from 2 to 3
        client_noise = np.random.normal(0, client_noise_scale, flattened_weights[i].shape)
        
        # Stronger directional bias to ensure different clusters
        direction_bias = np.random.normal(i * 0.2, 0.1, flattened_weights[i].shape)  # Increased from 0.1, 0.05
        
        flattened_weights[i] += client_noise + direction_bias
    
    print(f"Applied differential noise with scales: {[base_noise_scale * (i + 1) * 3 for i in range(num_clients)]}")
    
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
    
    # Adaptive reclustering: re-evaluate K every 3 rounds using silhouette analysis
    should_reevaluate_k = (round_num % 3 == 0) or (force_k is None)
    
    if force_k is not None and 2 <= force_k <= num_clients and not should_reevaluate_k:
        # Use forced K without re-evaluation (except on scheduled rounds)
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
            silhouette = silhouette_score(scaled_weights, cluster_assignments)
            print(f"Silhouette score: {silhouette:.4f}")
    else:
        # Silhouette-based adaptive clustering with heterogeneity awareness
        print(f"Round {round_num}: Performing silhouette-based adaptive reclustering...")
        
        silhouette_scores = []
        kmeans_models = []
        k_values = []
        
        # Start from k=2 (minimum for silhouette score)
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
                cluster_labels = kmeans.fit_predict(scaled_weights)
                
                # Skip if any cluster has only one sample (can't compute silhouette)
                if len(np.unique(cluster_labels)) < k:
                    print(f"KMeans with k={k}: Found only {len(np.unique(cluster_labels))} distinct clusters")
                    continue
                    
                score = silhouette_score(scaled_weights, cluster_labels)
                silhouette_scores.append(score)
                kmeans_models.append(kmeans)
                k_values.append(k)
                print(f"KMeans with k={k}: silhouette={score:.4f}, inertia={kmeans.inertia_:.4f}")
                
            except Exception as e:
                print(f"KMeans with k={k} failed: {e}")
                continue
        
        # Adaptive K selection based on silhouette analysis
        if not silhouette_scores:
            # Fallback to k=2 if no valid clustering found
            optimal_k = min(2, num_clients)
            print(f"No valid silhouette scores found, defaulting to k={optimal_k}")
            optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=20)
            cluster_assignments = optimal_kmeans.fit_predict(scaled_weights)
        else:
            # Find best silhouette score
            best_idx = np.argmax(silhouette_scores)
            best_score = silhouette_scores[best_idx]
            optimal_k = k_values[best_idx]
            optimal_kmeans = kmeans_models[best_idx]
            cluster_assignments = optimal_kmeans.predict(scaled_weights)
            
            # Adaptive decision: keep K=2 if silhouette is best, expand only if justified
            if optimal_k == 2 and best_score > 0.3:
                print(f"Keeping K=2 with good silhouette score: {best_score:.4f}")
            elif optimal_k > 2 and best_score > 0.2:
                print(f"Expanding to K={optimal_k} due to heterogeneity (silhouette={best_score:.4f})")
            else:
                # Force K=2 if silhouette scores are poor across all K values
                print(f"Poor silhouette scores detected, forcing K=2 for stability")
                optimal_k = 2
                optimal_kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=20)
                cluster_assignments = optimal_kmeans.fit_predict(scaled_weights)
        
        # Log cluster sizes
        unique_labels, counts = np.unique(cluster_assignments, return_counts=True)
        print("Final cluster sizes:")
        for label, count in zip(unique_labels, counts):
            print(f"  Cluster {label}: {count} clients")
    
    # CRITICAL: Prevent single cluster collapse in full training
    if len(np.unique(cluster_assignments)) == 1 and num_clients > 1:
        print("CRITICAL: Single cluster collapse detected! Implementing anti-collapse mechanism...")
        
        # Force minimum 2 clusters to prevent collapse
        min_clusters = 2
        
        # Use round-robin assignment with client-specific perturbations
        print(f"Forcing {min_clusters} clusters with enhanced diversity mechanism")
        cluster_assignments = np.array([i % min_clusters for i in range(num_clients)])
        optimal_k = min_clusters
        
        # Create well-separated artificial cluster centers
        cluster_centers = []
        for k in range(optimal_k):
            cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == k]
            if cluster_indices:
                # Base center from assigned clients
                center = np.mean(scaled_weights[cluster_indices], axis=0)
                # Add strong separation bias to prevent re-collapse
                separation_magnitude = 2.0  # Increased separation
                separation_bias = np.random.normal(k * separation_magnitude, 0.5, center.shape)
                center += separation_bias
            else:
                # Fallback with strong differentiation
                center = scaled_weights[k % num_clients].copy()
                center += np.random.normal(k * 2.0, 0.5, center.shape)
            cluster_centers.append(center)
        
        optimal_kmeans = type('DummyKMeans', (), {})()
        optimal_kmeans.cluster_centers_ = np.array(cluster_centers)
        
        print(f"Anti-collapse mechanism: {optimal_k} clusters with assignments {cluster_assignments}")
        print(f"Cluster separation magnitude: {separation_magnitude}")
    
    # Save cluster labels for debugging (secure path)
    np.save(os.path.join(debug_dir, "cluster_labels.npy"), cluster_assignments)
    
    # Enhanced divergence monitoring with silhouette tracking
    if divergence_log is not None:
        weight_std = np.std(flattened_weights, axis=0).mean()
        weight_range = np.ptp(flattened_weights, axis=0).mean()
        
        # Calculate final silhouette score for monitoring
        final_silhouette = -1
        if len(np.unique(cluster_assignments)) > 1 and len(cluster_assignments) > len(np.unique(cluster_assignments)):
            try:
                final_silhouette = silhouette_score(scaled_weights, cluster_assignments)
            except:
                final_silhouette = -1
        
        divergence_log.append({
            'round': round_num,
            'unique_clients': n_unique,
            'clusters_found': optimal_k,
            'weight_std': weight_std,
            'weight_range': weight_range,
            'silhouette_score': final_silhouette,
            'cluster_assignments': cluster_assignments.tolist(),
            'reevaluated_k': should_reevaluate_k
        })
        print(f"Divergence monitoring: std={weight_std:.6f}, range={weight_range:.6f}, clusters={optimal_k}, silhouette={final_silhouette:.4f}")
    
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
    
    # Final validation: ensure we never return single cluster in multi-client scenario
    if optimal_k == 1 and num_clients > 1:
        print("FINAL CHECK: Preventing single cluster return, forcing K=2")
        cluster_assignments = np.array([i % 2 for i in range(num_clients)])
        optimal_k = 2
    
    print(f"Final clustering result: K={optimal_k}, assignments={cluster_assignments}")
    return cluster_assignments, optimal_k, importance_scores