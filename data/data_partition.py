import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from PIL import Image
from torchvision import transforms

def class_partition(dataset, num_clients, classes_per_client, batch_size=32, seed=42):
    """
    Partition dataset by class to create heterogeneous data distribution.
    
    Args:
        dataset: The full dataset to partition.
        num_clients (int): Number of clients to create partitions for.
        classes_per_client (int): Number of classes each client should have.
        batch_size (int): Batch size for the data loaders.
        seed (int): Random seed for reproducibility.
        
    Returns:
        client_loaders: List of DataLoader objects for each client.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get all labels
    all_labels = []
    for _, label in dataset:
        all_labels.append(label)
    all_labels = np.array(all_labels)
    
    # Get unique classes
    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)
    
    # Ensure classes_per_client is valid
    if classes_per_client > num_classes:
        classes_per_client = num_classes
        print(f"Warning: classes_per_client reduced to {num_classes} (total number of classes)")
    
    # Assign classes to clients
    client_class_indices = []
    for i in range(num_clients):
        # Select random classes for this client
        client_classes = np.random.choice(unique_classes, classes_per_client, replace=False)
        client_class_indices.append(client_classes)
    
    # Define a custom collate function to handle PIL Images
    def custom_collate_fn(batch):
        images = []
        labels = []
        
        for image, label in batch:
            # Convert PIL Image to tensor if it's not already
            if isinstance(image, Image.Image):
                image = transforms.ToTensor()(image)
            images.append(image)
            
            # Ensure label is a scalar (flatten if needed)
            if isinstance(label, np.ndarray) and label.ndim > 0:
                label = label.item() if label.size == 1 else label.flatten()[0]
            elif isinstance(label, torch.Tensor) and label.dim() > 0:
                label = label.item() if label.numel() == 1 else label.flatten()[0]
            
            labels.append(label)
        
        # Stack images and labels
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return images, labels
    
    # Create client datasets
    client_loaders = []
    for client_classes in client_class_indices:
        # Get indices of samples belonging to the client's classes
        client_indices = []
        for cls in client_classes:
            indices = np.where(all_labels == cls)[0]
            client_indices.extend(indices)
        
        # Create client dataset and dataloader
        client_dataset = Subset(dataset, client_indices)
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        client_loaders.append(client_loader)
    
    return client_loaders

def dirichlet_partition(dataset, num_clients, alpha, batch_size=32, seed=42, min_samples_per_client=100, min_samples_per_class=10, min_classes_per_client=3):
    """
    Partition dataset using Dirichlet distribution to create non-IID data with balanced distribution.
    
    Args:
        dataset: The full dataset to partition.
        num_clients (int): Number of clients to create partitions for.
        alpha (float): Concentration parameter for Dirichlet distribution.
                      Lower alpha means more heterogeneity.
        batch_size (int): Batch size for the data loaders.
        seed (int): Random seed for reproducibility.
        min_samples_per_client (int): Minimum number of samples each client should have.
        min_samples_per_class (int): Minimum number of samples per class for each client.
        min_classes_per_client (int): Minimum number of classes each client should have.
        
    Returns:
        client_loaders: List of DataLoader objects for each client.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get all labels
    all_labels = []
    for _, label in dataset:
        all_labels.append(label)
    all_labels = np.array(all_labels)
    
    # Get unique classes
    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)
    
    # Initialize client indices and class counts
    client_indices = [[] for _ in range(num_clients)]
    client_class_counts = np.zeros((num_clients, num_classes), dtype=int)
    
    # First pass: distribute samples according to stratified Dirichlet distribution
    for k in range(num_classes):
        # Get indices of samples from this class
        idx_k = np.where(all_labels == k)[0]
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Calculate number of samples per client
        proportions = proportions / proportions.sum()
        num_samples_per_client = (proportions * len(idx_k)).astype(int)
        
        # Ensure the sum matches the total number of samples for this class
        remaining = len(idx_k) - num_samples_per_client.sum()
        if remaining > 0:
            # Distribute remaining samples to clients with the lowest counts
            for _ in range(remaining):
                idx = np.argmin(num_samples_per_client)
                num_samples_per_client[idx] += 1
        
        # Shuffle indices for this class
        np.random.shuffle(idx_k)
        
        # Assign samples to clients
        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + num_samples_per_client[i]
            if start_idx < end_idx:  # Only add if there are samples to add
                client_indices[i].extend(idx_k[start_idx:end_idx])
                client_class_counts[i, k] = end_idx - start_idx
            start_idx = end_idx
    
    # Second pass: ensure minimum samples per client and minimum classes per client
    total_samples = len(all_labels)
    min_client_size = max(min_samples_per_client, int(0.05 * total_samples))  # At least 5% of dataset
    
    # Check which clients have fewer than min_client_size samples
    client_sizes = np.array([len(indices) for indices in client_indices])
    small_clients = np.where(client_sizes < min_client_size)[0]
    
    if len(small_clients) > 0:
        print(f"Redistributing samples to {len(small_clients)} clients with fewer than {min_client_size} samples")
        
        # Identify clients with more than min_client_size samples
        large_clients = np.where(client_sizes > min_client_size)[0]
        
        for small_client in small_clients:
            samples_needed = min_client_size - client_sizes[small_client]
            
            # Take samples from larger clients
            for large_client in large_clients:
                # Don't take too many from any one client
                samples_to_take = min(samples_needed, int((client_sizes[large_client] - min_client_size) * 0.5))
                if samples_to_take <= 0:
                    continue
                
                # Select random samples from the large client
                large_client_indices = client_indices[large_client]
                np.random.shuffle(large_client_indices)
                samples_to_move = large_client_indices[:samples_to_take]
                
                # Move samples to small client
                client_indices[small_client].extend(samples_to_move)
                client_indices[large_client] = large_client_indices[samples_to_take:]
                
                # Update class counts
                for idx in samples_to_move:
                    label = all_labels[idx]
                    client_class_counts[small_client, label] += 1
                    client_class_counts[large_client, label] -= 1
                
                # Update client sizes
                client_sizes[small_client] += samples_to_take
                client_sizes[large_client] -= samples_to_take
                samples_needed -= samples_to_take
                
                if samples_needed <= 0:
                    break
    
    # Third pass: ensure minimum classes per client with minimum samples per class
    for i in range(num_clients):
        classes_with_min_samples = np.sum(client_class_counts[i] >= min_samples_per_class)
        
        if classes_with_min_samples < min_classes_per_client:
            print(f"Client {i} has only {classes_with_min_samples} classes with at least {min_samples_per_class} samples")
            
            # Find classes that need more samples
            classes_to_augment = np.where((client_class_counts[i] < min_samples_per_class) & 
                                         (client_class_counts[i] > 0))[0]
            
            for cls in classes_to_augment:
                samples_needed = min_samples_per_class - client_class_counts[i, cls]
                
                # Find clients with excess samples of this class
                clients_with_excess = np.where(client_class_counts[:, cls] > min_samples_per_class + samples_needed)[0]
                
                if len(clients_with_excess) > 0:
                    donor_client = clients_with_excess[0]
                    
                    # Find samples of this class from the donor client
                    donor_indices = client_indices[donor_client]
                    cls_indices = [idx for idx in donor_indices if all_labels[idx] == cls]
                    np.random.shuffle(cls_indices)
                    
                    # Move samples to the recipient client
                    samples_to_move = cls_indices[:samples_needed]
                    client_indices[i].extend(samples_to_move)
                    client_indices[donor_client] = [idx for idx in donor_indices if idx not in samples_to_move]
                    
                    # Update class counts
                    client_class_counts[i, cls] += samples_needed
                    client_class_counts[donor_client, cls] -= samples_needed
    
    # Final check: ensure no client has less than 5% of the dataset
    client_sizes = np.array([len(indices) for indices in client_indices])
    min_size_threshold = int(0.05 * total_samples)
    small_clients = np.where(client_sizes < min_size_threshold)[0]
    
    if len(small_clients) > 0:
        print(f"Warning: {len(small_clients)} clients still have fewer than 5% of the dataset")
        for small_client in small_clients:
            print(f"Client {small_client} has {client_sizes[small_client]} samples ({client_sizes[small_client]/total_samples:.1%} of dataset)")
    
    # Print distribution statistics
    client_sizes = np.array([len(indices) for indices in client_indices])
    print(f"Client data distribution: min={client_sizes.min()} samples ({client_sizes.min()/total_samples:.1%}), "
          f"max={client_sizes.max()} samples ({client_sizes.max()/total_samples:.1%}), "
          f"mean={client_sizes.mean():.1f} samples ({client_sizes.mean()/total_samples:.1%})")
    
    # Print class distribution per client
    for i in range(num_clients):
        classes_with_samples = np.sum(client_class_counts[i] > 0)
        classes_with_min_samples = np.sum(client_class_counts[i] >= min_samples_per_class)
        print(f"Client {i}: {classes_with_samples} classes total, {classes_with_min_samples} classes with ≥{min_samples_per_class} samples")
    
    # Define a custom collate function to handle PIL Images
    def custom_collate_fn(batch):
        images = []
        labels = []
        
        for image, label in batch:
            # Convert PIL Image to tensor if it's not already
            if isinstance(image, Image.Image):
                image = transforms.ToTensor()(image)
            images.append(image)
            
            # Ensure label is a scalar (flatten if needed)
            if isinstance(label, np.ndarray) and label.ndim > 0:
                label = label.item() if label.size == 1 else label.flatten()[0]
            elif isinstance(label, torch.Tensor) and label.dim() > 0:
                label = label.item() if label.numel() == 1 else label.flatten()[0]
            
            labels.append(label)
        
        # Stack images and labels
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return images, labels
    
    # Create client datasets and dataloaders
    client_loaders = []
    for indices in client_indices:
        client_dataset = Subset(dataset, indices)
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        client_loaders.append(client_loader)
    
    return client_loaders