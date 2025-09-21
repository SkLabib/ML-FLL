import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, model_name='efficientnet-b0', grayscale=False, 
                 use_fedbn=True, num_clusters=4, client_id=0):
        super(EfficientNetModel, self).__init__()
        
        # FedBN and cluster-specific parameters
        self.use_fedbn = use_fedbn
        self.num_clusters = num_clusters
        self.client_id = client_id
        self.current_cluster = 0  # Will be set by server
        
        # Always add the same conversion layer architecture for compatibility
        # All models will expect 3-channel input (RGB)
        # Grayscale images will be converted to 3-channel in data preprocessing
        self.grayscale = grayscale
        # Remove the conversion layer entirely - handle channel conversion in data preprocessing
        # This ensures all models have identical architectures
        
        # Load pretrained model based on model_name
        if model_name == 'efficientnet-b0':
            self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet-b1':
            self.efficientnet = models.efficientnet_b1(pretrained=pretrained)
        elif model_name == 'efficientnet-b2':
            self.efficientnet = models.efficientnet_b2(pretrained=pretrained)
        elif model_name == 'mobilenet_v2':
            self.efficientnet = models.mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get feature dimensions for cluster-specific heads
        if 'efficientnet' in model_name:
            in_features = self.efficientnet.classifier[1].in_features
            # Remove original classifier for feature extraction
            self.efficientnet.classifier = nn.Identity()
        elif model_name == 'mobilenet_v2':
            in_features = self.efficientnet.classifier[1].in_features
            # Remove original classifier for feature extraction
            self.efficientnet.classifier = nn.Identity()
        
        # Shared feature extractor (all layers except classifier)
        self.feature_extractor = self.efficientnet
        
        # Cluster-specific classification heads for personalization
        self.cluster_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes)
            ) for _ in range(num_clusters)
        ])
        
        # Global shared head as fallback
        self.global_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        
        # FedBN: Store local BN statistics (not shared in federation)
        if use_fedbn:
            self._setup_fedbn()
        
        # Initialize cluster heads after all components are created
        self._initialize_cluster_heads()
    
    def forward(self, x, use_cluster_head=True):
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Use cluster-specific head if available and requested
        if use_cluster_head and hasattr(self, 'current_cluster'):
            cluster_head = self.cluster_heads[self.current_cluster]
            return cluster_head(features)
        else:
            # Use global head as fallback
            return self.global_head(features)
    
    def _setup_fedbn(self):
        """
        Setup FedBN by identifying and marking BatchNorm layers as local-only.
        """
        self.local_bn_layers = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.local_bn_layers.append(name)
        print(f"FedBN: Found {len(self.local_bn_layers)} BatchNorm layers to keep local")
        print(f"Local BN layers: {self.local_bn_layers[:5]}...")  # Show first 5
    
    def set_cluster(self, cluster_id):
        """
        Set the current cluster for this client.
        
        Args:
            cluster_id (int): Cluster assignment for this client.
        """
        if cluster_id < self.num_clusters:
            self.current_cluster = cluster_id
            print(f"Client {self.client_id}: Set to cluster {cluster_id}")
        else:
            print(f"Warning: Invalid cluster_id {cluster_id}, using cluster 0")
            self.current_cluster = 0
    
    def _initialize_cluster_heads(self):
        """
        Initialize cluster heads with different random initializations for diversity.
        """
        for i, head in enumerate(self.cluster_heads):
            # Use different random seed for each cluster head
            torch.manual_seed(42 + i)
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        # Initialize global head
        torch.manual_seed(42)
        for layer in self.global_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def get_weights(self, include_bn=None):
        """
        Get model weights as a list of tensors.
        
        Args:
            include_bn (bool): Whether to include BatchNorm parameters (for FedBN).
                              If None, uses self.use_fedbn to decide.
        """
        if include_bn is None:
            include_bn = not self.use_fedbn  # Exclude BN if using FedBN
        
        if include_bn or not self.use_fedbn:
            # Include all parameters
            return [param.data.clone() for param in self.parameters()]
        else:
            # Exclude BatchNorm parameters (FedBN)
            weights = []
            for name, param in self.named_parameters():
                is_bn_param = any(bn_layer in name for bn_layer in self.local_bn_layers)
                if not is_bn_param:
                    weights.append(param.data.clone())
            return weights
    
    def get_shared_weights(self):
        """
        Get only shared feature extractor weights (excluding cluster heads).
        """
        shared_weights = []
        for name, param in self.named_parameters():
            # Include feature extractor and global head, exclude cluster heads
            if not name.startswith('cluster_heads.'):
                is_bn_param = any(bn_layer in name for bn_layer in self.local_bn_layers) if self.use_fedbn else False
                if not is_bn_param:  # Exclude BN if using FedBN
                    shared_weights.append(param.data.clone())
        return shared_weights
    
    def get_cluster_head_weights(self, cluster_id=None):
        """
        Get weights for a specific cluster head.
        
        Args:
            cluster_id (int): Cluster ID. If None, uses current cluster.
        """
        if cluster_id is None:
            cluster_id = self.current_cluster
        
        if cluster_id < len(self.cluster_heads):
            return [param.data.clone() for param in self.cluster_heads[cluster_id].parameters()]
        else:
            return [param.data.clone() for param in self.global_head.parameters()]
    
    def set_weights(self, weights, include_bn=None):
        """
        Set model weights from a list of tensors.
        
        Args:
            weights: List of weight tensors.
            include_bn (bool): Whether weights include BatchNorm parameters.
        """
        if include_bn is None:
            include_bn = not self.use_fedbn
        
        # Validate weight count
        expected_params = list(self.parameters()) if include_bn or not self.use_fedbn else [
            param for name, param in self.named_parameters() 
            if not any(bn_layer in name for bn_layer in self.local_bn_layers)
        ]
        
        if len(weights) != len(expected_params):
            raise ValueError(f"Weight count mismatch: expected {len(expected_params)}, got {len(weights)}")
        
        for param, weight in zip(expected_params, weights):
            param.data = weight.to(param.device).clone()
    
    def set_shared_weights(self, weights):
        """
        Set only shared feature extractor weights (excluding cluster heads).
        
        Args:
            weights: List of shared weight tensors.
        """
        shared_params = []
        for name, param in self.named_parameters():
            if not name.startswith('cluster_heads.'):
                is_bn_param = any(bn_layer in name for bn_layer in self.local_bn_layers) if self.use_fedbn else False
                if not is_bn_param:
                    shared_params.append(param)
        
        if len(weights) != len(shared_params):
            raise ValueError(f"Shared weight count mismatch: expected {len(shared_params)}, got {len(weights)}")
        
        for param, weight in zip(shared_params, weights):
            param.data = weight.to(param.device).clone()
    
    def set_cluster_head_weights(self, weights, cluster_id=None):
        """
        Set weights for a specific cluster head.
        
        Args:
            weights: List of weight tensors for the cluster head.
            cluster_id (int): Cluster ID. If None, uses current cluster.
        """
        if cluster_id is None:
            cluster_id = self.current_cluster
        
        if cluster_id < len(self.cluster_heads):
            head_params = list(self.cluster_heads[cluster_id].parameters())
        else:
            head_params = list(self.global_head.parameters())
        
        if len(weights) != len(head_params):
            raise ValueError(f"Cluster head weight count mismatch: expected {len(head_params)}, got {len(weights)}")
        
        for param, weight in zip(head_params, weights):
            param.data = weight.to(param.device).clone()
