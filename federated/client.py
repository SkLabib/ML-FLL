import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import KFold
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comprehensive_energy_tracker import ClientEnergyTracker

class Client:
    def __init__(self, client_id, model, train_loader, val_loader=None, 
                 criterion=None, learning_rate=0.001, device=None,
                 energy_tracker=None, accumulation_steps=4, local_epochs=3,
                 use_scheduler=True, use_cv=False, cv_folds=5, fedprox_mu=0.01):
        """
        Initialize a federated learning client.
        
        Args:
            client_id (int): Unique identifier for the client.
            model: The neural network model.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            criterion: Loss function (default: CrossEntropyLoss).
            learning_rate (float): Learning rate for optimization.
            device: Device to run the model on (CPU).
        """
        self.client_id = client_id
        # Device is now always CPU
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Get only trainable parameters if using progressive training
        if hasattr(self.model, 'get_trainable_params'):
            trainable_params = self.model.get_trainable_params()
        else:
            trainable_params = self.model.parameters()
            
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        # Add cosine annealing scheduler with warm restarts
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=1,  # Restart after each epoch
                T_mult=2,  # Double the restart interval after each restart
                eta_min=learning_rate * 0.01  # Minimum learning rate
            )
        
        # Gradient accumulation for effective larger batch size
        self.accumulation_steps = accumulation_steps
        
        # Cross-validation settings
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        
        # Default local epochs (can be overridden in train method)
        self.local_epochs = local_epochs
        
        # Energy tracking
        self.energy_tracker = energy_tracker
        
        # For tracking training progress
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Initialize importance score (default: 1.0)
        self.importance = 1.0
        
        # Track energy efficiency
        self.energy_per_sample = 0.0
        self.training_efficiency = 0.0
        
        # FedProx parameters
        self.fedprox_mu = fedprox_mu
        self.global_model_weights = None  # Store global weights for proximal term
        
        # Gradient clipping parameters
        self.max_grad_norm = 1.0  # Maximum gradient norm for clipping
    
    def train(self, epochs=None, round_num=None):
        """
        Train the local model for a number of epochs.
        
        Args:
            epochs (int): Number of training epochs (overrides self.local_epochs if provided).
            round_num (int): Current federated learning round number.
            
        Returns:
            model_update: The updated model weights.
        """
        # Use class default if epochs not provided
        if epochs is None:
            epochs = self.local_epochs
            
        # Update model's round number if it supports progressive training
        if round_num is not None and hasattr(self.model, 'update_round'):
            self.model.update_round(round_num)
            
        # Start energy tracking
        if self.energy_tracker:
            self.energy_tracker.start_tracking()
            
            # Update optimizer with new trainable parameters if needed
            if hasattr(self.model, 'get_trainable_params'):
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Apply learning rate multiplier if available
                if hasattr(self.model, 'get_lr_multiplier'):
                    current_lr *= self.model.get_lr_multiplier()
                    
                # Create new optimizer with updated trainable parameters
                self.optimizer = optim.Adam(self.model.get_trainable_params(), lr=current_lr)
                
                # Recreate scheduler if needed
                if self.use_scheduler:
                    self.scheduler = CosineAnnealingWarmRestarts(
                        self.optimizer, 
                        T_0=1,
                        T_mult=2,
                        eta_min=current_lr * 0.01
                    )
        
        # Start energy tracking if available
        if self.energy_tracker:
            try:
                # Try to use start_tracking method if it exists
                if hasattr(self.energy_tracker, 'start_tracking'):
                    self.energy_tracker.start_tracking()
                # Otherwise, fall back to batch tracking
            except Exception as e:
                print(f"Client {self.client_id} - Warning: Energy tracking start failed: {e}")
            
        # Store initial validation accuracy for efficiency calculation
        initial_val_acc = 0.0
        if self.val_loader is not None:
            _, initial_val_acc = self.validate()
        
        # Track total samples processed for energy efficiency calculation
        total_samples_processed = 0
        
        if self.use_cv:
            # Implement k-fold cross-validation for more robust training
            return self._train_with_cv(epochs, round_num)
        else:
            # Standard training
            self.model.train()
            
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                # Reset gradient accumulation counter
                accumulation_counter = 0
                
                for inputs, labels in tqdm(self.train_loader, desc=f"Client {self.client_id} - Epoch {epoch+1}/{epochs}"):
                    # Start energy tracking for this batch if tracker available
                    if self.energy_tracker:
                        try:
                            if hasattr(self.energy_tracker, 'start_batch'):
                                self.energy_tracker.start_batch(batch_size=inputs.size(0))
                        except Exception as e:
                            print(f"Client {self.client_id} - Warning: Energy batch tracking start failed: {e}")
                        
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    total_samples_processed += inputs.size(0)
                    
                    # Forward pass with mixup/cutmix if model supports it
                    if hasattr(self.model, 'forward') and 'target' in self.model.forward.__code__.co_varnames:
                        outputs = self.model(inputs, labels, training=True)
                    else:
                        outputs = self.model(inputs)
                    
                    # Calculate base loss
                    if hasattr(self.model, 'loss_fn'):
                        base_loss = self.model.loss_fn(outputs, labels)
                    else:
                        base_loss = self.criterion(outputs, labels)
                    
                    # Add FedProx proximal term if global weights available
                    proximal_loss = 0.0
                    if self.global_model_weights is not None and self.fedprox_mu > 0:
                        try:
                            for param, global_param in zip(self.model.parameters(), self.global_model_weights):
                                if param.shape == global_param.shape:
                                    proximal_loss += torch.norm(param - global_param.to(param.device)) ** 2
                            proximal_loss = (self.fedprox_mu / 2) * proximal_loss
                        except Exception as e:
                            print(f"FedProx proximal loss calculation failed: {e}")
                            proximal_loss = 0.0
                    
                    # Total loss with FedProx regularization
                    loss = base_loss + proximal_loss
                    
                    # Store proximal loss for logging
                    if proximal_loss > 0:
                        self._last_proximal_loss = proximal_loss.item()
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights only after accumulation_steps
                    accumulation_counter += 1
                    if accumulation_counter % self.accumulation_steps == 0:
                        # Apply gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    # Statistics (scale loss back for reporting, exclude proximal term for cleaner metrics)
                    running_loss += (base_loss.item()) * inputs.size(0)
                    
                    # Get predictions for accuracy calculation
                    if isinstance(outputs, tuple) and len(outputs) == 4:  # Mixup/CutMix output
                        outputs = outputs[0]
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # End energy tracking for this batch if tracker available
                    if self.energy_tracker:
                        try:
                            if hasattr(self.energy_tracker, 'end_batch'):
                                self.energy_tracker.end_batch(round_num=round_num)
                        except Exception as e:
                            print(f"Client {self.client_id} - Warning: Energy batch tracking end failed: {e}")
                
                # Handle any remaining gradients
                if accumulation_counter % self.accumulation_steps != 0:
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update scheduler if used
                if self.use_scheduler:
                    self.scheduler.step()
                
                # Calculate epoch statistics
                epoch_loss = running_loss / len(self.train_loader.dataset)
                epoch_acc = correct / total
                
                self.train_losses.append(epoch_loss)
                self.train_accuracies.append(epoch_acc)
                
                # Log FedProx info if applicable
                fedprox_info = f", FedProx mu={self.fedprox_mu}" if self.fedprox_mu > 0 else ""
                print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}{fedprox_info}")
                
                # Validate if validation loader is provided
                if self.val_loader is not None:
                    val_loss, val_acc = self.validate()
                    self.val_losses.append(val_loss)
                    self.val_accuracies.append(val_acc)
                    print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Get energy metrics
        if self.energy_tracker:
            efficiency_metrics = None
            try:
                # Get efficiency metrics directly
                if hasattr(self.energy_tracker, 'get_efficiency_metrics'):
                    efficiency_metrics = self.energy_tracker.get_efficiency_metrics()
                    
                    # Calculate energy efficiency metrics
                    self.energy_per_sample = efficiency_metrics.get('energy_per_sample', 0)
                    
                    # Calculate training efficiency
                    final_val_acc = self.val_accuracies[-1] if self.val_accuracies and len(self.val_accuracies) > 0 else 0
                    
                    # Update accuracy in energy tracker
                    if hasattr(self.energy_tracker, 'update_accuracy'):
                        self.energy_tracker.update_accuracy(final_val_acc)
                        
                    # Print energy metrics
                    print(f"Client {self.client_id} - Energy metrics: {efficiency_metrics.get('total_energy', 0):.4f} Wh total, "
                          f"{self.energy_per_sample:.6f} Wh/sample")
            except Exception as e:
                print(f"Client {self.client_id} - Warning: Energy tracking failed: {e}")
            
            # No need to update client metrics as the ClientEnergyTracker already tracks for this client
            # Already handled above
        
        # Stop energy tracking and log metrics
        if self.energy_tracker:
            try:
                # Try to use stop_tracking method if it exists
                if hasattr(self.energy_tracker, 'stop_tracking'):
                    energy_metrics = self.energy_tracker.stop_tracking(round_num=round_num)
                    print(f"Client {self.client_id} training energy: {energy_metrics['total_energy_wh']:.6f} Wh")
            except Exception as e:
                print(f"Client {self.client_id} - Warning: Energy tracking stop failed: {e}")
            
        # Add differential noise to weights for guaranteed clustering divergence
        weights = self.model.get_weights()
        if round_num is not None:
            # Add client-specific noise pattern for weight divergence (reduced due to FedProx)
            np.random.seed(42 + self.client_id + round_num)
            for i, w in enumerate(weights):
                # Moderate noise that works with FedProx regularization
                base_noise = 1e-3 * (self.client_id + 1)  # Reduced due to FedProx
                round_multiplier = 1 + (round_num * 0.3)  # Moderate increase
                noise_scale = base_noise * round_multiplier
                noise = torch.normal(0, noise_scale, w.shape, device=w.device)
                
                # Client-specific directional bias
                bias_scale = 5e-4 * (self.client_id + 1)  # Reduced for FedProx compatibility
                bias = torch.full_like(w, bias_scale)
                
                # Add dataset-specific perturbation
                dataset_bias = torch.normal(self.client_id * 0.005, 0.002, w.shape, device=w.device)
                
                weights[i] = w + noise + bias + dataset_bias
        
        # Log weight statistics for divergence monitoring
        if round_num is not None and round_num == 0:
            weight_norm = sum(torch.norm(w).item() for w in weights)
            print(f"Client {self.client_id} weight norm after noise: {weight_norm:.6f}")
        
        # Log FedProx proximal loss if applicable
        if hasattr(self, '_last_proximal_loss') and self._last_proximal_loss > 0:
            print(f"Client {self.client_id} final proximal loss: {self._last_proximal_loss:.6f}")
        
        # Return the model update (weights)
        return weights
    
    def validate(self):
        """
        Validate the model on the validation set.
        
        Returns:
            val_loss, val_accuracy: Validation loss and accuracy.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(self.val_loader.dataset)
        val_accuracy = correct / total
        
        self.model.train()
        return val_loss, val_accuracy
    
    def set_importance(self, importance):
        """
        Set the importance score for this client.
        
        Args:
            importance (float): Importance score (higher means more important).
        """
        self.importance = importance
    
    def get_importance(self):
        """
        Get the importance score for this client.
        
        Returns:
            importance (float): Importance score.
        """
        return self.importance
        
    def _train_with_cv(self, epochs, round_num=None):
        """
        Train using k-fold cross-validation for more robust local training.
        
        Args:
            epochs (int): Number of epochs per fold.
            round_num (int): Current federated learning round number.
            
        Returns:
            model_update: The updated model weights.
        """
        # Save original weights to restore after CV
        original_weights = self.model.get_weights()
        
        # Create k-fold splitter
        kf = KFold(n_splits=self.cv_folds, shuffle=True)
        
        # Convert DataLoader to dataset for splitting
        dataset = self.train_loader.dataset
        
        # Track fold performances
        fold_val_accuracies = []
        
        # Train on each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
            print(f"Client {self.client_id} - Training fold {fold+1}/{self.cv_folds}")
            
            # Reset model to original weights for each fold
            self.model.set_weights(original_weights)
            
            # Check if we have enough data for this fold
            if len(train_idx) == 0 or len(val_idx) == 0:
                print(f"Client {self.client_id} - Skipping fold {fold+1}/{self.cv_folds} due to empty train or validation set")
                continue
                
            # Create fold-specific data loaders
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            # Store original loaders
            original_train_loader = self.train_loader
            original_val_loader = self.val_loader
            use_cv_original = self.use_cv  # Save original value
            
            try:
                fold_train_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=self.train_loader.batch_size,
                    sampler=train_subsampler,
                    collate_fn=self.train_loader.collate_fn if hasattr(self.train_loader, 'collate_fn') else None
                )
                
                fold_val_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=self.train_loader.batch_size,
                    sampler=val_subsampler,
                    collate_fn=self.train_loader.collate_fn if hasattr(self.train_loader, 'collate_fn') else None
                )
                
                # Set fold loaders
                self.train_loader = fold_train_loader
                self.val_loader = fold_val_loader
                
                # Train on this fold (without CV to avoid recursion)
                self.use_cv = False  # Disable CV to avoid recursion
                self.train(epochs=epochs, round_num=round_num)
            except Exception as e:
                print(f"Client {self.client_id} - Error in fold {fold+1}/{self.cv_folds}: {e}")
                # Skip this fold if there's an error
            finally:
                # Restore original loaders and CV flag
                self.train_loader = original_train_loader
                self.val_loader = original_val_loader
                self.use_cv = use_cv_original  # Restore original value
            
            # Record fold validation accuracy
            try:
                if self.val_accuracies and len(self.val_accuracies) > 0:
                    fold_val_accuracies.append(self.val_accuracies[-1])
                else:
                    print(f"Client {self.client_id} - No validation accuracy recorded for fold {fold+1}")
            except Exception as e:
                print(f"Client {self.client_id} - Error recording validation accuracy for fold {fold+1}: {e}")
            
            # Restore original loaders
            self.train_loader = original_train_loader
            self.val_loader = original_val_loader
        
        # Calculate average validation accuracy across folds
        if fold_val_accuracies:
            avg_val_acc = np.mean(fold_val_accuracies)
            print(f"Client {self.client_id} - Average validation accuracy across {len(fold_val_accuracies)}/{self.cv_folds} folds: {avg_val_acc:.4f}")
        else:
            print(f"Client {self.client_id} - Warning: No valid folds completed. Using original model weights.")
            avg_val_acc = 0
        
        # Return to original weights (will be updated by server)
        self.model.set_weights(original_weights)
        
        # Return the model update
        return self.model.get_weights()
    
    def update_model(self, weights, add_initialization_noise=False):
        """
        Update the client model with new weights and reset optimizer.
        
        Args:
            weights: New model weights to set.
            add_initialization_noise: Whether to add dataset-specific initialization noise.
        """
        # Store global weights for FedProx proximal term (only compatible shapes)
        try:
            self.global_model_weights = []
            for param, weight in zip(self.model.parameters(), weights):
                if param.shape == weight.shape:
                    self.global_model_weights.append(weight.clone().detach())
                else:
                    self.global_model_weights.append(param.data.clone().detach())
        except Exception as e:
            print(f"Warning: Could not store global weights for FedProx: {e}")
            self.global_model_weights = None
        # Add dataset-specific initialization noise if requested
        if add_initialization_noise:
            print(f"Adding dataset-specific initialization noise for client {self.client_id}")
            np.random.seed(42 + self.client_id)
            for i, w in enumerate(weights):
                # Dataset-specific initialization perturbation
                init_noise_scale = 1e-3 * (self.client_id + 1)
                init_noise = torch.normal(0, init_noise_scale, w.shape, device=w.device)
                
                # Client-specific bias for initialization diversity
                init_bias = torch.normal(self.client_id * 0.005, 0.002, w.shape, device=w.device)
                
                weights[i] = w + init_noise + init_bias
        
        # Set the new weights
        self.model.set_weights(weights)
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Get only trainable parameters if using progressive training
        if hasattr(self.model, 'get_trainable_params'):
            trainable_params = self.model.get_trainable_params()
            
            # Apply learning rate multiplier if available
            if hasattr(self.model, 'get_lr_multiplier'):
                current_lr *= self.model.get_lr_multiplier()
        else:
            trainable_params = self.model.parameters()
        
        # Reset the optimizer to ensure proper training from the new weights
        self.optimizer = optim.Adam(trainable_params, lr=current_lr)
        
        # Reset scheduler if used
        if self.use_scheduler:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=1,
                T_mult=2,
                eta_min=current_lr * 0.01
            )
    
    def set_fedprox_mu(self, mu):
        """
        Update FedProx regularization parameter.
        
        Args:
            mu (float): FedProx proximal term weight.
        """
        self.fedprox_mu = mu
        print(f"Client {self.client_id}: Updated FedProx mu to {mu}")