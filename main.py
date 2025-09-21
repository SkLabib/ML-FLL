import os
import torch
import numpy as np
import random
import argparse
import pandas as pd
import json
from datetime import datetime

# Import custom modules
from data.data_loader import get_data_loaders, get_all_dataset_loaders
from data.data_partition import class_partition, dirichlet_partition
from models.model_factory import create_model  # Use the model factory instead
from federated.client import Client
from federated.server import Server
from utils.metrics import compute_metrics, plot_confusion_matrix, print_metrics_report
from utils.visualization import plot_training_curves, plot_global_performance, plot_client_importance
from utils.comprehensive_energy_tracker import EnergyTracker, ClientEnergyTracker

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device - Modified to use CPU only
    device = torch.device("cpu")
    # Verification step
    print(f"Using device: CPU")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"federated_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define datasets for each client with their number of classes
    # Try with HAM10000, fallback to MedMNIST only if HAM10000 unavailable
    try:
        # Test if HAM10000 is available
        from data.data_loader import get_data_loaders
        test_loader = get_data_loaders(args.data_dir, "ham10000", batch_size=1, apply_smote=False)
        datasets = ["octmnist", "tissuemnist", "pathmnist", "ham10000"]
        print("Using all 4 datasets including HAM10000")
    except:
        datasets = ["octmnist", "tissuemnist", "pathmnist", "octmnist"]  # Use octmnist twice as fallback
        print("HAM10000 not available, using MedMNIST datasets only")
    
    # Define number of classes for each dataset
    dataset_num_classes = {
        "octmnist": 4,
        "tissuemnist": 8,
        "pathmnist": 9,
        "ham10000": 7
    }
    
    # Initialize global model using the factory with FedBN and cluster-specific heads
    # Use the maximum number of classes for the global model to accommodate all datasets
    max_num_classes = max(dataset_num_classes.values())  # 9 classes (pathmnist)
    # Enable FedBN and cluster-specific heads for improved accuracy
    global_model = create_model(
        model_type=args.model_type, 
        model_name=args.model_name, 
        num_classes=max_num_classes, 
        pretrained=args.pretrained, 
        grayscale=False,  # All inputs converted to RGB in preprocessing
        use_fedbn=True,   # Enable FedBN for non-IID data
        num_clusters=4,   # Support cluster-specific heads
        client_id=0       # Global model (server)
    )
    
    # Initialize energy tracker
    energy_tracker = EnergyTracker()
    
    # Load test loaders for each dataset for evaluation using the new get_all_dataset_loaders function
    client_loaders, test_loaders = get_all_dataset_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_split=args.test_split,
        val_split=args.val_split,
        seed=args.seed
    )
    
    # Use a combined test loader for the server
    server = Server(
        global_model=global_model, 
        test_loader=test_loaders['ham10000'],  # Use HAM10000 test loader for early stopping
        device=device,
        mu=args.fedprox_mu,
        selection_ratio=args.selection_ratio,
        early_stopping_rounds=args.early_stopping_rounds,
        energy_tracker=energy_tracker
    )
    
    # Initialize clients with different datasets
    clients = []
    for i, dataset_name in enumerate(datasets):
        print(f"Loading {dataset_name} dataset for client {i}...")
        
        # Get client data from the pre-loaded client_loaders
        client_data = client_loaders[dataset_name]
        train_loader = client_data['train']
        val_loader = client_data['val']
        class_weights = client_data['class_weights']
        focal_criterion = client_data.get('focal_criterion', None)  # For HAM10000
        
        # Create a new model instance for each client with FedBN and cluster-specific heads
        # Use max_num_classes for all clients to ensure compatible architectures for aggregation
        # All models now use RGB input (grayscale conversion handled in data preprocessing)
        client_model = create_model(
            model_type=args.model_type, 
            model_name=args.model_name, 
            num_classes=max_num_classes, 
            pretrained=args.pretrained, 
            grayscale=False,
            use_fedbn=True,   # Enable FedBN for better non-IID handling
            num_clusters=4,   # Support cluster-specific heads
            client_id=i       # Client-specific ID
        )
        
        # Create client energy tracker
        client_energy_tracker = ClientEnergyTracker(client_id=i, global_tracker=energy_tracker)
        
        # Create client with FedProx and focal loss support
        client = Client(
            client_id=i,
            model=client_model,
            train_loader=train_loader,  # Use the dataset-specific train loader
            val_loader=val_loader,      # Use the dataset-specific validation loader
            learning_rate=args.learning_rate,
            device=device,
            energy_tracker=client_energy_tracker,
            accumulation_steps=args.accumulation_steps,
            local_epochs=args.local_epochs,
            use_scheduler=args.use_scheduler,
            use_cv=args.use_cv,
            cv_folds=args.cv_folds,
            fedprox_mu=args.fedprox_mu  # FedProx regularization
        )
        
        # Set focal loss criterion for HAM10000 clients
        if focal_criterion is not None:
            client.criterion = focal_criterion
            print(f"Client {i} ({dataset_name}): Using Focal Loss for rare class handling")
        # Initialize client model with global model weights + dataset-specific noise
        client.update_model(global_model.get_weights(), add_initialization_noise=True)
        clients.append(client)
    
    # Initialize results tracking
    results_data = {
        "round": [],
        "global_accuracy": [],
        "octmnist_accuracy": [],
        "tissuemnist_accuracy": [],
        "pathmnist_accuracy": [],
        "ham10000_accuracy": []
    }
    
    # Initialize energy tracking
    energy_data = {
        "round": [],
        "energy_client0": [],
        "energy_client1": [],
        "energy_client2": [],
        "energy_client3": [],
        "energy_global": []
    }
    
    # Federated learning process
    for round_idx in range(args.num_rounds):
        print(f"\n--- Round {round_idx+1}/{args.num_rounds} ---")
        
        # Train local models
        client_weights = []
        client_energy_metrics = []
        
        for client in clients:
            print(f"Training client {client.client_id} on {datasets[client.client_id]}...")
            # Pass round number for progressive training
            weights = client.train(round_num=round_idx)
            client_weights.append(weights)
            
            # Collect energy metrics
            if hasattr(client, 'energy_tracker') and client.energy_tracker is not None:
                metrics = client.energy_tracker.get_efficiency_metrics()
                client_energy_metrics.append(metrics)
                # Store client energy for this round
                if round_idx >= len(energy_data["round"]):
                    energy_data["energy_client" + str(client.client_id)].append(metrics["total_energy"])
        
        # Cluster and aggregate with silhouette-based adaptive clustering
        global_weights, cluster_assignments, importance_scores = server.cluster_and_aggregate(
            client_weights=client_weights,
            client_energy_metrics=client_energy_metrics if client_energy_metrics else None,
            max_k=4,  # Maximum K=4, but adaptive based on silhouette analysis
            random_state=args.seed
        )
        
        # Update client cluster assignments for cluster-specific heads
        for i, client in enumerate(clients):
            cluster_id = cluster_assignments[i]
            client.model.set_cluster(cluster_id)
            print(f"Client {i} assigned to cluster {cluster_id}")
        
        # Update client importance scores
        for i, client in enumerate(clients):
            client.set_importance(importance_scores[i])
        
        # Visualize client importance scores
        plot_client_importance(
            client_ids=[client.client_id for client in clients],
            importance_scores=importance_scores,
            output_dir=output_dir,
        )
        
        # Update the global model with the aggregated weights
        server.global_model.set_weights(global_weights)
        
        # Evaluate global model on each dataset's test set
        results_data["round"].append(round_idx + 1)
        
        # Track global accuracy (average of all datasets)
        global_acc_sum = 0
        global_acc_count = 0
        
        for dataset_name in datasets:
            # Evaluate on this dataset's test loader
            test_loader = test_loaders[dataset_name]
            test_loss, test_acc, test_preds, test_targets = server.evaluate(val_loader=test_loader)
            
            # Store accuracy for this dataset
            results_data[f"{dataset_name}_accuracy"].append(test_acc)
            global_acc_sum += test_acc
            global_acc_count += 1
            
            print(f"Round {round_idx+1} - {dataset_name} Test Accuracy: {test_acc:.4f}")
        
        # Calculate and store global accuracy
        global_acc = global_acc_sum / global_acc_count
        results_data["global_accuracy"].append(global_acc)
        print(f"Round {round_idx+1} - Global Test Accuracy: {global_acc:.4f}")
        
        # Store energy data for this round
        if round_idx >= len(energy_data["round"]):
            energy_data["round"].append(round_idx + 1)
            # Get global energy from the server's energy tracker
            if hasattr(server, 'energy_tracker') and server.energy_tracker is not None:
                global_energy = server.energy_tracker.get_round_energy(round_idx)
                energy_data["energy_global"].append(global_energy)
            else:
                energy_data["energy_global"].append(0.0)
        
        # Save intermediate results to CSV
        pd.DataFrame(results_data).to_csv(os.path.join(output_dir, "accuracy_results.csv"), index=False)
        pd.DataFrame(energy_data).to_csv(os.path.join(output_dir, "energy_results.csv"), index=False)
        
        # Check for early stopping based on global accuracy
        if global_acc >= 0.8 and all(acc >= 0.8 for acc in [results_data[f"{d}_accuracy"][-1] for d in datasets]):
            print("Target accuracy of 80% reached on all datasets! Stopping training.")
            break
        
        # Check for early stopping based on validation performance
        if server.should_early_stop():
            print("Early stopping triggered due to validation performance!")
            break
        
        # Distribute global model to clients and reset optimizers
        for client in clients:
            client.update_model(global_weights)  # No initialization noise for updates
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    
    # Evaluate on each dataset separately
    final_results = {
        "dataset": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    
    for dataset_name in datasets:
        print(f"\nEvaluating on {dataset_name} test set:")
        test_loader = test_loaders[dataset_name]
        test_loss, test_acc, all_preds, all_targets = server.evaluate(val_loader=test_loader)
        
        # Compute detailed metrics
        metrics = compute_metrics(all_targets, all_preds)
        print_metrics_report(metrics, title=f"Final Model Performance on {dataset_name}")
        
        # Store results
        final_results["dataset"].append(dataset_name)
        final_results["accuracy"].append(metrics["accuracy"])
        final_results["precision"].append(metrics["precision"])
        final_results["recall"].append(metrics["recall"])
        final_results["f1_score"].append(metrics["f1_score"])
        
        # Plot confusion matrix for each dataset
        if dataset_name.lower() == 'ham10000':
            class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        else:
            # For MedMNIST datasets, use numeric class names if specific names aren't available
            try:
                from data.data_loader import INFO
                dataset_num_classes = {
                    "octmnist": 4,
                    "tissuemnist": 8,
                    "pathmnist": 9,
                    "ham10000": 7
                }
                num_classes = dataset_num_classes[dataset_name]
                class_names = INFO[dataset_name.lower()]['label']
            except (KeyError, ImportError):
                dataset_num_classes = {
                    "octmnist": 4,
                    "tissuemnist": 8,
                    "pathmnist": 9,
                    "ham10000": 7
                }
                num_classes = dataset_num_classes[dataset_name]
                class_names = [str(i) for i in range(num_classes)]
        
        plot_confusion_matrix(all_targets, all_preds, class_names=class_names, 
                             output_dir=output_dir, filename=f"confusion_matrix_{dataset_name}.png")
    
    # Save final evaluation results
    pd.DataFrame(final_results).to_csv(os.path.join(output_dir, "final_evaluation.csv"), index=False)
    
    # Save final model weights
    model_save_path = os.path.join(output_dir, "final_model_weights.pt")
    torch.save(server.global_model.state_dict(), model_save_path)
    print(f"Final model weights saved to {model_save_path}")
    
    # Generate energy efficiency reports
    energy_tracker.generate_reports()
    energy_tracker.save_round_energy_log(os.path.join(output_dir, "round_energy_log.csv"))
    
    # Plot training curves
    plot_training_curves(clients, output_dir=output_dir)
    
    # Plot global model performance and energy metrics
    plot_global_performance(server, output_dir=output_dir)
    
    # Plot energy efficiency metrics
    from utils.visualization import plot_energy_metrics, plot_client_energy_heatmap, plot_carbon_footprint
    
    plot_energy_metrics(energy_tracker, output_dir=output_dir)
    plot_client_energy_heatmap(clients, output_dir=output_dir)
    plot_carbon_footprint(energy_tracker, output_dir=output_dir)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with Energy Efficiency")
    
    # Dataset parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Proportion of data to use for testing (default: 0.2)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Proportion of training data to use for validation (default: 0.1)")
    
    # Federated learning parameters - Increased for better convergence
    parser.add_argument("--num_rounds", type=int, default=50,
                        help="Number of communication rounds (default: 50)")
    parser.add_argument("--local_epochs", type=int, default=5,
                        help="Number of local training epochs per round (default: 5)")
    parser.add_argument("--selection_ratio", type=float, default=1.0,
                        help="Ratio of clients to select per round (default: 1.0)")
    parser.add_argument("--early_stopping_rounds", type=int, default=8,
                        help="Number of rounds with no improvement to trigger early stopping (default: 8)")
    
    # Training parameters - Reduced LR for stability
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate for local training (default: 0.0005)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps (default: 4)")
    parser.add_argument("--use_scheduler", action="store_true", default=False,
                        help="Use learning rate scheduler (default: False)")
    parser.add_argument("--use_cv", action="store_true", default=False,
                        help="Use cross-validation for local training (default: False)")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    
    # FedProx parameters - Increased for better non-IID handling
    parser.add_argument("--fedprox_mu", type=float, default=0.1,
                        help="FedProx proximal term weight (default: 0.1)")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="efficientnet",
                        choices=["efficientnet", "resnet18", "resnet50", "mobilenet"],
                        help="Model architecture to use (default: efficientnet)")
    parser.add_argument("--model_name", type=str, default="efficientnet-b0",
                        help="Specific model name (e.g., efficientnet-b0, efficientnet-b1, mobilenet_v2)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained model weights (default: True)")
    
    args = parser.parse_args()
    main(args)