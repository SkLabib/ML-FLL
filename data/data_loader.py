import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
import cv2
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, apply_clahe=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            apply_clahe (bool): Whether to apply CLAHE histogram equalization.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.apply_clahe = apply_clahe
        
        # Map diagnosis to numerical labels
        self.diagnosis_mapping = {
            'akiec': 0,  # Actinic Keratoses and Intraepithelial Carcinoma
            'bcc': 1,    # Basal Cell Carcinoma
            'bkl': 2,    # Benign Keratosis-like Lesions
            'df': 3,     # Dermatofibroma
            'mel': 4,    # Melanoma
            'nv': 5,     # Melanocytic Nevi
            'vasc': 6    # Vascular Lesions
        }
        
        self.data_frame['label'] = self.data_frame['dx'].map(self.diagnosis_mapping)
        
        # Set number of classes attribute
        self.num_classes = len(self.diagnosis_mapping)
        
        # Create a CLAHE object for histogram equalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 1] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx]['label']
        
        # Apply CLAHE histogram equalization if enabled
        if self.apply_clahe:
            # Convert PIL image to OpenCV format
            img_array = np.array(image)
            # Convert RGB to LAB color space
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            # Apply CLAHE to L channel
            lab_planes = list(cv2.split(lab))  # Convert tuple to list
            lab_planes[0] = self.clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            # Convert back to RGB
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            # Convert back to PIL image
            image = Image.fromarray(img_array)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class MedMNISTDataset(Dataset):
    def __init__(self, dataset_name, split='train', transform=None):
        """
        Args:
            dataset_name (string): Name of the MedMNIST dataset (e.g., 'tissuemnist', 'pathmnist', 'octmnist').
            split (string): 'train', 'val', or 'test' split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        
        # Load the dataset
        info = INFO[dataset_name]
        self.data_class = getattr(medmnist, info['python_class'])
        
        # Create a custom transform that converts numpy arrays to PIL Images first
        def numpy_to_pil_transform(img):
            if isinstance(img, np.ndarray):
                return Image.fromarray(img)
            return img
            
        # Apply our transform and then the user's transform if provided
        if transform is not None:
            dataset_transform = transforms.Compose([numpy_to_pil_transform, transform])
        else:
            dataset_transform = numpy_to_pil_transform
            
        self.dataset = self.data_class(split=split, transform=dataset_transform, download=True)
        
        # Get class names
        self.class_names = info['label']
        self.num_classes = len(self.class_names)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Ensure label is a scalar (flatten if needed)
        if isinstance(label, np.ndarray) and label.ndim > 0:
            label = label.item() if label.size == 1 else label.flatten()[0]
        elif isinstance(label, torch.Tensor) and label.dim() > 0:
            label = label.item() if label.numel() == 1 else label.flatten()[0]
        
        return img, label

def get_data_loaders(data_dir, dataset_name='ham10000', batch_size=32, test_split=0.2, val_split=0.1, seed=42, apply_smote=True, client_id=None):
    """
    Create train, validation, and test data loaders for the specified dataset.
    
    Args:
        data_dir (string): Directory with the dataset.
        dataset_name (string): Name of the dataset ('ham10000', 'tissuemnist', 'pathmnist', 'octmnist').
        batch_size (int): Batch size for the data loaders.
        test_split (float): Proportion of data to use for testing (only for HAM10000).
        val_split (float): Proportion of training data to use for validation (only for HAM10000).
        seed (int): Random seed for reproducibility.
        apply_smote (bool): Whether to apply SMOTE/ADASYN for balancing rare classes.
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation, and testing.
        class_weights: Class weights for handling class imbalance.
        full_dataset: The dataset object for partitioning.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Define data transforms with reduced augmentation as requested
    # Medical-specific mean/std values (calculated from HAM10000 dataset)
    # These values are more appropriate for skin lesion images than ImageNet values
    med_mean = [0.7630, 0.5456, 0.5700]  # Medical image mean (RGB)
    med_std = [0.1409, 0.1521, 0.1698]   # Medical image std (RGB)
    
    # Grayscale mean/std for MedMNIST datasets
    gray_mean = [0.5]  # Grayscale mean
    gray_std = [0.5]   # Grayscale std
    
    # Check if we're using a MedMNIST dataset (grayscale) or HAM10000 (RGB)
    is_medmnist = dataset_name.lower() in ['octmnist', 'tissuemnist', 'pathmnist']
    
    if is_medmnist:
        # MedMNIST transforms (grayscale converted to RGB)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize(med_mean, med_std)  # Using RGB values for consistency
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize(med_mean, med_std)  # Using RGB values for consistency
        ])
    else:
        # HAM10000 transforms (RGB)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(med_mean, med_std)  # Using RGB values
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(med_mean, med_std)  # Using RGB values
        ])
    
    # Handle different datasets
    if dataset_name.lower() == 'ham10000':
        # Load metadata
        # Check if we're in the data directory or the parent directory
        if os.path.exists(os.path.join(data_dir, 'HAM10000_metadata.csv')):
            metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
            img_dir = os.path.join(data_dir, 'images')
        else:
            # Try the parent directory where HAM10000_dataset folder might be
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            metadata_path = os.path.join(parent_dir, 'HAM10000_dataset', 'HAM10000_metadata.csv')
            img_dir = os.path.join(parent_dir, 'HAM10000_dataset', 'images')
        
        # Read metadata
        metadata = pd.read_csv(metadata_path)
        
        # Split data into train+val and test
        unique_patients = metadata['lesion_id'].unique()
        np.random.shuffle(unique_patients)
        
        test_size = int(len(unique_patients) * test_split)
        test_patients = unique_patients[:test_size]
        train_val_patients = unique_patients[test_size:]
        
        test_idx = metadata[metadata['lesion_id'].isin(test_patients)].index
        train_val_idx = metadata[metadata['lesion_id'].isin(train_val_patients)].index
        
        # Split train+val into train and val
        val_size = int(len(train_val_idx) * val_split)
        val_idx = np.random.choice(train_val_idx, val_size, replace=False)
        train_idx = np.array(list(set(train_val_idx) - set(val_idx)))
        
        # Create datasets with CLAHE histogram equalization
        full_dataset = HAM10000Dataset(metadata_path, img_dir, transform=None, apply_clahe=True)
        
        # Apply SMOTE/ADASYN to balance rare classes before creating train dataset
        if apply_smote:
            print("Applying SMOTE to balance rare classes in HAM10000 dataset...")
            # Extract features and labels for SMOTE
            X_train = []
            y_train = []
            
            # Create a temporary transform for feature extraction
            temp_transform = transforms.Compose([
                transforms.Resize((64, 64)),  # Smaller size for faster processing
                transforms.ToTensor()
            ])
            
            # Extract features from images
            for idx in train_idx:
                img_name = os.path.join(img_dir, metadata.iloc[idx, 1] + '.jpg')
                image = Image.open(img_name).convert('RGB')
                # Apply CLAHE
                img_array = np.array(image)
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                lab_planes = list(cv2.split(lab))  # Convert tuple to list so we can modify it
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab_planes[0] = clahe.apply(lab_planes[0])
                lab = cv2.merge(lab_planes)
                img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                image = Image.fromarray(img_array)
                
                # Transform and flatten for SMOTE
                img_tensor = temp_transform(image)
                img_flat = img_tensor.flatten().numpy()
                X_train.append(img_flat)
                y_train.append(full_dataset.diagnosis_mapping[metadata.iloc[idx]['dx']])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Print class distribution before SMOTE
            print("Class distribution before SMOTE:", Counter(y_train))
            
            # Apply SMOTE for classes with fewer samples (mel, df, vasc)
            try:
                # Use ADASYN with 'auto' strategy for better handling
                adasyn = ADASYN(sampling_strategy='auto', random_state=seed)
                X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
                print("Class distribution after ADASYN:", Counter(y_resampled))
                
                # Create new indices for the balanced dataset
                # Original indices remain the same, new synthetic samples get new indices
                original_count = len(train_idx)
                synthetic_count = len(y_resampled) - original_count
                new_indices = np.concatenate([train_idx, np.array([-i-1 for i in range(synthetic_count)])]) 
                
                # Store synthetic samples for later retrieval
                full_dataset.synthetic_samples = {}
                for i in range(original_count, len(y_resampled)):
                    # Reshape flattened image back to tensor
                    img_flat = X_resampled[i]
                    img_tensor = torch.from_numpy(img_flat).reshape(3, 64, 64)
                    # Resize to original size
                    img_resized = transforms.Resize((224, 224))(img_tensor)
                    full_dataset.synthetic_samples[-i-1+original_count] = (img_resized, y_resampled[i])
                
                # Update train_idx with new indices
                train_idx = new_indices
                
                # Set flag to indicate ADASYN was successful
                adasyn_successful = True
                print(f"ADASYN successfully applied. Added {synthetic_count} synthetic samples.")
                
                # Add method to HAM10000Dataset to handle synthetic samples
                def get_synthetic_sample(self, idx):
                    img_tensor, label = self.synthetic_samples[idx]
                    if self.transform:
                        # Apply remaining transforms (except resize which was already done)
                        transforms_list = self.transform.transforms
                        for t in transforms_list:
                            if not isinstance(t, transforms.Resize):
                                img_tensor = t(img_tensor)
                    return img_tensor, label
                
                # Add the method to the class
                HAM10000Dataset.get_synthetic_sample = get_synthetic_sample
                
                # Override __getitem__ to handle synthetic samples
                original_getitem = HAM10000Dataset.__getitem__
                def new_getitem(self, idx):
                    if idx < 0:  # Synthetic sample
                        return self.get_synthetic_sample(idx)
                    else:  # Original sample
                        return original_getitem(self, idx)
                
                HAM10000Dataset.__getitem__ = new_getitem
                
            except Exception as e:
                print(f"Error applying ADASYN: {e}. Proceeding with original dataset.")
                # Ensure we use the original dataset without synthetic samples
                adasyn_successful = False
                # Make sure synthetic_samples attribute exists but is empty
                full_dataset.synthetic_samples = {}
                
                # Define a dummy method that will be used if ADASYN fails
                def get_synthetic_sample(self, idx):
                    # This should never be called if ADASYN failed
                    raise ValueError("No synthetic samples available")
                    
                HAM10000Dataset.get_synthetic_sample = get_synthetic_sample
        
        # Create train, val, and test datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        train_dataset.dataset.transform = train_transform
        
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        val_dataset.dataset.transform = test_transform
        
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        test_dataset.dataset.transform = test_transform
        
        # Calculate class weights for handling class imbalance
        if apply_smote and 'adasyn_successful' in locals() and adasyn_successful and 'y_resampled' in locals():
            # Use resampled labels for class weights if SMOTE was applied successfully
            class_counts = np.bincount(y_resampled)
        else:
            # Otherwise use original labels
            try:
                labels = [full_dataset.data_frame.iloc[i]['label'] if i >= 0 else full_dataset.synthetic_samples[i][1] 
                         for i in train_idx if (i >= 0 and i < len(full_dataset.data_frame)) or 
                         (i < 0 and hasattr(full_dataset, 'synthetic_samples') and i in full_dataset.synthetic_samples)]
                class_counts = np.bincount(labels)
            except Exception as e:
                print(f"Error calculating class weights: {e}. Using uniform weights.")
                # Fallback to uniform weights
                class_counts = np.ones(full_dataset.num_classes)
        
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
    
    else:  # MedMNIST datasets
        # Create MedMNIST datasets
        full_dataset = MedMNISTDataset(dataset_name, split='train', transform=None)
        
        train_dataset = MedMNISTDataset(dataset_name, split='train', transform=train_transform)
        val_dataset = MedMNISTDataset(dataset_name, split='val', transform=test_transform)
        test_dataset = MedMNISTDataset(dataset_name, split='test', transform=test_transform)
        
        # Calculate class weights
        train_labels = []
        for _, label in DataLoader(train_dataset, batch_size=batch_size):
            # Flatten labels if they are multi-dimensional
            if len(label.shape) > 1:
                label = label.flatten()
            train_labels.extend(label.numpy())
        
        train_labels = np.array(train_labels).flatten()
        class_counts = np.bincount(train_labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
    
    # Define a custom collate function to handle PIL Images
    def custom_collate_fn(batch):
        images = []
        labels = []
        
        for image, label in batch:
            # Convert PIL Image to tensor if it's not already
            if isinstance(image, Image.Image):
                image = transforms.ToTensor()(image)
            images.append(image)
            labels.append(label)
        
        # Stack images and labels
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return images, labels
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader, test_loader, class_weights, full_dataset

def get_all_dataset_loaders(data_dir, batch_size=32, test_split=0.2, val_split=0.1, seed=42, apply_smote=True):
    """
    Create train, validation, and test data loaders for all four datasets.
    
    Args:
        data_dir (string): Directory with the datasets.
        batch_size (int): Batch size for the data loaders.
        test_split (float): Proportion of data to use for testing (only for HAM10000).
        val_split (float): Proportion of training data to use for validation (only for HAM10000).
        seed (int): Random seed for reproducibility.
        apply_smote (bool): Whether to apply SMOTE/ADASYN for balancing rare classes.
        
    Returns:
        dict: Dictionary containing data loaders for each dataset with keys:
              'octmnist', 'tissuemnist', 'pathmnist', 'ham10000'
        dict: Dictionary containing test loaders for each dataset.
    """
    datasets = ['octmnist', 'tissuemnist', 'pathmnist', 'ham10000']
    client_loaders = {}
    test_loaders = {}
    
    for i, dataset_name in enumerate(datasets):
        print(f"Loading dataset {dataset_name} for client {i+1}...")
        train_loader, val_loader, test_loader, class_weights, _ = get_data_loaders(
            data_dir, dataset_name, batch_size, test_split, val_split, seed, apply_smote, client_id=i+1
        )
        
        client_loaders[dataset_name] = {
            'train': train_loader,
            'val': val_loader,
            'class_weights': class_weights,
            'client_id': i+1
        }
        
        test_loaders[dataset_name] = test_loader
    
    return client_loaders, test_loaders