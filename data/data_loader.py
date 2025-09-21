import os
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
import cv2
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in HAM10000 dataset.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ElasticTransform:
    """
    Elastic deformation transform for medical image augmentation.
    """
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, random_state=None):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.random_state = random_state
    
    def __call__(self, image):
        if self.random_state:
            np.random.seed(self.random_state)
        
        image = np.array(image)
        shape = image.shape
        shape_size = shape[:2]
        
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + np.random.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        
        # Elastic deformation
        dx = np.random.uniform(-1, 1, shape_size) * self.alpha
        dy = np.random.uniform(-1, 1, shape_size) * self.alpha
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)
        
        x, y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        if len(shape) == 3:
            for i in range(shape[2]):
                image[:,:,i] = cv2.remap(image[:,:,i], indices[1].astype(np.float32), indices[0].astype(np.float32), cv2.INTER_LINEAR)
        else:
            image = cv2.remap(image, indices[1].astype(np.float32), indices[0].astype(np.float32), cv2.INTER_LINEAR)
        
        return Image.fromarray(image)

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
        
        # Create a custom transform that ensures PIL Image format
        def ensure_pil_transform(img):
            if isinstance(img, np.ndarray):
                # Handle different array shapes
                if img.ndim == 2:  # Grayscale
                    return Image.fromarray(img, mode='L')
                elif img.ndim == 3 and img.shape[2] == 1:  # Grayscale with channel dim
                    return Image.fromarray(img.squeeze(), mode='L')
                elif img.ndim == 3 and img.shape[2] == 3:  # RGB
                    return Image.fromarray(img, mode='RGB')
                else:
                    return Image.fromarray(img.squeeze(), mode='L')
            elif isinstance(img, torch.Tensor):
                # Convert tensor to numpy first, then to PIL
                img_np = img.numpy()
                if img_np.ndim == 2:
                    return Image.fromarray(img_np, mode='L')
                elif img_np.ndim == 3 and img_np.shape[0] == 1:
                    return Image.fromarray(img_np.squeeze(), mode='L')
                elif img_np.ndim == 3 and img_np.shape[0] == 3:
                    return Image.fromarray(img_np.transpose(1, 2, 0), mode='RGB')
                else:
                    return Image.fromarray(img_np.squeeze(), mode='L')
            elif isinstance(img, Image.Image):
                return img
            else:
                # Fallback: try to convert to array first
                return Image.fromarray(np.array(img))
            
        # Apply our transform and then the user's transform if provided
        if transform is not None:
            dataset_transform = transforms.Compose([ensure_pil_transform, transform])
        else:
            dataset_transform = ensure_pil_transform
            
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
        # HAM10000 transforms (RGB) with enhanced augmentation for rare classes
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Larger initial size
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive cropping
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),  # Increased rotation
            # Enhanced color augmentation for skin lesion diversity
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            # Add elastic transform for medical image augmentation (with probability)
            transforms.Lambda(lambda x: ElasticTransform(alpha=1, sigma=50)(x) if np.random.random() < 0.3 else x),
            # Additional geometric transforms
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            # Ensure we have PIL Image for remaining transforms
            transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else transforms.ToPILImage()(x)),
            # Perspective transform for more diversity
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),  # Convert to tensor first
            transforms.Normalize(med_mean, med_std),  # Normalize
            # Add random erasing for regularization (after normalization)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Must be last transform
            transforms.Normalize(med_mean, med_std)
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
        
        # Enhanced class balancing with WeightedRandomSampler and focal loss preparation
        if apply_smote:
            print("Configuring enhanced class balancing for HAM10000 rare classes...")
            # Calculate class weights for WeightedRandomSampler
            y_train = []
            for idx in train_idx:
                y_train.append(full_dataset.diagnosis_mapping[metadata.iloc[idx]['dx']])
            
            y_train = np.array(y_train)
            class_distribution = Counter(y_train)
            print("Class distribution:", class_distribution)
            
            # Calculate inverse frequency weights with smoothing for rare classes
            class_counts = np.bincount(y_train)
            # Add smoothing factor to prevent extreme weights
            smoothing_factor = 0.1
            class_weights_array = 1. / (class_counts + smoothing_factor)
            
            # Apply extra boost to very rare classes (< 5% of data)
            total_samples = len(y_train)
            for i, count in enumerate(class_counts):
                if count < 0.05 * total_samples:  # Very rare class
                    class_weights_array[i] *= 2.0  # Extra boost
                    print(f"Applied rare class boost to class {i} (count: {count})")
            
            sample_weights = class_weights_array[y_train]
            
            # Store sample weights and class info for WeightedRandomSampler and focal loss
            full_dataset.sample_weights = sample_weights
            full_dataset.class_weights_array = class_weights_array
            full_dataset.use_weighted_sampling = True
            
            print("Enhanced class balancing configured:")
            print(f"  - WeightedRandomSampler with rare class boost")
            print(f"  - Class weights: {class_weights_array}")
        
        # Create train, val, and test datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        train_dataset.dataset.transform = train_transform
        
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        val_dataset.dataset.transform = test_transform
        
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        test_dataset.dataset.transform = test_transform
        
        # Calculate class weights for handling class imbalance (for loss functions)
        labels = [full_dataset.data_frame.iloc[i]['label'] for i in train_idx]
        class_counts = np.bincount(labels)
        
        # Create focal loss alpha weights (inverse frequency with smoothing)
        focal_alpha = 1. / (class_counts + 0.1)  # Add smoothing
        focal_alpha = focal_alpha / focal_alpha.sum() * len(focal_alpha)  # Normalize
        
        # Standard class weights for CrossEntropyLoss
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        
        # Store focal loss alpha for use in training
        full_dataset.focal_alpha = torch.tensor(focal_alpha, dtype=torch.float)
        
        print(f"Focal loss alpha weights: {focal_alpha}")
    
    else:  # MedMNIST datasets
        # Create MedMNIST datasets
        full_dataset = MedMNISTDataset(dataset_name, split='train', transform=None)
        
        train_dataset = MedMNISTDataset(dataset_name, split='train', transform=train_transform)
        val_dataset = MedMNISTDataset(dataset_name, split='val', transform=test_transform)
        test_dataset = MedMNISTDataset(dataset_name, split='test', transform=test_transform)
        
        # Calculate class weights more efficiently
        train_labels = []
        # Sample a subset for efficiency
        sample_size = min(1000, len(train_dataset))
        indices = np.random.choice(len(train_dataset), sample_size, replace=False)
        
        for idx in indices:
            _, label = train_dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item() if label.dim() == 0 else label.flatten()[0]
            elif isinstance(label, np.ndarray):
                label = label.item() if label.size == 1 else label.flatten()[0]
            train_labels.append(label)
        
        train_labels = np.array(train_labels).flatten()
        class_counts = np.bincount(train_labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        
        # No focal loss for MedMNIST datasets
        focal_criterion = None
    
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
    
    # Create data loaders with custom collate function and WeightedRandomSampler if needed
    if dataset_name.lower() == 'ham10000' and apply_smote and hasattr(full_dataset, 'use_weighted_sampling'):
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(full_dataset.sample_weights, len(full_dataset.sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Create focal loss criterion for HAM10000
    focal_criterion = None
    if dataset_name.lower() == 'ham10000' and hasattr(full_dataset, 'focal_alpha'):
        focal_criterion = FocalLoss(alpha=full_dataset.focal_alpha, gamma=2.0)
        print("Created Focal Loss criterion for HAM10000 rare class handling")
    
    return train_loader, val_loader, test_loader, class_weights, full_dataset, focal_criterion, focal_criterion

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
        result = get_data_loaders(
            data_dir, dataset_name, batch_size, test_split, val_split, seed, apply_smote, client_id=i+1
        )
        
        # Handle both old and new return formats
        if len(result) == 6:
            train_loader, val_loader, test_loader, class_weights, _, focal_criterion = result
        else:
            train_loader, val_loader, test_loader, class_weights, _ = result[:5]
            focal_criterion = None
        
        client_loaders[dataset_name] = {
            'train': train_loader,
            'val': val_loader,
            'class_weights': class_weights,
            'focal_criterion': focal_criterion,
            'client_id': i+1
        }
        
        test_loaders[dataset_name] = test_loader
    
    return client_loaders, test_loaders