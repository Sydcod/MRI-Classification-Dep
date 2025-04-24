import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from .preprocessing import get_preprocessing_transforms

class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for Brain MRI images
    
    Loads images and converts them to grayscale, applies transformations
    """
    
    def __init__(self, data_frame=None, root_dir=None, transform=None, 
                 data_csv=None, class_to_idx=None):
        """
        Initialize the dataset
        
        Args:
            data_frame (pd.DataFrame, optional): DataFrame with file_path and label columns
            root_dir (str, optional): Root directory for images
            transform (callable, optional): Optional transform to be applied
            data_csv (str, optional): Path to CSV file with file paths and labels
            class_to_idx (dict, optional): Mapping from class names to indices
        """
        # Can initialize with either a DataFrame or a CSV file
        if data_frame is not None:
            self.data_frame = data_frame
        elif data_csv is not None:
            self.data_frame = pd.read_csv(data_csv)
        else:
            raise ValueError("Either data_frame or data_csv must be provided")
        
        self.root_dir = root_dir if root_dir else ''
        self.transform = transform
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        """Get the number of samples in the dataset"""
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        
        # Load image and convert to grayscale
        image = Image.open(img_path).convert('L')
        
        # Get label
        label = self.data_frame.iloc[idx, 1]
        
        # Convert label to numeric if needed
        if self.class_to_idx is not None and isinstance(label, str):
            label = self.class_to_idx[label]
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, int(label)
    
    def get_classes(self):
        """Get the list of unique classes in the dataset"""
        return sorted(self.data_frame['label'].unique())

def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split the dataset into training, validation and test sets
    
    Args:
        data_dir (str): Directory containing class subdirectories
        output_dir (str): Output directory for CSV files
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
        test_ratio (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df) DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Create lists to store file paths and labels
    file_paths = []
    labels = []
    
    # Iterate through class directories
    for class_label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_label)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
            
        # Process all images in the class directory
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                # Store relative path and label
                rel_path = os.path.join(class_label, img_file)
                file_paths.append(rel_path)
                labels.append(class_label)
    
    # Create DataFrame with all data
    df = pd.DataFrame({'file_path': file_paths, 'label': labels})
    
    # Perform stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['file_path'], df['label'],
        train_size=train_ratio,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Split remaining data into validation and test sets
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_test_ratio,
        stratify=y_temp,
        random_state=random_state
    )
    
    # Create DataFrames
    train_df = pd.DataFrame({'file_path': X_train, 'label': y_train})
    val_df = pd.DataFrame({'file_path': X_val, 'label': y_val})
    test_df = pd.DataFrame({'file_path': X_test, 'label': y_test})
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DataFrames to CSV
    train_df.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_dataset.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)
    
    print(f"Dataset split saved to {output_dir}")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def create_data_loaders(data_dir, batch_size=16, num_workers=4, csv_dir=None):
    """
    Create DataLoaders for training, validation and test sets
    
    Args:
        data_dir (str): Root directory for images
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
        csv_dir (str, optional): Directory containing CSV files
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_to_idx)
    """
    # Determine CSV directory
    if csv_dir is None:
        csv_dir = data_dir
    
    # Load CSV files
    train_csv = os.path.join(csv_dir, 'train_dataset.csv')
    val_csv = os.path.join(csv_dir, 'val_dataset.csv')
    test_csv = os.path.join(csv_dir, 'test_dataset.csv')
    
    # Get transforms
    train_transform = get_preprocessing_transforms(mode='train')
    val_transform = get_preprocessing_transforms(mode='val')
    test_transform = get_preprocessing_transforms(mode='test')
    
    # Create datasets
    train_dataset = BrainMRIDataset(data_csv=train_csv, root_dir=data_dir, transform=train_transform)
    val_dataset = BrainMRIDataset(data_csv=val_csv, root_dir=data_dir, transform=val_transform)
    test_dataset = BrainMRIDataset(data_csv=test_csv, root_dir=data_dir, transform=test_transform)
    
    # Get class to index mapping
    classes = train_dataset.get_classes()
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Update datasets with class_to_idx mapping
    train_dataset.class_to_idx = class_to_idx
    val_dataset.class_to_idx = class_to_idx
    test_dataset.class_to_idx = class_to_idx
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_to_idx 