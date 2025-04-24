import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def get_preprocessing_transforms(mode='train'):
    """
    Get preprocessing transforms for MRI images
    
    Args:
        mode (str): 'train', 'val', or 'test'
        
    Returns:
        transforms.Compose: Composed transforms
    """
    if mode == 'train':
        # Apply augmentation only to training data
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            # Correct normalization for grayscale
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        # No augmentation for validation and test sets
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # Correct normalization for grayscale
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

def preprocess_image_for_prediction(image_path_or_file, device='cpu'):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path_or_file: Path to image or file-like object
        device (str): Device to load tensor on
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if isinstance(image_path_or_file, str):
        # Load from path
        image = Image.open(image_path_or_file).convert('L')  # Convert to grayscale
    else:
        # Load from file-like object
        image = Image.open(image_path_or_file).convert('L')
    
    # Apply preprocessing transforms for inference
    transform = get_preprocessing_transforms(mode='test')
    
    # Add batch dimension and move to device
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    return image_tensor

def preprocess_image_for_display(image_path_or_file):
    """
    Preprocess image for display purposes
    
    Args:
        image_path_or_file: Path to image or file-like object
        
    Returns:
        np.ndarray: Preprocessed image as numpy array
    """
    if isinstance(image_path_or_file, str):
        # Load from path
        image = Image.open(image_path_or_file).convert('L')  # Convert to grayscale
    else:
        # Load from file-like object
        image = Image.open(image_path_or_file).convert('L')
    
    # Resize the image to standard size without normalization
    image = image.resize((512, 512))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    return image_array

def apply_preprocessing_to_dataset(dataset_dir, output_dir):
    """
    Apply preprocessing to all images in a dataset
    
    Args:
        dataset_dir (str): Path to dataset directory
        output_dir (str): Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all the subdirectories (classes)
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):
            # Create corresponding output directory
            class_output_dir = os.path.join(output_dir, class_dir)
            os.makedirs(class_output_dir, exist_ok=True)
            
            # Process all images in this class
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = os.path.join(class_path, img_file)
                    
                    try:
                        # Open and convert to grayscale
                        image = Image.open(img_path).convert('L')
                        
                        # Resize to standard size
                        image = image.resize((512, 512))
                        
                        # Save the processed image
                        output_path = os.path.join(class_output_dir, img_file)
                        image.save(output_path)
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}") 