import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class ScoreCAM:
    """
    Score-CAM implementation for CNN-based models
    
    Creates class activation maps to highlight important regions in the image
    that influenced the model's decision, without requiring gradients.
    
    This makes it more robust for models in evaluation mode or when gradients
    are not available.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Score-CAM
        
        Args:
            model (nn.Module): Trained model
            target_layer (nn.Module): Layer to generate CAM from, typically the last convolutional layer
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # Register hook for capturing activations
        self.hook = self.target_layer.register_forward_hook(self._get_activations)
    
    def _get_activations(self, module, input, output):
        """Hook to capture activations"""
        self.activations = output.detach()
    
    def _remove_hook(self):
        """Remove registered hook to prevent memory leaks"""
        self.hook.remove()
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Score-CAM for the input tensor
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [1, C, H, W]
            target_class (int, optional): Target class index. If None, uses the predicted class.
            
        Returns:
            np.ndarray: Score-CAM heatmap, values between 0 and 1, shape [H, W]
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass to get activations
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Get the predicted class if target_class is not specified
            if target_class is None:
                target_class = outputs.argmax(dim=1).item()
        
        # Get activation maps
        activation_maps = self.activations
        
        # Create empty mask
        B, C, H, W = activation_maps.shape
        cam = torch.zeros((H, W), dtype=torch.float32, device=activation_maps.device)
        
        # Loop through each activation channel
        for i in range(C):
            # Create a masked input by upsampling the activation map
            mask = activation_maps[0, i:i+1, :, :]
            
            # Normalize the mask
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            
            # Upsample the mask to the input size
            mask_upsampled = F.interpolate(
                mask.unsqueeze(0),
                size=(input_tensor.shape[2], input_tensor.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            
            # Apply the mask to the input
            masked_input = input_tensor * mask_upsampled
            
            # Forward pass with the masked input
            with torch.no_grad():
                masked_output = self.model(masked_input)
            
            # Get the score for the target class
            score = masked_output[0, target_class].item()
            
            # Add the weighted mask to the CAM
            cam += score * mask.squeeze().cpu()
        
        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Normalize CAM
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Convert to numpy
        cam = cam.numpy()
        
        # Clean up to prevent memory leaks
        self._remove_hook()
        
        return cam
    
    def visualize(self, input_tensor, original_image, target_class=None, alpha=0.5, colormap='jet'):
        """
        Generate and visualize Score-CAM overlay on the original image
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [1, C, H, W]
            original_image (PIL.Image or np.ndarray): Original image for overlay
            target_class (int, optional): Target class index. If None, uses the predicted class.
            alpha (float): Transparency of the heatmap overlay
            colormap (str): Matplotlib colormap name
            
        Returns:
            tuple: (overlay_image, cam_heatmap) as numpy arrays
        """
        # Generate Score-CAM
        cam = self.generate(input_tensor, target_class)
        
        # Convert to heatmap
        heatmap = self._to_heatmap(cam, colormap)
        
        # Ensure original_image is a numpy array
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Convert grayscale to RGB if needed
        if len(original_image.shape) == 2:
            original_image = np.stack([original_image] * 3, axis=2)
        
        # Create overlay
        overlay = self._create_overlay(original_image, heatmap, alpha)
        
        return overlay, heatmap
    
    def _to_heatmap(self, cam, colormap='jet'):
        """
        Convert Score-CAM to a colored heatmap
        
        Args:
            cam (np.ndarray): Score-CAM array, values between 0 and 1
            colormap (str): Matplotlib colormap name
            
        Returns:
            np.ndarray: Colored heatmap as RGB array
        """
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(cam)
        
        # Convert to RGB (remove alpha channel)
        heatmap = heatmap[:, :, :3]
        
        # Scale to 0-255 range
        heatmap = (heatmap * 255).astype(np.uint8)
        
        return heatmap
    
    def _create_overlay(self, original_image, heatmap, alpha=0.5):
        """
        Create an overlay of the heatmap on the original image
        
        Args:
            original_image (np.ndarray): Original image, shape [H, W, 3]
            heatmap (np.ndarray): Colored heatmap, shape [H, W, 3]
            alpha (float): Transparency of the heatmap overlay
            
        Returns:
            np.ndarray: Overlay image
        """
        # Resize heatmap to match original image if needed
        if heatmap.shape[:2] != original_image.shape[:2]:
            heatmap = np.array(Image.fromarray(heatmap).resize(
                (original_image.shape[1], original_image.shape[0]),
                Image.LANCZOS
            ))
        
        # Create overlay
        overlay = (1 - alpha) * original_image + alpha * heatmap
        
        # Clip values to valid range
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay 