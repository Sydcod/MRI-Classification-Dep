import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCAM:
    """
    Grad-CAM implementation for CNN-based models
    
    Creates class activation maps to highlight important regions in the image
    that influenced the model's decision.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model (nn.Module): Trained model
            target_layer (nn.Module): Layer to generate CAM from, typically the last convolutional layer
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        
        # Hook for capturing activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        # Hook for capturing gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def _remove_hooks(self):
        """Remove registered hooks to prevent memory leaks"""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM for the input tensor
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [1, C, H, W]
            target_class (int, optional): Target class index. If None, uses the predicted class.
            
        Returns:
            np.ndarray: Grad-CAM heatmap, values between 0 and 1, shape [H, W]
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass
        outputs = self.model(input_tensor)
        
        # Get the predicted class if target_class is not specified
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()
        
        # Create one-hot encoding for the target class
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1
        
        # Zero the gradients
        self.model.zero_grad()
        
        # Backward pass (calculate gradients)
        outputs.backward(gradient=one_hot, retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations
        gradients = self.gradients
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU to focus on features that have a positive influence on the target class
        cam = F.relu(cam)
        
        # Resize to match input image size
        cam = F.interpolate(
            cam, 
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize to 0-1 range
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Convert to numpy array
        cam = cam.detach().cpu().numpy()[0, 0]
        
        # Clean up to prevent memory leaks
        self._remove_hooks()
        
        return cam
    
    def visualize(self, input_tensor, original_image, target_class=None, alpha=0.5, colormap='jet'):
        """
        Generate and visualize Grad-CAM overlay on the original image
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [1, C, H, W]
            original_image (PIL.Image or np.ndarray): Original image for overlay
            target_class (int, optional): Target class index. If None, uses the predicted class.
            alpha (float): Transparency of the heatmap overlay
            colormap (str): Matplotlib colormap name
            
        Returns:
            tuple: (overlay_image, cam_heatmap) as numpy arrays
        """
        # Generate Grad-CAM
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
        Convert Grad-CAM to a colored heatmap
        
        Args:
            cam (np.ndarray): Grad-CAM array, values between 0 and 1
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