import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SHAPExplainer:
    """
    SHAP-based explainer for CNN-based models
    
    Implements a simplified Shapley value approach for MRI image models 
    to highlight important regions that influence the model's decision.
    
    This is a feature attribution method that assigns an importance value to
    each pixel in the input image.
    """
    
    def __init__(self, model, baseline=None, n_samples=10):
        """
        Initialize SHAP Explainer
        
        Args:
            model (nn.Module): Trained model
            baseline (torch.Tensor, optional): Baseline image for integration
                If None, a black image (zeros) will be used
            n_samples (int): Number of samples for approximation
        """
        self.model = model
        self.baseline = baseline
        self.n_samples = n_samples
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate SHAP values for the input tensor
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [1, C, H, W]
            target_class (int, optional): Target class index. If None, uses the predicted class.
            
        Returns:
            np.ndarray: SHAP values, shape [H, W]
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Get the predicted class if target_class is not specified
            if target_class is None:
                target_class = outputs.argmax(dim=1).item()
        
        # Create a baseline of zeros if not provided
        if self.baseline is None:
            self.baseline = torch.zeros_like(input_tensor)
        
        # Get input dimensions
        C, H, W = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
        
        # Create empty SHAP map
        shap_values = torch.zeros((H, W), dtype=torch.float32, device=input_tensor.device)
        
        # Create random masks for sampling
        # We'll use a simplified approach with random sampling for efficiency
        for _ in range(self.n_samples):
            # Create a random binary mask
            mask = torch.rand((1, 1, H, W), device=input_tensor.device) > 0.5
            mask = mask.float()
            
            # Create perturbed input
            perturbed_input = input_tensor * mask + self.baseline * (1 - mask)
            
            # Forward pass with perturbed input
            with torch.no_grad():
                perturbed_output = self.model(perturbed_input)
                perturbed_score = perturbed_output[0, target_class]
            
            # Compute contribution
            contribution = mask[0, 0] * perturbed_score
            shap_values += contribution
        
        # Average over samples
        shap_values /= self.n_samples
        
        # Normalize SHAP values
        if shap_values.max() > shap_values.min():
            shap_values = (shap_values - shap_values.min()) / (shap_values.max() - shap_values.min())
        
        # Convert to numpy
        shap_values = shap_values.cpu().numpy()
        
        return shap_values
    
    def visualize(self, input_tensor, original_image, target_class=None, alpha=0.5, colormap='RdBu_r'):
        """
        Generate and visualize SHAP values overlay on the original image
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [1, C, H, W]
            original_image (PIL.Image or np.ndarray): Original image for overlay
            target_class (int, optional): Target class index. If None, uses the predicted class.
            alpha (float): Transparency of the heatmap overlay
            colormap (str): Matplotlib colormap name, default is RdBu_r for SHAP
            
        Returns:
            tuple: (overlay_image, shap_heatmap) as numpy arrays
        """
        # Generate SHAP values
        shap_values = self.generate(input_tensor, target_class)
        
        # Convert to heatmap
        heatmap = self._to_heatmap(shap_values, colormap)
        
        # Ensure original_image is a numpy array
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Convert grayscale to RGB if needed
        if len(original_image.shape) == 2:
            original_image = np.stack([original_image] * 3, axis=2)
        
        # Create overlay
        overlay = self._create_overlay(original_image, heatmap, alpha)
        
        return overlay, heatmap
    
    def _to_heatmap(self, shap_values, colormap='RdBu_r'):
        """
        Convert SHAP values to a colored heatmap
        
        Args:
            shap_values (np.ndarray): SHAP values, values between 0 and 1
            colormap (str): Matplotlib colormap name, default is RdBu_r for SHAP
            
        Returns:
            np.ndarray: Colored heatmap as RGB array
        """
        # Center the colormap for diverging visualization
        # This is important for SHAP where both positive and negative contributions are meaningful
        centered_values = 2 * shap_values - 1  # Map from [0,1] to [-1,1]
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(0.5 + centered_values/2)  # Center at 0.5
        
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