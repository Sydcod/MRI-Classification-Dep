import torch
import torch.nn as nn
from torchvision import models

class DenseNet169MRI(nn.Module):
    """
    MobileNetV2 model adapted for brain MRI classification with grayscale input.
    (Note: Class name kept as DenseNet169MRI for backward compatibility)
    
    Features:
    - Uses lightweight MobileNetV2 architecture (much smaller memory footprint)
    - Adapts first layer for grayscale input
    - Preserves pretrained weights by averaging RGB channels
    - Customizable classification head
    - Support for freezing/unfreezing layers
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        """
        Initialize MobileNetV2 model for MRI classification
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use ImageNet pretrained weights
        """
        super(DenseNet169MRI, self).__init__()
        
        # Load pretrained model - using MobileNetV2 instead of DenseNet169 to reduce memory usage
        if pretrained:
            self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        else:
            self.model = models.mobilenet_v2(weights=None)
        
        # Adapt first convolutional layer to accept grayscale input (1 channel)
        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # If using pretrained weights, adapt the first layer by averaging RGB channels
        if pretrained:
            with torch.no_grad():
                self.model.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Modify the classifier to match the number of classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
    
    def freeze_features(self):
        """Freeze all feature layers, leaving only classifier trainable"""
        for param in self.model.features.parameters():
            param.requires_grad = False
            
    def unfreeze_features(self):
        """Unfreeze all feature layers for fine-tuning"""
        for param in self.model.features.parameters():
            param.requires_grad = True
            
    def freeze_up_to(self, block_number):
        """
        Freeze layers up to a specific block
        
        Args:
            block_number (int): Block number up to which to freeze 
                               (0-16 for MobileNetV2 features)
        """
        assert 0 <= block_number <= 17, "Block number must be between 0-17"
        
        # Freeze up to the specified block number
        for i in range(min(block_number + 1, len(self.model.features))):
            for param in self.model.features[i].parameters():
                param.requires_grad = False
                
    def get_grad_cam_target_layer(self):
        """Return the target layer to use for Grad-CAM visualization"""
        # Use the last convolutional layer for Grad-CAM
        return self.model.features[-1]
