import torch
import torch.nn as nn
from torchvision import models

class DenseNet169MRI(nn.Module):
    """
    DenseNet169 model adapted for brain MRI classification with grayscale input.
    
    Features:
    - Adapts first layer for grayscale input
    - Preserves pretrained weights by averaging RGB channels
    - Customizable classification head
    - Support for freezing/unfreezing layers
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        """
        Initialize DenseNet169 model for MRI classification
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use ImageNet pretrained weights
        """
        super(DenseNet169MRI, self).__init__()
        
        # Load pretrained model
        if pretrained:
            self.model = models.densenet169(weights='IMAGENET1K_V1')
        else:
            self.model = models.densenet169(weights=None)
        
        # Adapt first convolutional layer to accept grayscale input (1 channel)
        original_conv = self.model.features.conv0
        self.model.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # If using pretrained weights, adapt the first layer by averaging RGB channels
        if pretrained:
            with torch.no_grad():
                self.model.features.conv0.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Modify the classifier to match the number of classes
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
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
        Freeze layers up to a specific DenseBlock
        
        Args:
            block_number (int): Block number up to which to freeze 
                               (1-4 for DenseNet169)
        """
        assert 0 <= block_number <= 4, "Block number must be between 0-4"
        
        if block_number >= 0:
            for param in self.model.features.conv0.parameters():
                param.requires_grad = False
            
        if block_number >= 1:
            for param in self.model.features.denseblock1.parameters():
                param.requires_grad = False
            for param in self.model.features.transition1.parameters():
                param.requires_grad = False
                
        if block_number >= 2:
            for param in self.model.features.denseblock2.parameters():
                param.requires_grad = False
            for param in self.model.features.transition2.parameters():
                param.requires_grad = False
                
        if block_number >= 3:
            for param in self.model.features.denseblock3.parameters():
                param.requires_grad = False
            for param in self.model.features.transition3.parameters():
                param.requires_grad = False
                
        if block_number >= 4:
            for param in self.model.features.denseblock4.parameters():
                param.requires_grad = False
                
    def get_grad_cam_target_layer(self):
        """Return the target layer to use for Grad-CAM visualization"""
        return self.model.features.denseblock4 