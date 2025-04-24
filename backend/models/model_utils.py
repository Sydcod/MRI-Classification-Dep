import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Implementation of the focal loss function:
    FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.1):
        """
        Initialize Focal Loss
        
        Args:
            alpha (float): Weighting factor in range (0,1) for handling class imbalance
            gamma (float): Focusing parameter, reduces loss for well-classified examples
            reduction (str): 'mean', 'sum' or 'none'
            label_smoothing (float): Label smoothing factor, helps prevent overconfidence
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs (torch.Tensor): Model predictions, shape [B, C]
            targets (torch.Tensor): Target values, shape [B]
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Apply softmax to get probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Get probability for the target class
        num_classes = inputs.shape[1]
        
        # One-hot encode the targets with label smoothing
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + (self.label_smoothing / num_classes)
        
        # Calculate focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        # Calculate the loss with alpha weighting and focal weight
        loss = -self.alpha * focal_weight * targets_one_hot * log_probs
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss.sum(dim=1)

class EarlyStopping:
    """
    Early stopping to prevent overfitting
    
    Stops training when a monitored metric has stopped improving
    """
    
    def __init__(self, patience=7, min_delta=0.001, checkpoint_path='model_checkpoint.pt'):
        """
        Initialize early stopping
        
        Args:
            patience (int): Number of epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            checkpoint_path (str): Path to save best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        """
        Check if training should be stopped
        
        Args:
            val_loss (float): Validation loss for current epoch
            model (nn.Module): Model to save if improved
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        score = -val_loss  # Higher score is better (negative loss)
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
        return self.early_stop
        
    def save_checkpoint(self, model):
        """Save model checkpoint when validation loss decreases"""
        torch.save(model.state_dict(), self.checkpoint_path)
        
def train_epoch(model, train_loader, criterion, optimizer, device, 
                use_mixed_precision=True, gradient_clipping=True, clip_value=1.0):
    """
    Train for one epoch
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        use_mixed_precision (bool): Whether to use mixed precision training
        gradient_clipping (bool): Whether to use gradient clipping
        clip_value (float): Value to clip gradients to
        
    Returns:
        float: Average training loss
    """
    model.train()
    train_loss = 0.0
    scaler = GradScaler() if use_mixed_precision else None
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixed precision training
        if scaler:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if gradient_clipping:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
        
        train_loss += loss.item()
        
    return train_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to validate on
        
    Returns:
        tuple: Average validation loss, accuracy, F1 score
    """
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(val_targets, val_preds)
    f1 = f1_score(val_targets, val_preds, average='macro')
    
    return val_loss / len(val_loader), accuracy, f1

def save_model(model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir, filename):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler: Learning rate scheduler
        epoch (int): Current epoch
        val_metrics (dict): Validation metrics
        checkpoint_dir (str): Directory to save checkpoint to
        filename (str): Filename for the checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': val_metrics
    }
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    
    # Also save a JSON with metrics for easy loading
    metrics_file = os.path.join(checkpoint_dir, f"{os.path.splitext(filename)[0]}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(val_metrics, f, indent=4)

def load_model(model, checkpoint_path, device):
    """
    Load model from checkpoint
    
    Args:
        model (nn.Module): Model to load checkpoint into
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Device to load model on
        
    Returns:
        nn.Module: Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

def create_optimizer(model, learning_rate=1e-4, weight_decay=1e-5):
    """
    Create AdamW optimizer with appropriate parameters
    
    Args:
        model (nn.Module): Model to optimize
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay (L2 penalty)
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def create_scheduler(optimizer, T_0=5, T_mult=1, eta_min=1e-6):
    """
    Create learning rate scheduler
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        T_0 (int): Number of iterations for the first restart
        T_mult (int): Factor for increasing T_i after a restart
        eta_min (float): Minimum learning rate
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured scheduler
    """
    return CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=T_0, 
        T_mult=T_mult, 
        eta_min=eta_min
    ) 