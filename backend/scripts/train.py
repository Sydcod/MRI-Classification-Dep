import os
import sys
import argparse
import json
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.densenet169 import DenseNet169MRI
from models.model_utils import (
    FocalLoss, EarlyStopping, train_epoch, validate,
    save_model, create_optimizer, create_scheduler
)
from data.dataset import split_dataset, create_data_loaders

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train brain MRI classification model")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="../results", help="Path to output directory")
    parser.add_argument("--split_dataset", action="store_true", help="Split dataset before training")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty)")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--freeze_layers", type=int, default=2, help="Number of DenseNet blocks to freeze (0-4)")
    
    # Loss function parameters
    parser.add_argument("--focal_loss_alpha", type=float, default=0.25, help="Alpha parameter for focal loss")
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0, help="Gamma parameter for focal loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    
    # Scheduler parameters
    parser.add_argument("--scheduler_t0", type=int, default=5, help="T_0 parameter for cosine annealing")
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6, help="Minimum learning rate")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=7, help="Patience for early stopping")
    parser.add_argument("--early_stopping_delta", type=float, default=0.001, help="Minimum improvement for early stopping")
    
    # Hardware optimization
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_clipping", action="store_true", help="Use gradient clipping")
    parser.add_argument("--grad_clip_value", type=float, default=1.0, help="Gradient clipping value")
    
    return parser.parse_args()

def main():
    """Train the model"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directories
    checkpoint_dir = os.path.join(args.output_dir, "models")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up TensorBoard writer
    run_name = f"densenet169_mri_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_log_dir = os.path.join(log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Split dataset if requested
    if args.split_dataset:
        print("Splitting dataset...")
        splits_dir = os.path.join(args.data_dir, "splits")
        
        # Create splits directory if it doesn't exist
        try:
            os.makedirs(splits_dir, exist_ok=True)
            
            split_dataset(
                args.data_dir,
                splits_dir,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_state=42
            )
        except PermissionError:
            # If we can't write to the data_dir, use a subdirectory in output_dir instead
            print(f"Warning: Cannot write to {splits_dir}. Using alternate location.")
            splits_dir = os.path.join(args.output_dir, "splits")
            os.makedirs(splits_dir, exist_ok=True)
            
            split_dataset(
                args.data_dir,
                splits_dir,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_state=42
            )
    
    # Create dataloaders
    print("Creating data loaders...")
    try:
        # First try loading from the data_dir/splits
        splits_dir = os.path.join(args.data_dir, "splits")
        if not os.path.exists(splits_dir) or not os.path.isfile(os.path.join(splits_dir, "train_dataset.csv")):
            # If not found, try the alternate location
            splits_dir = os.path.join(args.output_dir, "splits")
            
        train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            csv_dir=splits_dir
        )
    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        print("Please make sure the dataset is correctly formatted and splits are available.")
        return
    
    # Save class mapping
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    with open(os.path.join(checkpoint_dir, f"{run_name}_classes.json"), 'w') as f:
        json.dump(idx_to_class, f, indent=4)
    
    # Create model
    print("Creating model...")
    model = DenseNet169MRI(num_classes=args.num_classes, pretrained=True)
    
    # Freeze specified number of layers
    if args.freeze_layers > 0:
        print(f"Freezing {args.freeze_layers} DenseNet blocks...")
        model.freeze_up_to(args.freeze_layers)
    
    # Move model to device
    model = model.to(device)
    
    # Create loss function, optimizer and scheduler
    criterion = FocalLoss(
        alpha=args.focal_loss_alpha,
        gamma=args.focal_loss_gamma,
        label_smoothing=args.label_smoothing
    )
    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = create_scheduler(
        optimizer,
        T_0=args.scheduler_t0,
        eta_min=args.scheduler_min_lr
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.early_stopping_delta,
        checkpoint_path=os.path.join(checkpoint_dir, f"{run_name}_best.pth")
    )
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        # Training phase
        start_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_mixed_precision=args.mixed_precision,
            gradient_clipping=args.gradient_clipping,
            clip_value=args.grad_clip_value
        )
        
        # Validation phase
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1-Score/val', val_f1, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(
                model,
                optimizer,
                scheduler,
                epoch,
                {
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_f1': val_f1
                },
                checkpoint_dir,
                f"{run_name}_best.pth"
            )
            print(f"Saved new best model with F1 score: {val_f1:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_model(
                model,
                optimizer,
                scheduler,
                epoch,
                {
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_f1': val_f1
                },
                checkpoint_dir,
                f"{run_name}_epoch{epoch+1}.pth"
            )
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{run_name}_best.pth")))
    
    # Evaluate on test set
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
    print(f"\nTest Results: Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1 Score: {test_f1:.4f}")
    
    # Save final metrics
    final_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'best_val_f1': best_val_f1
    }
    with open(os.path.join(checkpoint_dir, f"{run_name}_metrics.json"), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    # Save a copy of the best model with a simpler name for easier loading
    os.makedirs(os.path.join(checkpoint_dir), exist_ok=True)
    
    # Instead of just saving the state dict, save the full checkpoint for consistency
    best_model_path = os.path.join(checkpoint_dir, f"{run_name}_best.pth")
    default_model_path = os.path.join(checkpoint_dir, "default.pth")
    
    try:
        # Copy the best model to default.pth
        if os.path.exists(best_model_path):
            print(f"Saving default model...")
            # Load the best model checkpoint
            best_checkpoint = torch.load(best_model_path, map_location=device)
            
            # Save with the same format for consistency
            torch.save(
                best_checkpoint,
                default_model_path
            )
            
            # Save class mapping for the default model
            with open(os.path.join(checkpoint_dir, "default_classes.json"), 'w') as f:
                json.dump(idx_to_class, f, indent=4)
            
            print(f"Default model saved to {default_model_path}")
        else:
            print(f"Warning: Best model file not found at {best_model_path}")
    except Exception as e:
        print(f"Error saving default model: {str(e)}")
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"Training completed. Best model saved to {best_model_path}")
    if os.path.exists(default_model_path):
        print(f"Default model saved to {default_model_path}")

if __name__ == "__main__":
    main() 