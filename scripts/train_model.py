"""
Training script for ZKML-Guard model

Reproduces the 94.7% accuracy results from the paper.

Training configuration:
- Optimizer: Adam (lr=0.001)
- Batch size: 256
- Dropout: 0.3
- Early stopping on validation loss
- Weighted sampling for class imbalance
- 5-fold cross-validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.zkml_guard_model import ZKMLGuardModel


class TransactionDataset(Dataset):
    """Dataset for transaction risk classification."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_training_data(data_dir: Path):
    """
    Load training data from directory.
    
    Expected structure:
        data_dir/
            features.npy  # Shape: (n_samples, 42)
            labels.npy    # Shape: (n_samples,), values: 0-4
    """
    features_path = data_dir / "features.npy"
    labels_path = data_dir / "labels.npy"
    
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Training data not found in {data_dir}. "
            "Expected files: features.npy, labels.npy"
        )
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"Loaded training data:")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Class distribution: {np.bincount(labels)}")
    
    return features, labels


def create_weighted_sampler(labels):
    """Create weighted sampler for imbalanced dataset."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(features)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            features, labels = features.to(device), labels.to(device)
            
            logits, probs = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Detailed metrics per class
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    return avg_loss, accuracy, precision, recall, f1, support


def train_model(args):
    """Main training function."""
    print("=" * 80)
    print("ZKML-Guard Model Training")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    features, labels = load_training_data(Path(args.data))
    
    # Setup for cross-validation
    if args.cross_validate:
        print(f"\nRunning {args.n_folds}-fold cross-validation...")
        kfold = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(features, labels)):
            print(f"\n{'='*80}")
            print(f"Fold {fold + 1}/{args.n_folds}")
            print(f"{'='*80}")
            
            # Split data
            train_features, train_labels = features[train_idx], labels[train_idx]
            val_features, val_labels = features[val_idx], labels[val_idx]
            
            # Train fold
            model, metrics = train_single_fold(
                train_features, train_labels,
                val_features, val_labels,
                args, device, fold
            )
            
            fold_results.append(metrics)
        
        # Aggregate cross-validation results
        print(f"\n{'='*80}")
        print("Cross-Validation Results")
        print(f"{'='*80}")
        
        accuracies = [r['best_val_accuracy'] for r in fold_results]
        print(f"\nAccuracies by fold: {[f'{a:.3f}' for a in accuracies]}")
        print(f"Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"Paper result: 0.947 ± 0.012")
        
    else:
        # Single training run
        # Split into train and validation
        from sklearn.model_selection import train_test_split
        train_features, val_features, train_labels, val_labels = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        model, metrics = train_single_fold(
            train_features, train_labels,
            val_features, val_labels,
            args, device, fold=None
        )
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")


def train_single_fold(train_features, train_labels, val_features, val_labels, args, device, fold=None):
    """Train a single model (or fold)."""
    
    # Create datasets
    train_dataset = TransactionDataset(train_features, train_labels)
    val_dataset = TransactionDataset(val_features, val_labels)
    
    # Create dataloaders with weighted sampling
    train_sampler = create_weighted_sampler(train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Initialize model
    model = ZKMLGuardModel(
        input_dim=train_features.shape[1],
        num_classes=5,
        dropout_rate=args.dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, precision, recall, f1, support = validate(
            model, val_loader, criterion, device
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Print per-class metrics
        class_names = ['Safe', 'Low Risk', 'Medium Risk', 'High Risk', 'Critical']
        print("\nPer-class metrics:")
        for i, name in enumerate(class_names):
            print(f"  {name:15s}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, N={support[i]}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            patience_counter = 0
            
            # Save best model
            if fold is not None:
                save_path = Path(args.output) / f"zkml_guard_fold{fold}.pth"
            else:
                save_path = Path(args.output) / "zkml_guard.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(save_path)
            print(f"✓ Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered (patience={args.patience})")
                break
    
    # Save training history
    if fold is not None:
        history_path = Path(args.output) / f"history_fold{fold}.json"
    else:
        history_path = Path(args.output) / "history.json"
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation on validation set
    print(f"\n{'='*60}")
    print("Final Validation Results")
    print(f"{'='*60}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Accuracy: {best_val_accuracy:.4f}")
    
    metrics = {
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy,
        'final_epoch': epoch + 1,
        'history': history
    }
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train ZKML-Guard model")
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--output', type=str, default='models/pytorch',
                       help='Output directory for models')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Cross-validation
    parser.add_argument('--cross-validate', action='store_true',
                       help='Run k-fold cross-validation')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    
    # Device
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Train
    train_model(args)


if __name__ == "__main__":
    main()
