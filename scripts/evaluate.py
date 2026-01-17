"""
Evaluation script for ZKML-Guard

Reproduces the results from the paper:
- Overall Accuracy: 94.7%
- Critical Precision: 97.3%
- Critical Recall: 91.8%
- False Positive Rate: 0.28%
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.zkml_guard_model import ZKMLGuardModel


def load_test_data(data_path: Path):
    """Load test dataset."""
    features_path = data_path / "features.npy"
    labels_path = data_path / "labels.npy"
    
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Test data not found in {data_path}")
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"Test set size: {len(labels)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return features, labels


def evaluate_model(model, features, labels, device):
    """Evaluate model on test set."""
    model.eval()
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features).to(device)
    
    # Get predictions
    with torch.no_grad():
        pred_classes, confidences, probabilities = model.predict(features_tensor)
    
    # Convert to numpy
    predictions = pred_classes.cpu().numpy()
    confidences = confidences.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    return predictions, confidences, probabilities


def compute_metrics(labels, predictions, probabilities):
    """Compute all evaluation metrics."""
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # False positive rate for Safe class (class 0)
    # FPR = FP / (FP + TN)
    # For Safe class: FP = sum of column 0 excluding diagonal
    safe_fp = cm[:, 0].sum() - cm[0, 0]  # False positives for Safe
    safe_tn_fp = cm[:, 0].sum()  # All predicted as Safe
    false_positive_rate = safe_fp / safe_tn_fp if safe_tn_fp > 0 else 0.0
    
    metrics = {
        'overall_accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'false_positive_rate': false_positive_rate
    }
    
    return metrics


def print_results(metrics):
    """Print evaluation results in paper format."""
    class_names = ['Safe', 'Low Risk', 'Medium Risk', 'High Risk', 'Critical']
    
    print("\n" + "=" * 80)
    print("ZKML-Guard Evaluation Results")
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"Paper result: 0.947")
    
    print(f"\nFalse Positive Rate (Safe class): {metrics['false_positive_rate']:.4f}")
    print(f"Paper result: 0.0028")
    
    print("\n" + "-" * 80)
    print("Per-Class Performance")
    print("-" * 80)
    print(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for i, name in enumerate(class_names):
        print(f"{name:<15} "
              f"{metrics['precision'][i]:>11.3f} "
              f"{metrics['recall'][i]:>11.3f} "
              f"{metrics['f1_score'][i]:>11.3f} "
              f"{metrics['support'][i]:>9}")
    
    print("-" * 80)
    
    # Compare with paper results
    print("\nComparison with Paper Results:")
    paper_results = {
        'Safe': {'precision': 0.962, 'recall': 0.978},
        'Low Risk': {'precision': 0.914, 'recall': 0.892},
        'Medium Risk': {'precision': 0.887, 'recall': 0.853},
        'High Risk': {'precision': 0.936, 'recall': 0.901},
        'Critical': {'precision': 0.973, 'recall': 0.918}
    }
    
    for i, name in enumerate(class_names):
        paper = paper_results[name]
        print(f"{name}:")
        print(f"  Precision: {metrics['precision'][i]:.3f} (Paper: {paper['precision']:.3f})")
        print(f"  Recall:    {metrics['recall'][i]:.3f} (Paper: {paper['recall']:.3f})")


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {output_path}")
    plt.close()


def plot_precision_recall(metrics, output_path):
    """Plot precision and recall by category."""
    class_names = ['Safe', 'Low Risk', 'Medium Risk', 'High Risk', 'Critical']
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    precision = [p * 100 for p in metrics['precision']]
    recall = [r * 100 for r in metrics['recall']]
    
    bars1 = ax.bar(x - width/2, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x + width/2, recall, width, label='Recall', alpha=0.8)
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Precision and Recall by Risk Category')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([80, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall plot saved to {output_path}")
    plt.close()


def analyze_confidence(predictions, labels, confidences):
    """Analyze model confidence scores."""
    print("\n" + "-" * 80)
    print("Confidence Analysis")
    print("-" * 80)
    
    correct = predictions == labels
    
    print(f"\nCorrect predictions:")
    print(f"  Mean confidence: {confidences[correct].mean():.3f}")
    print(f"  Median confidence: {np.median(confidences[correct]):.3f}")
    
    print(f"\nIncorrect predictions:")
    if (~correct).sum() > 0:
        print(f"  Mean confidence: {confidences[~correct].mean():.3f}")
        print(f"  Median confidence: {np.median(confidences[~correct]):.3f}")
    else:
        print("  No incorrect predictions!")
    
    # Confidence by class
    print(f"\nConfidence by predicted class:")
    class_names = ['Safe', 'Low Risk', 'Medium Risk', 'High Risk', 'Critical']
    for i, name in enumerate(class_names):
        mask = predictions == i
        if mask.sum() > 0:
            print(f"  {name:<15}: {confidences[mask].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ZKML-Guard model")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth)')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = ZKMLGuardModel.load_pretrained(args.model).to(device)
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    features, labels = load_test_data(Path(args.test_data))
    
    # Evaluate
    print("\nEvaluating...")
    predictions, confidences, probabilities = evaluate_model(model, features, labels, device)
    
    # Compute metrics
    metrics = compute_metrics(labels, predictions, probabilities)
    
    # Print results
    print_results(metrics)
    
    # Analyze confidence
    analyze_confidence(predictions, labels, confidences)
    
    # Save metrics
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Generate plots
    class_names = ['Safe', 'Low Risk', 'Medium Risk', 'High Risk', 'Critical']
    
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')
    plot_precision_recall(metrics, output_dir / 'precision_recall.png')
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
