# pylint: disable=import-error, possibly-used-before-assignment
"""
Evaluation utilities for PyTorch multiclass classification models.

This module provides reusable functions to evaluate PyTorch models on 
classification tasks, including metrics calculation, confusion matrix 
visualization, and model comparison utilities.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Optional
import pandas as pd


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    Evaluate a PyTorch model on a dataset.
    
    Args:
        model: The PyTorch model to evaluate
        data_loader: DataLoader containing the evaluation dataset
        device: Device to run evaluation on (cuda/cpu)
        class_names: Optional list of class names for reporting
        return_predictions: If True, return predictions and true labels
        
    Returns:
        Dictionary containing evaluation metrics (accuracy, precision, recall, F1)
        If return_predictions=True, also returns (predictions, true_labels)
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:  # EmbeddingBag format: (text, offsets, labels)
                    text, offsets, labels = batch
                    text = text.to(device)
                    offsets = offsets.to(device)
                    labels = labels.to(device)
                    outputs = model(text, offsets)
                elif len(batch) == 2:  # Standard format: (text, labels)
                    text, labels = batch
                    text = text.to(device)
                    labels = labels.to(device)
                    outputs = model(text)
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics (macro-averaged for multiclass)
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_predictions, average='macro', zero_division=0),
        'f1_score': f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    }
    
    if return_predictions:
        return metrics, all_predictions, all_labels
    
    return metrics

def load_model_from_checkpoint(
    filepath: str,
    model_class: type,
    device: torch.device = None
) -> Tuple[nn.Module, Dict]:
    """
    Load a PyTorch model from a checkpoint file.
    
    This function handles the checkpoint format used in the baseline notebook,
    which includes model weights, configuration, and metadata.
    
    Args:
        filepath: Path to checkpoint file (.pkl or .pth)
        model_class: The model class to instantiate (e.g., CharacterClassifier)
        device: Device to load model on (defaults to CPU)
        
    Returns:
        Tuple of (loaded_model, checkpoint_dict)
        
    Example:
        >>> from models import CharacterClassifier
        >>> model, checkpoint = load_model_from_checkpoint(
        ...     'baseline_model_k12.pt',
        ...     CharacterClassifier,
        ...     device=torch.device('cuda')
        ... )
        >>> # Now use with evaluate_model()
        >>> metrics = evaluate_model(model, test_loader, device)
    """
    if device is None:
        device = torch.device('cpu')
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Validate checkpoint format
    required_keys = ['model_state_dict', 'config']
    if not all(key in checkpoint for key in required_keys):
        raise ValueError(
            f"Invalid checkpoint format. Expected keys: {required_keys}, "
            f"Found: {list(checkpoint.keys())}"
        )
    
    # Extract config
    config = checkpoint['config']
    
    # Reconstruct model
    model = model_class(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_classes=config['K'],
        num_hidden_layers=config.get('num_hidden_layers', 0),
        hidden_dim=config.get('hidden_dim', 128),
        dropout_rate=config.get('dropout_rate', 0.5)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def evaluate_saved_model(
    checkpoint_path: str,
    model_class: type,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Convenience function to load and evaluate a saved model in one step.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        model_class: Model class to instantiate
        data_loader: DataLoader with evaluation data
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path, 
        model_class, 
        device
    )
    
    # Get class names if available
    class_names = None
    if 'label_to_char' in checkpoint:
        label_to_char = checkpoint['label_to_char']
        class_names = [label_to_char[i] for i in range(len(label_to_char))]
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        class_names=class_names
    )
    
    return metrics

def print_evaluation_report(
    metrics: Dict[str, float],
    model_name: str = "Model"
) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model being evaluated
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (macro)")
    print(f"Recall:    {metrics['recall']:.4f} (macro)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} (macro)")
    print(f"{'='*50}\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a confusion matrix with annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Title for the plot
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def get_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> pd.DataFrame:
    """
    Calculate per-class precision, recall, and F1-score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        DataFrame with per-class metrics
    """
    # Calculate per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Count support (number of samples per class)
    support = np.bincount(y_true, minlength=len(class_names))

    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    return metrics_df


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> None:
    """
    Print a detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\nDetailed Classification Report:")
    print("="*70)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    metric: str = 'f1_score',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models based on their evaluation metrics.

    Args:
        results_dict: Dictionary mapping model names to their metrics
        metric: Primary metric to sort by (default: 'f1_score')
        save_path: Optional path to save comparison plot

    Returns:
        DataFrame with model comparison
    """
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_dict).T
    comparison_df = comparison_df.sort_values(by=metric, ascending=False)

    # Print comparison table
    print("\nModel Comparison:")
    print("="*70)
    print(comparison_df.to_string())
    print("="*70)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        comparison_df[metric_name].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return comparison_df

