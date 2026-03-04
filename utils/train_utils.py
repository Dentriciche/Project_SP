# training_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from focal_loss import FocalLoss


# ============================================================================
# CLASS WEIGHTS
# ============================================================================

def compute_balanced_class_weights(labels, K, device):
    """
    Compute balanced class weights for imbalanced datasets.
    
    Args:
        labels: Training labels (numpy array or list)
        K: Number of classes
        device: torch device
    
    Returns:
        torch.Tensor: Class weights on specified device
    """
    class_weights = compute_class_weight('balanced', classes=np.arange(K), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    return class_weights


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def get_loss_function(loss_type='cross_entropy', class_weights=None, **kwargs):
    """
    Get loss function by name.
    
    Args:
        loss_type: 'cross_entropy', 'focal', 'label_smoothing_ce'
        class_weights: Optional class weights tensor
        **kwargs: Loss-specific parameters
            For cross_entropy: (no extra params)
            For label_smoothing_ce: label_smoothing (float)
            For focal: alpha, gamma
    
    Returns:
        Loss function
    
    Examples:
        # Standard cross-entropy
        criterion = get_loss_function('cross_entropy', class_weights=weights)
        
        # With label smoothing
        criterion = get_loss_function('label_smoothing_ce', class_weights=weights, 
                                     label_smoothing=0.1)
        
        # Focal loss
        criterion = get_loss_function('focal', class_weights=weights, 
                                     alpha=0.25, gamma=2.0)
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'label_smoothing_ce':
        label_smoothing = kwargs.get('label_smoothing', 0.1)
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    elif loss_type == 'focal':
        # You'd need to implement FocalLoss class separately
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma, task_type="multi-class", num_classes=31)
    
        raise NotImplementedError("Focal loss requires custom implementation")
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                        f"Choose from: 'cross_entropy', 'label_smoothing_ce', 'focal'")


# ============================================================================
# OPTIMIZERS
# ============================================================================

def get_optimizer(optimizer_type, model_parameters, lr, **kwargs):
    """
    Get optimizer by name.
    
    Args:
        optimizer_type: 'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'
        model_parameters: model.parameters()
        lr: Learning rate
        **kwargs: Optimizer-specific parameters
            For adam/adamw: weight_decay, betas, eps
            For sgd: momentum, weight_decay, nesterov
            For rmsprop: alpha, weight_decay, momentum
    
    Returns:
        Optimizer
    
    Examples:
        # AdamW with weight decay
        optimizer = get_optimizer('adamw', model.parameters(), lr=1e-4, 
                                 weight_decay=0.01)
        
        # SGD with momentum
        optimizer = get_optimizer('sgd', model.parameters(), lr=0.01, 
                                 momentum=0.9, weight_decay=1e-4)
        
        # Adam (standard)
        optimizer = get_optimizer('adam', model.parameters(), lr=0.001)
    """
    if optimizer_type == 'adam':
        return optim.Adam(
            model_parameters,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    
    elif optimizer_type == 'adamw':
        return optim.AdamW(
            model_parameters,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.01)
        )
    
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model_parameters,
            lr=lr,
            momentum=kwargs.get('momentum', 0.0),
            weight_decay=kwargs.get('weight_decay', 0.0),
            nesterov=kwargs.get('nesterov', False)
        )
    
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            model_parameters,
            lr=lr,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0),
            momentum=kwargs.get('momentum', 0.0)
        )
    
    elif optimizer_type == 'adagrad':
        return optim.Adagrad(
            model_parameters,
            lr=lr,
            lr_decay=kwargs.get('lr_decay', 0.0),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}. "
                        f"Choose from: 'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'")


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

def get_scheduler(scheduler_type, optimizer, **kwargs):
    """
    Get learning rate scheduler by name.
    
    Args:
        scheduler_type: 'onecycle', 'reduce_on_plateau', 'cosine', 'step', None
        optimizer: Optimizer instance
        **kwargs: Scheduler-specific parameters
            For onecycle: max_lr, epochs, steps_per_epoch, pct_start, etc.
            For reduce_on_plateau: mode, factor, patience, verbose
            For cosine: T_max, eta_min
            For step: step_size, gamma
    
    Returns:
        Scheduler or None
    
    Examples:
        # OneCycleLR (for transformers)
        scheduler = get_scheduler('onecycle', optimizer, 
                                  max_lr=5e-4, epochs=50, 
                                  steps_per_epoch=len(train_loader))
        
        # ReduceLROnPlateau (for RNNs)
        scheduler = get_scheduler('reduce_on_plateau', optimizer, 
                                  mode='min', factor=0.5, patience=3)
        
        # No scheduler
        scheduler = get_scheduler(None, optimizer)
    """
    if scheduler_type is None or scheduler_type == 'none':
        return None
    
    elif scheduler_type == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            epochs=kwargs.get('epochs', 50),
            steps_per_epoch=kwargs['steps_per_epoch'],  # Required!
            pct_start=kwargs.get('pct_start', 0.1),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos'),
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 10000.0)
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 3),
            verbose=kwargs.get('verbose', True),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}. "
                        f"Choose from: 'onecycle', 'reduce_on_plateau', 'cosine', 'step', None")


# ============================================================================
# CONVENIENCE FUNCTION (OPTIONAL)
# ============================================================================

def create_training_components(model, train_labels, K, device,
                               optimizer_type='adam', lr=1e-3,
                               loss_type='cross_entropy',
                               scheduler_type=None,
                               use_class_weights=True,
                               **kwargs):
    """
    One-stop factory to create all training components with full flexibility.
    
    Args:
        model: Model instance
        train_labels: Training labels for class weight computation
        K: Number of classes
        device: torch device
        optimizer_type: 'adam', 'adamw', 'sgd', etc.
        lr: Learning rate
        loss_type: 'cross_entropy', 'label_smoothing_ce', etc.
        scheduler_type: 'onecycle', 'reduce_on_plateau', etc., or None
        use_class_weights: Whether to compute and use class weights
        **kwargs: Additional parameters for optimizer/loss/scheduler
    
    Returns:
        dict with 'criterion', 'optimizer', 'scheduler', 'class_weights'
    
    Examples:
        # BERT with AdamW + OneCycleLR + Label Smoothing
        components = create_training_components(
            model=bert_model,
            train_labels=train_labels,
            K=31,
            device=device,
            optimizer_type='adamw',
            lr=1e-4,
            loss_type='label_smoothing_ce',
            scheduler_type='onecycle',
            weight_decay=0.01,
            label_smoothing=0.1,
            max_lr=5e-4,
            epochs=50,
            steps_per_epoch=len(train_loader)
        )
        
        # RNN with Adam + ReduceLROnPlateau
        components = create_training_components(
            model=rnn_model,
            train_labels=train_labels,
            K=31,
            device=device,
            optimizer_type='adam',
            lr=0.003,
            loss_type='cross_entropy',
            scheduler_type='reduce_on_plateau',
            weight_decay=1e-4,
            patience=3
        )
        
        # Experiment: BERT with SGD + Cosine Annealing
        components = create_training_components(
            model=bert_model,
            train_labels=train_labels,
            K=31,
            device=device,
            optimizer_type='sgd',
            lr=0.01,
            loss_type='cross_entropy',
            scheduler_type='cosine',
            momentum=0.9,
            weight_decay=1e-4,
            T_max=50
        )
    """
    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        class_weights = compute_balanced_class_weights(train_labels, K, device)
    
    # Get loss function
    criterion = get_loss_function(loss_type, class_weights=class_weights, **kwargs)
    
    # Get optimizer
    optimizer = get_optimizer(optimizer_type, model.parameters(), lr, **kwargs)
    
    # Get scheduler
    scheduler = get_scheduler(scheduler_type, optimizer, **kwargs)
    
    return {
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'class_weights': class_weights
    }
    
    
    