import torch
import torch.nn as nn
from typing import Dict, Tuple

def format_params(num: int) -> str:
    """Format parameter count in terms of K (1000)"""
    if num >= 1000:
        return f"{num/1000:.2f}K"
    return str(num)

def count_parameters(model: nn.Module, verbose: bool = False) -> Dict[str, float]:
    """
    Count total, classifier, and non-classifier trainable parameters in a PyTorch model.
    Returns values in terms of K (1000) parameters.
    
    Args:
        model: PyTorch model
        verbose: If True, print parameter counts for each layer
        
    Returns:
        Dictionary containing parameter counts in K (1000s)
    """
    def is_classifier_layer(name: str) -> bool:
        """Check if the layer is part of classifier based on common naming patterns"""
        classifier_keywords = ['classifier', 'fc', 'linear', 'head']
        return any(keyword in name.lower() for keyword in classifier_keywords)
    
    total_params = 0
    classifier_params = 0
    non_classifier_params = 0
    
    # Iterate through all parameters
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            total_params += param_count
            
            if is_classifier_layer(name):
                classifier_params += param_count
            else:
                non_classifier_params += param_count
                
            if verbose:
                print(f"{name}: {format_params(param_count)} parameters "
                      f"{'(Classifier)' if is_classifier_layer(name) else '(Non-classifier)'}")
    
    results = {
        'total_trainable_params': total_params / 1000,  # Convert to K
        'classifier_params': classifier_params / 1000,   # Convert to K
        'non_classifier_params': non_classifier_params / 1000  # Convert to K
    }
    
    print("\nSummary:")
    print(f"Total trainable parameters (K): {format_params(total_params)}")
    print(f"Classifier parameters (K): {format_params(classifier_params)}")
    print(f"Non-classifier parameters (K): {format_params(non_classifier_params)}")
    print(f"Classifier parameters percentage (K): {(classifier_params/total_params)*100:.2f}%")
    
    return results
