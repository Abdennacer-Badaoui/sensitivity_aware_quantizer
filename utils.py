import torch
import json
import os

def get_model_size_mb(model, bit_config=None):
    """Calculate model size in MB.
    Args:
        model: The PyTorch model
        bit_config: Dictionary mapping layer names to their bit precision (e.g., {"layer1": 8, "layer2": 4})
    """
    param_size = 0
    for name, param in model.named_parameters():
        # Find the matching layer name in bit_config
        matching_layer = None
        for layer_name in bit_config.keys() if bit_config else []:
            if layer_name in name:  # Check if layer_name is a substring of the parameter name
                matching_layer = layer_name
                break
        
        if matching_layer and bit_config:
            # For quantized layers, use the specified bit precision
            bits = bit_config[matching_layer]
            param_size += param.nelement() * (bits / 8)  # Convert bits to bytes
        else:
            # For non-quantized layers, use default 32-bit precision
            param_size += param.nelement() * 4  # 4 bytes for 32-bit float
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f) 