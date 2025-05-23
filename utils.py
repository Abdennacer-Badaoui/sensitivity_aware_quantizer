import json


def get_model_size_mb(model, bit_config=None):
    """Calculate model size in MB considering mixed precision quantization.

    Args:
        model: PyTorch model.
        bit_config: Dict mapping layer names to their quantization bits.
                    Layers are matched by prefix (e.g., 'layer1' matches 'layer1.weight').
    """
    param_size = 0
    sorted_layers = []
    if bit_config:
        # Sort layers by descending length to prioritize specific prefixes first
        sorted_layers = sorted(bit_config.keys(), key=lambda x: (-len(x), x))

    for name, param in model.named_parameters():
        bits = 32  # Default to 32-bit (unquantized)

        # Check if parameter belongs to a quantized layer
        for layer_name in sorted_layers:
            if name == layer_name or name.startswith(f"{layer_name}."):
                bits = bit_config[layer_name]
                break

        # Calculate size in bytes (bits/8)
        param_size += param.nelement() * (bits / 8)

    # Calculate buffer size (always full precision)
    buffer_size = sum(
        buffer.nelement() * buffer.element_size() for buffer in model.buffers()
    )

    total_size_bytes = param_size + buffer_size
    return total_size_bytes / (1024**2)  # Convert to MB


def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
