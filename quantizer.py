import torch
import torch.nn as nn
import torch.nn.functional as F


class W4A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()

        # Pack two 4-bit weights into each uint8 byte
        # If in_features is odd, we'll pad with one extra weight
        self.in_features = in_features
        self.out_features = out_features
        packed_in_features = (in_features + 1) // 2

        self.register_buffer(
            "packed_weights",
            torch.randint(
                0, 255, (out_features, packed_in_features), dtype=torch.uint8
            ),
        )

        # Asymmetric quantization parameters
        self.register_buffer("scales", torch.randn((out_features,), dtype=dtype))
        self.register_buffer(
            "zero_points", torch.randint(0, 15, (out_features,), dtype=torch.uint8)
        )

        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weights):
        """Asymmetric quantization of weights to 4-bit with packing"""
        w_fp32 = weights.clone().to(torch.float32)

        # Find min and max values per output channel
        w_min = w_fp32.min(dim=-1).values
        w_max = w_fp32.max(dim=-1).values

        # Calculate scale and zero point for asymmetric quantization
        # Range for 4-bit: [0, 15]
        scales = (w_max - w_min) / 15.0
        zero_points = torch.round(-w_min / scales).clamp(0, 15)

        # Quantize weights
        int4_weights = torch.round(
            w_fp32 / scales.unsqueeze(1) + zero_points.unsqueeze(1)
        )
        int4_weights = int4_weights.clamp(0, 15).to(torch.uint8)

        # Pack weights: combine pairs of 4-bit weights into single uint8
        packed_weights = self._pack_weights(int4_weights)

        # Update buffers
        self.packed_weights = packed_weights
        self.scales = scales.to(weights.dtype)
        self.zero_points = zero_points

    def _pack_weights(self, int4_weights):
        """Pack two 4-bit weights into each uint8 byte"""
        out_features, in_features = int4_weights.shape

        # Pad with zeros if in_features is odd
        if in_features % 2 == 1:
            padding = torch.zeros(
                out_features, 1, dtype=torch.uint8, device=int4_weights.device
            )
            int4_weights = torch.cat([int4_weights, padding], dim=1)
            in_features += 1

        # Reshape to group pairs of weights
        int4_weights = int4_weights.view(out_features, in_features // 2, 2)

        # Pack: first weight in lower 4 bits, second weight in upper 4 bits
        packed = (int4_weights[:, :, 1] << 4) | int4_weights[:, :, 0]

        return packed

    def _unpack_weights(self, packed_weights):
        """Unpack uint8 bytes back to pairs of 4-bit weights"""
        out_features, packed_in_features = packed_weights.shape

        # Extract lower and upper 4 bits
        lower_4bits = packed_weights & 0x0F  # Extract lower 4 bits
        upper_4bits = (packed_weights >> 4) & 0x0F  # Extract upper 4 bits

        # Interleave to restore original order
        unpacked = torch.stack([lower_4bits, upper_4bits], dim=2)
        unpacked = unpacked.view(out_features, packed_in_features * 2)

        # Trim to original size if we padded
        if unpacked.shape[1] > self.in_features:
            unpacked = unpacked[:, : self.in_features]

        return unpacked

    def dequantize_weights(self, input_dtype):
        """Dequantize packed 4-bit weights back to floating point"""
        # Unpack the weights first
        unpacked_weights = self._unpack_weights(self.packed_weights)

        # Convert to input dtype and apply asymmetric dequantization
        dequantized = (
            unpacked_weights.to(input_dtype)
            - self.zero_points.unsqueeze(1).to(input_dtype)
        ) * self.scales.unsqueeze(1)
        return dequantized

    def forward(self, input):
        # Dequantize weights on-the-fly
        dequantized_weights = self.dequantize_weights(input.dtype)

        # Standard linear operation
        output = F.linear(input, dequantized_weights)

        if self.bias is not None:
            output = output + self.bias

        return output
    
class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.register_buffer(
            "int8_weights",
            torch.randint(0, 255, (out_features, in_features), dtype=torch.uint8),  # 0-255 for asymmetric
        )
        self.register_buffer("scales", torch.randn((out_features), dtype=dtype))
        self.register_buffer("zero_points", torch.randint(0, 255, (out_features,), dtype=torch.uint8))  # Zero points for asymmetric
        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weights):
        """
        Asymmetric quantization: maps [min_val, max_val] to [0, 255]
        """
        w_fp32 = weights.clone().to(torch.float32)
        
        # Find min and max values per output channel (row-wise)
        min_vals = w_fp32.min(dim=-1).values  # Shape: (out_features,)
        max_vals = w_fp32.max(dim=-1).values  # Shape: (out_features,)
        
        # Calculate scales: (max - min) / 255
        scales = (max_vals - min_vals) / 255.0
        scales = torch.clamp(scales, min=1e-8)  # Prevent division by zero
        scales = scales.to(weights.dtype)
        
        # Calculate zero points: -min_val / scale
        zero_points = (-min_vals / scales).round().clamp(0, 255)
        zero_points = zero_points.to(torch.uint8)
        
        # Quantize: q = round(x / scale + zero_point)
        int8_weights = torch.round(w_fp32 / scales.unsqueeze(1) + zero_points.unsqueeze(1))
        int8_weights = torch.clamp(int8_weights, 0, 255).to(torch.uint8)
        
        # Update buffers
        self.int8_weights = int8_weights
        self.scales = scales
        self.zero_points = zero_points

    def forward(self, input):
        # Dequantize: x = scale * (q - zero_point)
        casted_weights = self.int8_weights.to(input.dtype)
        zero_points_casted = self.zero_points.to(input.dtype)
        
        # Dequantize weights
        dequantized_weights = self.scales.unsqueeze(1) * (casted_weights - zero_points_casted.unsqueeze(1))
        
        # Perform linear operation
        output = F.linear(input, dequantized_weights)
        
        if self.bias is not None:
            output = output + self.bias
        return output


class W16A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.bfloat16):
        super().__init__()

        self.register_buffer(
            "bfloat16_weights",
            torch.zeros((out_features, in_features), dtype=torch.bfloat16),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weights):
        # Convert to bfloat16
        self.bfloat16_weights = weights.to(torch.bfloat16)

    def forward(self, input):
        casted_weights = self.bfloat16_weights.to(input.dtype)
        output = F.linear(input, casted_weights)

        if self.bias is not None:
            output = output + self.bias
        return output


def replace_linear_with_target_and_quantize(
    module, target_class, module_name_to_exclude
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any(
            [x == name for x in module_name_to_exclude]
        ):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(
                child.in_features,
                child.out_features,
                old_bias is not None,
                child.weight.dtype,
            )
            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)

            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(
                child, target_class, module_name_to_exclude
            )


def replace_single_linear_with_target(model, target_class, layer_to_replace_name):
    """
    Replace a specific layer in the model with the target quantized class.

    Args:
        model (nn.Module): The original model.
        target_class (nn.Module): The quantized replacement class (e.g., W8A16LinearLayer).
        layer_to_replace_name (str): Full dot-separated name of the layer to replace
                                     (e.g., "encoder.layers.3.linear1").

    Returns:
        nn.Module: The modified model with the specified layer replaced.
    """

    def get_submodule_and_last_name(model, layer_name):
        names = layer_name.split(".")
        submodule = model
        for name in names[:-1]:
            submodule = getattr(submodule, name)
        return submodule, names[-1]

    parent_module, last_name = get_submodule_and_last_name(model, layer_to_replace_name)
    original_layer = getattr(parent_module, last_name)

    # Create new quantized layer with same config
    new_layer = target_class(
        original_layer.in_features,
        original_layer.out_features,
        original_layer.bias is not None,
        original_layer.weight.dtype,
    )
    new_layer.quantize(original_layer.weight)

    if original_layer.bias is not None:
        new_layer.bias = original_layer.bias

    setattr(parent_module, last_name, new_layer)


def dequantize_model_to_standard_format(model, target_dtype=torch.float32):
    """
    Convert a quantized model back to standard linear layers with dequantized weights (in-place).

    Args:
        model (nn.Module): The quantized model containing W4A16LinearLayer, W8A16LinearLayer,
                          or W16A16LinearLayer instances
        target_dtype (torch.dtype): Target dtype for the dequantized weights (default: torch.float32)

    Returns:
        nn.Module: The same model instance with quantized layers replaced by standard nn.Linear layers
    """

    def convert_and_replace(module):
        """Convert a single quantized layer to standard linear layer and replace it"""
        if isinstance(module, W4A16LinearLayer):
            # Dequantize 4-bit weights
            dequantized_weights = module.dequantize_weights(target_dtype)

            # Create standard linear layer
            linear_layer = nn.Linear(
                module.in_features, module.out_features, bias=module.bias is not None
            )

            # Set dequantized weights
            linear_layer.weight = nn.Parameter(dequantized_weights)

            # Set bias if exists
            if module.bias is not None:
                linear_layer.bias = nn.Parameter(
                    module.bias.squeeze(0).to(target_dtype)
                )

            return linear_layer

        elif isinstance(module, W8A16LinearLayer):
            # Dequantize 8-bit weights
            dequantized_weights = module.int8_weights.to(
                target_dtype
            ) * module.scales.unsqueeze(1).to(target_dtype)

            # Create standard linear layer
            linear_layer = nn.Linear(
                module.int8_weights.shape[1],
                module.int8_weights.shape[0],
                bias=module.bias is not None,
            )

            # Set dequantized weights
            linear_layer.weight = nn.Parameter(dequantized_weights)

            # Set bias if exists
            if module.bias is not None:
                linear_layer.bias = nn.Parameter(
                    module.bias.squeeze(0).to(target_dtype)
                )

            return linear_layer

        elif isinstance(module, W16A16LinearLayer):
            # Convert bfloat16 weights to target dtype
            dequantized_weights = module.bfloat16_weights.to(target_dtype)

            # Create standard linear layer
            linear_layer = nn.Linear(
                module.bfloat16_weights.shape[1],
                module.bfloat16_weights.shape[0],
                bias=module.bias is not None,
            )

            # Set weights
            linear_layer.weight = nn.Parameter(dequantized_weights)

            # Set bias if exists
            if module.bias is not None:
                linear_layer.bias = nn.Parameter(
                    module.bias.squeeze(0).to(target_dtype)
                )

            return linear_layer

        return None

    # Process the model in-place
    for name, module in list(model.named_children()):
        # Check if this module is a quantized layer
        converted = convert_and_replace(module)
        if converted is not None:
            setattr(model, name, converted)
        else:
            # Recursively process child modules
            dequantize_model_to_standard_format(module, target_dtype)

    return model
