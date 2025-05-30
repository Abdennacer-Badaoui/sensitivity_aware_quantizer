import torch
import torch.nn as nn
import torch.nn.functional as F


class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, 
                 bias=True, dtype=torch.float32):
        super().__init__()
        
        
        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128, 127, (out_features, in_features), dtype=torch.int8
            )
        )
        
        self.register_buffer("scales", 
                             torch.randn((out_features), dtype=dtype))
        
        if bias:
            self.register_buffer("bias", 
                                 torch.randn((1, out_features), 
                                             dtype=dtype))
        
        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights
                        /scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales
    
    def forward(self, input):
        casted_weights = self.int8_weights.to(input.dtype)
        output = F.linear(input, casted_weights) * self.scales
        
        if self.bias is not None:
            output = output + self.bias      
        return output    
    

class W16A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.bfloat16):
        super().__init__()
        
        self.register_buffer(
            "bfloat16_weights",
            torch.zeros((out_features, in_features), dtype=torch.bfloat16)
        )
        
        if bias:
            self.register_buffer("bias", 
                               torch.zeros((1, out_features), dtype=dtype))
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
    

def replace_linear_with_target_and_quantize(module, 
                               target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(child.in_features, 
                                      child.out_features, 
                                      old_bias is not None, 
                                      child.weight.dtype)
            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)
            
            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, 
                     target_class, module_name_to_exclude)
            
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
