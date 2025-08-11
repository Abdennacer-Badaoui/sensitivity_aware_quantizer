import torch
import torch.nn as nn


def perplexity(model, tokenizer, eval_data, num_samples, device):
    """Evaluate model performance using perplexity on evaluation samples."""
    try:
        # Prepare evaluation texts
        eval_texts = []
        for i in range(min(num_samples, len(eval_data["input_ids"]))):
            text = tokenizer.decode(eval_data["input_ids"][i], skip_special_tokens=True)
            if len(text.strip()) > 0:
                eval_texts.append(text)

        if not eval_texts:
            raise ValueError("No valid evaluation texts were found.")

        # Tokenize texts
        encodings = tokenizer(
            eval_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # Initialize loss function
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        total_loss = 0
        total_tokens = 0

        batch_size = 1  # Process one text at a time for stability
        for start_index in range(0, len(encoded_texts), batch_size):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            with torch.no_grad():
                outputs = model(encoded_batch, attention_mask=attn_mask)
                logits = outputs.logits

            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = encoded_batch[..., 1:].contiguous()
            shift_attention_mask = attn_mask[..., 1:].contiguous()

            # Calculate loss
            loss = (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask
            )
            total_loss += loss.sum().item()
            total_tokens += shift_attention_mask.sum().item()

        # Calculate final perplexity
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
        else:
            perplexity = float("inf")

        return perplexity

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return float("inf")


def get_model_size_mb(model):
    """Calculate model size in MB considering mixed precision and quantized layers.

    Automatically detects:
    - Standard parameter dtypes (float32, float16, bfloat16, etc.)
    - Custom quantized layers (W4A16, W8A16, etc.)
    - Packed weight representations

    Args:
        model: PyTorch model with potentially mixed precision layers

    Returns:
        total_size_mb: model size in MB
    """
    import torch

    def get_dtype_bits(dtype):
        """Map PyTorch dtypes to their bit sizes"""
        dtype_bits = {
            torch.float32: 32,
            torch.float: 32,
            torch.float64: 64,
            torch.double: 64,
            torch.float16: 16,
            torch.half: 16,
            torch.bfloat16: 16,
            torch.int32: 32,
            torch.int: 32,
            torch.int64: 64,
            torch.long: 64,
            torch.int16: 16,
            torch.short: 16,
            torch.int8: 8,
            torch.uint8: 8,
            torch.bool: 1,
        }
        return dtype_bits.get(dtype, 32)  # Default to 32 if unknown

    def detect_quantized_layer_info(module):
        """Detect quantized layer type and return bit width info"""
        module_name = module.__class__.__name__

        # Check for common quantized layer patterns
        if "W4A16LinearLayer" in module_name:
            return 4, "W4A16LinearLayer"
        elif "W8A16LinearLayer" in module_name:
            return 8, "W8A16LinearLayer"
        elif "W16A16LinearLayer" in module_name:
            return 16, "W16A16LinearLayer"

        return None, None

    # Count parameters and buffers at the top level only to avoid double counting
    param_size = 0
    buffer_size = 0
    layer_breakdown = {}
    quantized_layers = {}
    counted_params = set()
    counted_buffers = set()

    # First, collect all parameter and buffer names to avoid double counting
    for name, param in model.named_parameters():
        if id(param) not in counted_params:
            bits = get_dtype_bits(param.dtype)
            size = param.nelement() * (bits / 8)
            param_size += size
            counted_params.add(id(param))

    for name, buffer in model.named_buffers():
        if id(buffer) not in counted_buffers:
            size = buffer.nelement() * buffer.element_size()
            buffer_size += size
            counted_buffers.add(id(buffer))

    # Now analyze layer-by-layer for breakdown (but don't add to total again)
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:  
            continue

        # Check if this is a quantized layer
        weight_bits, quant_type = detect_quantized_layer_info(module)

        if weight_bits is not None:
            # Handle quantized layers
            quantized_layers[name] = quant_type

            module_param_size = 0
            module_buffer_size = 0

            for param_name, param in module.named_parameters():
                bits = get_dtype_bits(param.dtype)
                size = param.nelement() * (bits / 8)
                module_param_size += size

            for buffer_name, buffer in module.named_buffers():
                if "weight" in buffer_name.lower() or "packed" in buffer_name.lower():
                    if "packed" in buffer_name.lower():
                        size = buffer.nelement() * buffer.element_size()
                    else:
                        size = buffer.nelement() * buffer.element_size()
                else:
                    size = buffer.nelement() * buffer.element_size()
                module_buffer_size += size

            layer_breakdown[name] = {
                "type": quant_type,
                "param_size": module_param_size,
                "buffer_size": module_buffer_size,
                "total_size": module_param_size + module_buffer_size,
            }

        else:
            # Handle standard (non-quantized) layers
            module_param_size = 0
            module_buffer_size = 0

            for param_name, param in module.named_parameters():
                bits = get_dtype_bits(param.dtype)
                size = param.nelement() * (bits / 8)
                module_param_size += size

            for buffer_name, buffer in module.named_buffers():
                size = buffer.nelement() * buffer.element_size()
                module_buffer_size += size

            if module_param_size > 0 or module_buffer_size > 0:
                layer_breakdown[name] = {
                    "type": f"Standard ({module.__class__.__name__})",
                    "param_size": module_param_size,
                    "buffer_size": module_buffer_size,
                    "total_size": module_param_size + module_buffer_size,
                }

    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024**2)
    return total_size_mb