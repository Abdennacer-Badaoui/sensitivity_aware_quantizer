import torch
import torch.nn as nn


def evaluate_model(model, tokenizer, eval_data, num_samples, device):
    """Evaluate model performance using perplexity on evaluation samples."""
    try:
        # Prepare evaluation texts
        eval_texts = []
        for i in range(min(num_samples, len(eval_data["input_ids"]))):
            text = tokenizer.decode(
                eval_data["input_ids"][i], skip_special_tokens=True
            )
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

        # Process in batches
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