import torch
import math
import numpy as np

class SensitivityMetrics:
    @staticmethod
    def compute_divergence_based_sensitivities(baseline_outputs, quantized_outputs):
        """Compute the divergence between baseline and quantized outputs."""
        scores = []
        for baseline_batch, quantized_batch in zip(baseline_outputs, quantized_outputs):  
            baseline_logits = baseline_batch["logits"]
            quantized_logits = quantized_batch["logits"]
            baseline_probs = torch.softmax(baseline_logits / 1.0, dim=-1)
            quantized_probs = torch.softmax(quantized_logits / 1.0, dim=-1)

            # Ensure probabilities are properly normalized and add small epsilon
            eps = 1e-10
            baseline_probs = baseline_probs + eps
            quantized_probs = quantized_probs + eps
            baseline_probs = baseline_probs / baseline_probs.sum(
                dim=-1, keepdim=True
            )
            quantized_probs = quantized_probs / quantized_probs.sum(
                dim=-1, keepdim=True
            )

            # Calculate Jensen-Shannon Divergence
            m = 0.5 * (baseline_probs + quantized_probs)

            kl_p_m = torch.sum(
                baseline_probs * torch.log(baseline_probs / m), dim=-1
            )
            kl_q_m = torch.sum(
                quantized_probs * torch.log(quantized_probs / m), dim=-1
            )

            # JSD is the average of the two KL divergences, then averaged across all positions
            jsd = 0.5 * (kl_p_m + kl_q_m).mean().item()
            normalized_jsd = jsd / math.log(2)
            scores.append(normalized_jsd)

        sensitivity_score = np.mean(scores)
        return sensitivity_score
    
    @staticmethod
    def compute_hessian_based_sensitivities(model, layer_name, layer_module, calibration_data):
        """Compute the trace of the Hessian diagonal for a specific layer using Fisher Information approximation."""

        # Set model to train mode for gradient computation
        was_training = model.training
        model.train()

        trace_sum = 0.0
        num_samples = 0

        # Use a smaller subset for Hessian computation to manage memory
        max_samples_for_hessian = min(20, len(calibration_data["input_ids"]))

        try:
            for i in range(0, max_samples_for_hessian, 1):  # Process one sample at a time
                batch_inputs = {
                    "input_ids": calibration_data["input_ids"][i:i+1],
                    "attention_mask": calibration_data["attention_mask"][i:i+1],
                }
                
                # Clear gradients
                model.zero_grad()
                
                # Forward pass
                outputs = model(**batch_inputs)
                logits = outputs.logits
                
                # Use log probabilities for numerical stability
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Sample multiple tokens from the distribution for better approximation
                seq_len = log_probs.size(1)
                samples_per_seq = min(5, seq_len)  # Sample a few positions per sequence
                
                for pos_idx in range(0, seq_len, max(1, seq_len // samples_per_seq)):
                    if pos_idx >= seq_len:
                        break
                        
                    # Clear gradients for this position
                    model.zero_grad()
                    
                    # Sample from categorical distribution at this position
                    probs = torch.softmax(logits[0, pos_idx], dim=0)
                    sampled_token = torch.multinomial(probs, 1)
                    
                    # Compute log probability of sampled token  
                    log_prob = log_probs[0, pos_idx, sampled_token]
                    
                    # Backward pass to compute gradients
                    log_prob.backward(retain_graph=True)
                    
                    # Get gradient for this layer
                    if layer_module.weight.grad is not None:
                        grad = layer_module.weight.grad.clone()
                        # Fisher Information approximation: trace of outer product g ⊗ g
                        trace_contribution = torch.sum(grad * grad).item()
                        trace_sum += trace_contribution
                        num_samples += 1
            
            # Normalize by number of samples
            hessian_trace = trace_sum / num_samples if num_samples > 0 else 0.0
            
            # Normalize by parameter count for fair comparison across layers
            param_count = layer_module.weight.numel()
            normalized_trace = hessian_trace / param_count if param_count > 0 else 0.0
            
            # Apply log scaling to compress the range
            sensitivity_score = math.log(1 + normalized_trace)
            
        except Exception as e:
            print(f"Warning: Hessian computation failed for {layer_name}: {str(e)}")
            sensitivity_score = 1.0  # Default sensitivity

        finally:
            # Restore original training mode
            model.train(was_training)
            # Clear any remaining gradients
            model.zero_grad()

        return sensitivity_score