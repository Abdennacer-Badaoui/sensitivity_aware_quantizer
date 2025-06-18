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
            baseline_probs = baseline_probs / baseline_probs.sum(dim=-1, keepdim=True)
            quantized_probs = quantized_probs / quantized_probs.sum(
                dim=-1, keepdim=True
            )

            # Calculate Jensen-Shannon Divergence
            m = 0.5 * (baseline_probs + quantized_probs)

            kl_p_m = torch.sum(baseline_probs * torch.log(baseline_probs / m), dim=-1)
            kl_q_m = torch.sum(quantized_probs * torch.log(quantized_probs / m), dim=-1)

            # JSD is the average of the two KL divergences, then averaged across all positions
            jsd = 0.5 * (kl_p_m + kl_q_m).mean().item()
            normalized_jsd = jsd / math.log(2)
            scores.append(normalized_jsd)

        sensitivity_score = np.mean(scores)
        return sensitivity_score

    @staticmethod
    def compute_hessian_based_sensitivities(model, tokenizer, weight_tensor, input_data, 
                                            device="cuda", max_iter=20, epsilon=1e-4):
        """
        Hessian eigenvalue approximation using finite differences with power iteration.
        
        Args:
            model: The neural network model
            tokenizer: Tokenizer for preparing labels
            weight_tensor: Weight tensor of the target layer
            input_data: Input data for Hessian calculation
            device: Device to use for computation
            max_iter: Number of power iterations
            epsilon: Finite difference step size
            
        Returns:
            float: Approximated top eigenvalue of the Hessian matrix
        """
        # Save original model state
        original_state = model.training
        model.eval()
        
        # Prepare data
        input_ids = input_data["input_ids"].to(device)
        attention_mask = input_data["attention_mask"].to(device)
        labels = input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        
        # Initialize random direction vector
        v = torch.randn_like(weight_tensor, device=device)
        v = v / torch.norm(v)  # Normalize to unit vector
        
        eigenvalue = None
        
        try:
            # Compute original gradient
            weight_tensor.requires_grad_(True)
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            grad_original = weight_tensor.grad.clone().detach()
            
            # Power iteration with finite differences
            for i in range(max_iter):
                # Perturb parameter in v direction
                with torch.no_grad():
                    original_weight = weight_tensor.data.clone()
                    weight_tensor.data += epsilon * v
                
                # Compute gradient at perturbed point
                model.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                grad_perturbed = weight_tensor.grad.clone().detach()
                
                # Restore original parameter
                with torch.no_grad():
                    weight_tensor.data.copy_(original_weight)
                
                # Approximate Hessian-vector product
                Hv = (grad_perturbed - grad_original) / epsilon
                
                # Update eigenvalue estimate (Rayleigh quotient)
                new_eigenvalue = torch.sum(v * Hv).item()
                
                # Check for convergence
                if eigenvalue is not None and abs(new_eigenvalue - eigenvalue) < 1e-6:
                    break
                    
                eigenvalue = new_eigenvalue
                
                # Update direction vector for next iteration
                v = Hv / (torch.norm(Hv) + 1e-8)
            
            return abs(eigenvalue) if eigenvalue is not None else 0.0
            
        except Exception as e:
            print(f"Error in finite difference approximation: {e}")
            return 0.0
        finally:
            # Restore original state
            model.train(original_state)
            weight_tensor.requires_grad_(False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        