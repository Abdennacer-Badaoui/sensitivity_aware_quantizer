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
    def compute_hessian_based_sensitivities(
        model,
        tokenizer,
        weight_tensor,
        calibration_data,
        batch_size,
        device="cuda",
        max_iter=20,
        epsilon=1e-4,
    ):
        """
        Hessian eigenvalue approximation using finite differences with power iteration.
        Args:
            model: The neural network model
            tokenizer: Tokenizer for preparing labels
            weight_tensor: Weight tensor of the target layer
            calibration_data: Input data for Hessian calculation
            device: Device to use for computation
            max_iter: Number of power iterations
            epsilon: Finite difference step size
        Returns:
            float: Approximated top eigenvalue of the Hessian matrix
        """

        # Save original model state
        original_state = model.training
        model.eval()

        # Helper function to compute Hessian-vector product for a batch
        def compute_hvp_batch(input_batch, attention_batch, labels_batch, v_direction):
            # Compute original gradient for this batch
            weight_tensor.requires_grad_(True)
            model.zero_grad()
            outputs = model(
                input_ids=input_batch,
                attention_mask=attention_batch,
                labels=labels_batch,
            )
            loss = outputs.loss
            loss.backward()
            grad_original = (
                weight_tensor.grad.clone().detach()
                if weight_tensor.grad is not None
                else torch.zeros_like(weight_tensor)
            )

            # Perturb parameter in v direction
            original_weight = weight_tensor.data.clone()
            weight_tensor.data += epsilon * v_direction

            # Compute gradient at perturbed point
            model.zero_grad()
            outputs = model(
                input_ids=input_batch,
                attention_mask=attention_batch,
                labels=labels_batch,
            )
            loss = outputs.loss
            loss.backward()
            grad_perturbed = (
                weight_tensor.grad.clone().detach()
                if weight_tensor.grad is not None
                else torch.zeros_like(weight_tensor)
            )

            # Restore original parameter
            weight_tensor.data.copy_(original_weight)
            weight_tensor.requires_grad_(False)

            # Approximate Hessian-vector product
            hvp = (grad_perturbed - grad_original) / epsilon
            return hvp

        # Initialize random direction vector for power iteration
        v = torch.randn_like(weight_tensor, device=device)
        v = v / torch.norm(v)  # Normalize to unit vector

        eigenvalue_prev = None

        # Power iteration
        for iteration in range(max_iter):
            # Accumulate Hessian-vector product across all batches
            Hv_accumulated = torch.zeros_like(weight_tensor, device=device)
            total_samples = 0

            for batch_idx in range(0, len(calibration_data["input_ids"]), batch_size):
                end_idx = min(
                    batch_idx + batch_size, len(calibration_data["input_ids"])
                )
                batch_size_actual = end_idx - batch_idx

                # Prepare data
                input_ids = calibration_data["input_ids"][batch_idx:end_idx].to(device)
                attention_mask = calibration_data["attention_mask"][
                    batch_idx:end_idx
                ].to(device)
                labels = input_ids.clone()

                if tokenizer.pad_token_id is not None:
                    labels[labels == tokenizer.pad_token_id] = -100

                # Compute Hessian-vector product for this batch
                hvp_batch = compute_hvp_batch(input_ids, attention_mask, labels, v)

                # Accumulate weighted by batch size
                Hv_accumulated += hvp_batch * batch_size_actual
                total_samples += batch_size_actual

            # Average the accumulated Hessian-vector product
            Hv = Hv_accumulated / total_samples

            # Compute eigenvalue estimate (Rayleigh quotient)
            eigenvalue = torch.dot(v.flatten(), Hv.flatten()).item()

            # Check for convergence
            if eigenvalue_prev is not None and abs(eigenvalue - eigenvalue_prev) < 1e-6:
                print(f"Power iteration converged after {iteration + 1} iterations")
                break

            eigenvalue_prev = eigenvalue

            # Update direction vector for next iteration (FIXED: proper power iteration)
            v_norm = torch.norm(Hv)
            if v_norm > 1e-8:
                v = Hv / v_norm
            else:
                # If Hv is too small, reinitialize randomly
                v = torch.randn_like(weight_tensor, device=device)
                v = v / torch.norm(v)
                print(f"Reinitializing v at iteration {iteration} due to small Hv norm")

        # Clean up
        model.train(original_state)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return abs(eigenvalue) if eigenvalue_prev is not None else 0.0
