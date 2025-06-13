from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import copy
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sensitivity_metrics import SensitivityMetrics
from reporter import run_full_analysis_report
from model_utils import evaluate_model, get_model_size_mb
from quantizer import (
    replace_single_linear_with_target,
    W4A16LinearLayer,
    W8A16LinearLayer,
    W16A16LinearLayer,
)


class LayerSensitivityAnalyzer:
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        device: str = "auto",
        calibration_data=None,
        eval_data=None,
        calibration_num_samples: int = 100,
        eval_num_samples: int = 100,
        batch_size: int = 16,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
    ):
        """Initialize the analyzer with model and dataset configuration."""
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None
        ).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.calibration_data = calibration_data or self._prepare_hf_dataset(
            split="train",
            num_samples=calibration_num_samples,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split_name="calibration",
        )
        self.eval_data = eval_data or self._prepare_hf_dataset(
            split="validation",
            num_samples=eval_num_samples,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split_name="evaluation",
        )
        self.calibration_num_samples = calibration_num_samples
        self.eval_num_samples = eval_num_samples
        self.sensitivity_scores = {}

    def _prepare_hf_dataset(
        self,
        split: str,
        num_samples: int,
        dataset_name: str,
        dataset_config: str,
        split_name: str,
    ):
        """Prepare and tokenize dataset samples from HuggingFace datasets."""
        print(
            f"Preparing {split_name} data from HuggingFace dataset: {dataset_name}/{dataset_config} [{split}] ..."
        )
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        texts = []
        for example in dataset:
            text = example["text"].strip()
            if len(text) > 50:
                texts.append(text)
            if len(texts) >= num_samples:
                break
        if len(texts) < num_samples:
            print(f"Warning: Only {len(texts)} samples found for {split_name}.")
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    def _get_layer_modules(self):
        """Extract all quantizable layer modules from the model."""
        layers = {}
        for name, module in self.model.named_modules():
            # Only include layers that have weights
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                layers[name] = module
        return layers

    def _compute_activation_statistics(self):
        """Compute mean, std, and magnitude statistics for layer activations."""
        print("Computing activation statistics...")
        activations = {}
        hooks = []

        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
                elif isinstance(output, tuple) and len(output) > 0:
                    activations[name] = output[0].detach()

            return hook

        layers = self._get_layer_modules()
        for name, module in layers.items():
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
        with torch.no_grad():
            for i in range(0, len(self.calibration_data["input_ids"]), self.batch_size):
                end_idx = min(
                    i + self.batch_size, len(self.calibration_data["input_ids"])
                )
                batch_inputs = {
                    "input_ids": self.calibration_data["input_ids"][i:end_idx],
                    "attention_mask": self.calibration_data["attention_mask"][
                        i:end_idx
                    ],
                }
                outputs = self.model(**batch_inputs)
        for hook in hooks:
            hook.remove()
        stats = {}
        for name, activation in activations.items():
            stats[name] = {
                "mean": activation.mean().item(),
                "std": activation.std().item(),
                "magnitude": activation.norm().item(),
            }
        return stats

    def analyze_layer_sensitivity(
        self, bits: int = 8, sensitivity_method: str = "divergence"
    ):
        """Analyze the sensitivity of each layer to quantization using the specified method."""
        print(f"Analyzing layer sensitivity with {bits}-bit quantization...")
        baseline_outputs = self._compute_baseline()
        activation_stats = self._compute_activation_statistics()
        layers = self._get_layer_modules()
        sensitivity_scores = {}
        from tqdm import tqdm

        for layer_name, layer_module in tqdm(layers.items(), desc="Analyzing layers"):
            model_copy = copy.deepcopy(self.model)
            replace_single_linear_with_target(
                model_copy,
                W8A16LinearLayer, 
                layer_name,
            )

            quantized_outputs = self._compute_quantized(model_copy)
            if sensitivity_method == "divergence":
                sensitivity_scores[
                    layer_name
                ] = SensitivityMetrics.compute_divergence_based_sensitivities(
                    baseline_outputs, quantized_outputs
                )
            elif sensitivity_method == "hessian":
                sensitivity_scores[
                    layer_name
                ] = SensitivityMetrics.compute_hessian_based_sensitivities(
                    model_copy, layer_name, layer_module, self.calibration_data
                )

            # Normalize activation magnitude to [0,1] range
            if layer_name in activation_stats:
                # Get the maximum activation magnitude across all layers for normalization
                max_magnitude = max(
                    stat["magnitude"] for stat in activation_stats.values()
                )
                normalized_magnitude = (
                    activation_stats[layer_name]["magnitude"] / max_magnitude
                )
                # Apply a small boost to the score based on normalized magnitude
                sensitivity_scores[layer_name] *= 1.0 + 0.5 * normalized_magnitude

            del model_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.sensitivity_scores = sensitivity_scores
        return sensitivity_scores

    def _compute_baseline(self):
        """Compute and store outputs from the original model."""
        if hasattr(self, "_cached_baseline_outputs"):
            return self._cached_baseline_outputs

        baseline_outputs = []
        with torch.no_grad():
            for i in range(0, len(self.calibration_data["input_ids"]), self.batch_size):
                end_idx = min(
                    i + self.batch_size, len(self.calibration_data["input_ids"])
                )
                batch_inputs = {
                    "input_ids": self.calibration_data["input_ids"][i:end_idx],
                    "attention_mask": self.calibration_data["attention_mask"][
                        i:end_idx
                    ],
                }
                outputs = self.model(**batch_inputs)
                baseline_outputs.append(
                    {
                        "logits": outputs.logits.cpu(),
                        "last_hidden_state": outputs.last_hidden_state.cpu()
                        if hasattr(outputs, "last_hidden_state")
                        else None,
                    }
                )
        # Cache the baseline outputs for later use
        self._cached_baseline_outputs = baseline_outputs
        return baseline_outputs

    def _compute_quantized(self, quantized_model):
        """Compute and store outputs from the quantized model."""
        quantized_outputs = []
        with torch.no_grad():
            for i in range(0, len(self.calibration_data["input_ids"]), self.batch_size):
                end_idx = min(
                    i + self.batch_size, len(self.calibration_data["input_ids"])
                )
                batch_inputs = {
                    "input_ids": self.calibration_data["input_ids"][i:end_idx],
                    "attention_mask": self.calibration_data["attention_mask"][
                        i:end_idx
                    ],
                }
                outputs = quantized_model(**batch_inputs)
                quantized_outputs.append(
                    {
                        "logits": outputs.logits.cpu(),
                        "last_hidden_state": outputs.last_hidden_state.cpu()
                        if hasattr(outputs, "last_hidden_state")
                        else None,
                    }
                )
        return quantized_outputs

    def get_sensitivity_based_config(
        self, sensitivity_distribution: str = "adaptive_threshold"
    ):
        """
        Generate mixed precision configuration based on sensitivity score distribution.

        Args:
            sensitivity_distribution: Strategy for bit allocation
                - "adaptive_threshold": Use statistical thresholds based on sensitivity distribution
                - "percentile": Use percentile-based allocation
                - "exponential": Use exponential decay mapping
                - "conservative": Prioritize model accuracy over compression
                - "aggressive": Prioritize compression over accuracy
                - "int8_only": Use INT8 for all layers (not recommended unless with iterative refinement)
                - "int4_only": Use INT4 for all layers (not recommended unless with iterative refinement)
        """
        if not self.sensitivity_scores:
            raise ValueError(
                "No sensitivity scores available. Run analyze_layer_sensitivity first."
            )

        # Get sensitivity values and statistics
        sensitivities = list(self.sensitivity_scores.values())
        mean_sens = np.mean(sensitivities)
        std_sens = np.std(sensitivities)
        min_sens = min(sensitivities)
        max_sens = max(sensitivities)

        print(
            f"Sensitivity stats - Mean: {mean_sens:.4f}, Std: {std_sens:.4f}, Range: [{min_sens:.4f}, {max_sens:.4f}]"
        )

        config = {}

        if sensitivity_distribution == "adaptive_threshold":
            # Use statistical thresholds based on mean and standard deviation
            high_threshold = mean_sens + 0.5 * std_sens  # Very sensitive layers
            medium_threshold = mean_sens  # Moderately sensitive layers
            low_threshold = mean_sens - 0.5 * std_sens  # Less sensitive layers

            for layer_name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= high_threshold:
                    config[layer_name] = 32  # Keep most sensitive layers in FP32
                elif sensitivity >= medium_threshold:
                    config[layer_name] = 16  # Medium sensitivity -> BF16
                elif sensitivity >= low_threshold:
                    config[layer_name] = 8  # Lower sensitivity -> INT8
                else:
                    config[layer_name] = 4  # Least sensitive -> INT4

        elif sensitivity_distribution == "percentile":
            # Use percentile-based allocation
            p75 = np.percentile(sensitivities, 75)  # Top 25% most sensitive
            p50 = np.percentile(sensitivities, 50)  # Top 50% most sensitive
            p25 = np.percentile(sensitivities, 25)  # Top 75% most sensitive

            for layer_name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= p75:
                    config[layer_name] = 32
                elif sensitivity >= p50:
                    config[layer_name] = 16
                elif sensitivity >= p25:
                    config[layer_name] = 8
                else:
                    config[layer_name] = 4

        elif sensitivity_distribution == "exponential":
            # Exponential mapping: more gradual transition
            # Normalize sensitivities to [0, 1] range
            norm_sensitivities = {}
            sens_range = max_sens - min_sens
            if sens_range == 0:  # All layers have same sensitivity
                sens_range = 1

            for layer_name, sensitivity in self.sensitivity_scores.items():
                normalized = (sensitivity - min_sens) / sens_range
                # Apply exponential mapping to emphasize high sensitivity
                exp_score = np.exp(2 * normalized) / np.exp(2)  # Scale to [1/e^2, 1]
                norm_sensitivities[layer_name] = exp_score

            for layer_name, norm_sens in norm_sensitivities.items():
                if norm_sens >= 0.75:
                    config[layer_name] = 32
                elif norm_sens >= 0.5:
                    config[layer_name] = 16
                elif norm_sens >= 0.25:
                    config[layer_name] = 8
                else:
                    config[layer_name] = 4

        elif sensitivity_distribution == "conservative":
            # Conservative: err on the side of higher precision
            low_threshold = mean_sens - 0.25 * std_sens
            medium_threshold = mean_sens + 0.25 * std_sens
            high_threshold = mean_sens + 0.75 * std_sens

            for layer_name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= high_threshold:
                    config[layer_name] = 32
                elif sensitivity >= medium_threshold:
                    config[layer_name] = 32  # More layers in FP32
                elif sensitivity >= low_threshold:
                    config[layer_name] = 16
                else:
                    config[layer_name] = 8  # Avoid INT4 for most layers

        elif sensitivity_distribution == "aggressive":
            # Aggressive: prioritize compression
            high_threshold = (
                mean_sens + 1.0 * std_sens
            )  # Only very high sensitivity gets FP32
            medium_threshold = mean_sens + 0.25 * std_sens
            low_threshold = mean_sens - 0.25 * std_sens

            for layer_name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= high_threshold:
                    config[layer_name] = 32
                elif sensitivity >= medium_threshold:
                    config[layer_name] = 16
                elif sensitivity >= low_threshold:
                    config[layer_name] = 8
                else:
                    config[layer_name] = 4

        elif sensitivity_distribution == "int8_only":
            # INT8 only: all layers quantized to INT8
            for layer_name in self.sensitivity_scores.keys():
                config[layer_name] = 8

        elif sensitivity_distribution == "int4_only":
            # INT4 only: all layers quantized to INT4
            for layer_name in self.sensitivity_scores.keys():
                config[layer_name] = 4

        # Print distribution summary
        bit_counts = {32: 0, 16: 0, 8: 0, 4: 0}
        for bits in config.values():
            bit_counts[bits] += 1

        total_layers = len(config)
        print(f"\nBit allocation distribution:")
        print(f"FP32: {bit_counts[32]} layers ({bit_counts[32]/total_layers*100:.1f}%)")
        print(f"BF16: {bit_counts[16]} layers ({bit_counts[16]/total_layers*100:.1f}%)")
        print(f"INT8: {bit_counts[8]} layers ({bit_counts[8]/total_layers*100:.1f}%)")
        print(f"INT4: {bit_counts[4]} layers ({bit_counts[4]/total_layers*100:.1f}%)")

        return config

    def get_iterative_config(
        self, 
        max_perplexity_increase: float = 0.1, 
        initial_strategy: str = "aggressive",
        layers_per_iteration: int = 3,
        max_iterations: int = 50
    ):
        """
        Iteratively adjust quantization config with progressive bit-level upgrades.
        
        Strategy:
        1. Start with initial configuration (e.g., aggressive)
        2. For each bit level (4→8→16→32), upgrade ALL layers at that level before moving to next
        3. Upgrade 'layers_per_iteration' layers at a time before evaluating
        4. Continue until perplexity constraint is met or max iterations reached

        Args:
            max_perplexity_increase: Maximum allowed perplexity increase (relative)
            initial_strategy: Starting quantization strategy
            layers_per_iteration: Number of layers to upgrade per iteration
            max_iterations: Maximum total iterations allowed across all levels
        """
        print(f"Starting progressive iterative configuration:")
        print(f"  - Max perplexity increase: {max_perplexity_increase}")
        print(f"  - Layers per iteration: {layers_per_iteration}")
        print(f"  - Max total iterations: {max_iterations}")

        # Get baseline perplexity
        baseline_ppl = evaluate_model(
            self.model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        print(f"Baseline perplexity: {baseline_ppl:.4f}")

        # Start with initial configuration
        current_config = self.get_sensitivity_based_config(initial_strategy)
        
        # Sort layers by sensitivity (descending) for upgrade priority
        sorted_layers = sorted(
            self.sensitivity_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        # Define bit upgrade levels
        upgrade_levels = [(4, 8), (8, 16), (16, 32)]
        
        total_iteration = 0
        
        for from_bits, to_bits in upgrade_levels:
            print(f"\n{'='*50}")
            print(f"UPGRADING LAYERS FROM {from_bits} TO {to_bits} BITS")
            print(f"{'='*50}")
            
            # Get layers at current bit level, sorted by sensitivity (most sensitive first)
            layers_at_current_level = [
                (layer_name, sensitivity) for layer_name, sensitivity in sorted_layers
                if current_config[layer_name] == from_bits
            ]
            
            if not layers_at_current_level:
                print(f"No layers at {from_bits} bits. Skipping this level.")
                continue
                
            print(f"Found {len(layers_at_current_level)} layers at {from_bits} bits")
            
            layers_upgraded_this_level = 0
            
            # Continue upgrading layers at this level until all are upgraded or constraints are met
            while layers_at_current_level and total_iteration < max_iterations:
                total_iteration += 1
                
                # Check current perplexity
                quantized_model = self.apply_mixed_precision(current_config)
                current_ppl = evaluate_model(
                    quantized_model,
                    self.tokenizer,
                    self.eval_data,
                    self.eval_num_samples,
                    self.device,
                )
                
                perplexity_increase = (current_ppl - baseline_ppl) / baseline_ppl
                print(f"\nLevel {from_bits}→{to_bits}, Iteration {total_iteration}:")
                print(f"  Current perplexity: {current_ppl:.4f} (increase: {perplexity_increase:.3f})")
                
                # Check if constraint is satisfied
                if perplexity_increase <= max_perplexity_increase:
                    print(f"  ✓ Perplexity constraint satisfied!")
                    del quantized_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return current_config
                
                # Get next batch of layers to upgrade (up to layers_per_iteration)
                layers_to_upgrade = layers_at_current_level[:layers_per_iteration]
                
                # Upgrade the selected layers
                print(f"  Upgrading {len(layers_to_upgrade)} layers:")
                for layer_name, sensitivity in layers_to_upgrade:
                    current_config[layer_name] = to_bits
                    layers_upgraded_this_level += 1
                    print(f"    - {layer_name} (sensitivity: {sensitivity:.4f})")
                
                # Remove upgraded layers from the current level list
                layers_at_current_level = layers_at_current_level[layers_per_iteration:]
                
                del quantized_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"\nCompleted {from_bits}→{to_bits} bit level:")
            print(f"  - Layers upgraded: {layers_upgraded_this_level}")
            print(f"  - Remaining layers at {from_bits} bits: {len(layers_at_current_level)}")
            
            # Check if we've reached max iterations
            if total_iteration >= max_iterations:
                print(f"  - Reached maximum iterations ({max_iterations})")
                break
            
            # Print current distribution
            bit_counts = {32: 0, 16: 0, 8: 0, 4: 0}
            for bits in current_config.values():
                bit_counts[bits] += 1
            
            total_layers = len(current_config)
            print(f"  - Current distribution:")
            print(f"    FP32: {bit_counts[32]} ({bit_counts[32]/total_layers*100:.1f}%)")
            print(f"    BF16: {bit_counts[16]} ({bit_counts[16]/total_layers*100:.1f}%)")
            print(f"    INT8: {bit_counts[8]} ({bit_counts[8]/total_layers*100:.1f}%)")
            print(f"    INT4: {bit_counts[4]} ({bit_counts[4]/total_layers*100:.1f}%)")
        
        # Final evaluation
        print(f"\n{'='*50}")
        print("FINAL EVALUATION")
        print(f"{'='*50}")
        
        quantized_model = self.apply_mixed_precision(current_config)
        final_ppl = evaluate_model(
            quantized_model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        
        final_perplexity_increase = (final_ppl - baseline_ppl) / baseline_ppl
        print(f"Final perplexity: {final_ppl:.4f} (increase: {final_perplexity_increase:.3f})")
        print(f"Total iterations used: {total_iteration}")
        
        if final_perplexity_increase <= max_perplexity_increase:
            print("✓ Final configuration meets perplexity constraint!")
        else:
            print("⚠ Final configuration exceeds perplexity constraint.")
            print("  Consider using a less aggressive initial strategy or higher constraint.")
        
        del quantized_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return current_config

    def apply_mixed_precision(self, config: Dict[str, int]):
        """Apply mixed precision quantization to the model based on the provided configuration."""
        print("Applying mixed precision quantization...")
        quantized_model = copy.deepcopy(self.model)
        for layer_name, bits in config.items():
            # Skip empty layer names
            if not layer_name.strip():
                continue

            try:
                parts = layer_name.split(".")
                current = self.model
                for part in parts:
                    if not part:  # Skip empty parts
                        continue
                    current = getattr(current, part)

                if bits == 32:
                    continue

                elif bits == 16:
                    replace_single_linear_with_target(
                        quantized_model, W16A16LinearLayer, layer_name
                    )

                elif bits == 8:
                    replace_single_linear_with_target(
                        quantized_model, W8A16LinearLayer, layer_name
                    )
                else:  # bits==4
                    replace_single_linear_with_target(
                        quantized_model, W4A16LinearLayer, layer_name
                    )

            except Exception as e:
                print(f"Warning: Could not quantize layer {layer_name}: {str(e)}")
                continue
        return quantized_model

    def run_full_analysis(
        self,
        sensitivity_method,
        config_strategy,
        use_iterative,
        max_perplexity_increase,
        layers_per_iteration,
        max_iterations,
    ):
        """
        Run full analysis with the new sensitivity-based configuration approach.

        Args:
            sensitivity_method: Method for computing sensitivity scores
            config_strategy: Strategy for bit allocation based on sensitivity
            use_iterative: Whether to use iterative refinement based on perplexity
            max_perplexity_increase: Max allowed perplexity increase (for iterative mode)
            layers_per_iteration: Number of layers to upgrade per iteration (for iterative mode)
            max_iterations_per_level: Max iterations per bit level (for iterative mode)
        """
        print("=" * 60)
        print("RUNNING FULL MIXED PRECISION ANALYSIS")
        print("=" * 60)

        # Analyze layer sensitivity
        sensitivity_scores = self.analyze_layer_sensitivity(
            sensitivity_method=sensitivity_method
        )

        # Get mixed precision config based on sensitivity
        if use_iterative:
            mp_config = self.get_iterative_config(
                max_perplexity_increase=max_perplexity_increase,
                initial_strategy=config_strategy,
                layers_per_iteration=layers_per_iteration,
                max_iterations=max_iterations,
            )
        else:
            mp_config = self.get_sensitivity_based_config(config_strategy)

        # Evaluate models
        print("\nEvaluating models...")
        original_ppl = evaluate_model(
            self.model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        original_size = get_model_size_mb(self.model)

        quantized_model = self.apply_mixed_precision(mp_config)
        quantized_ppl = evaluate_model(
            quantized_model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        quantized_size = get_model_size_mb(quantized_model)

        results = run_full_analysis_report(
            sensitivity_scores,
            mp_config,
            original_ppl,
            quantized_ppl,
            original_size,
            quantized_size,
            sensitivity_method=sensitivity_method,
        )

        return results