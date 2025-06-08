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
            # if ("transformer.h." in name and any(x in name for x in [".c_attn", ".c_proj", ".c_fc"])) or \
            # "lm_head" in name:
            # layers[name] = module

            # Only include layers that have weights
            if isinstance(module, nn.Linear):  # and "lm_head" not in name:
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
                W8A16LinearLayer,  # Assuming we want to quantize Linear layers
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

        # Calculate average bits and compression ratio
        # total_bits = sum(bits for bits in config.values())
        # avg_bits = total_bits / total_layers
        # compression_ratio = 32 / avg_bits
        # print(f"Average bits per layer: {avg_bits:.2f}")
        # print(f"Theoretical compression ratio: {compression_ratio:.2f}x")

        return config

    def get_iterative_config(
        self, max_perplexity_increase: float = 0.1, initial_strategy: str = "aggressive"
    ):
        """
        Iteratively adjust quantization config based on perplexity feedback.
        Start aggressive and progressively increase precision for most sensitive layers.

        Args:
            max_perplexity_increase: Maximum allowed perplexity increase (relative)
            initial_strategy: Starting quantization strategy
        """
        print(
            f"Starting iterative configuration with max perplexity increase: {max_perplexity_increase}"
        )

        # Get baseline perplexity
        baseline_ppl = evaluate_model(
            self.model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        print(f"Baseline perplexity: {baseline_ppl:.4f}")

        # Start with aggressive configuration
        current_config = self.get_sensitivity_based_config(initial_strategy)

        # Sort layers by sensitivity (descending) for upgrade priority
        sorted_layers = sorted(
            self.sensitivity_scores.items(), key=lambda x: x[1], reverse=True
        )

        upgrade_order = [4, 8, 16, 32]  # Priority order for upgrades

        iteration = 0
        while iteration < 100:  # Max iterations to prevent infinite loop
            iteration += 1
            print(f"\nIteration {iteration}:")

            # Test current configuration
            quantized_model = self.apply_mixed_precision(current_config)
            current_ppl = evaluate_model(
                quantized_model,
                self.tokenizer,
                self.eval_data,
                self.eval_num_samples,
                self.device,
            )

            perplexity_increase = (current_ppl - baseline_ppl) / baseline_ppl
            print(
                f"Current perplexity: {current_ppl:.4f} (increase: {perplexity_increase:.3f})"
            )

            if perplexity_increase <= max_perplexity_increase:
                print("Perplexity constraint satisfied!")
                break

            # Find layers to upgrade (increase precision)
            upgraded = False
            for layer_name, sensitivity in sorted_layers:
                current_bits = current_config[layer_name]

                # Find next higher precision level
                next_bits = None
                for bits in upgrade_order:
                    if bits > current_bits:
                        next_bits = bits
                        break

                if next_bits is not None:
                    print(
                        f"Upgrading {layer_name} from {current_bits} to {next_bits} bits"
                    )
                    current_config[layer_name] = next_bits
                    upgraded = True
                    break

            if not upgraded:
                print("No more layers to upgrade. Stopping.")
                break

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
        sensitivity_method: str = "divergence",
        config_strategy: str = "aggressive",
        use_iterative: bool = True,
        max_perplexity_increase: float = 0.1,
    ):
        """
        Run full analysis with the new sensitivity-based configuration approach.

        Args:
            sensitivity_method: Method for computing sensitivity scores
            config_strategy: Strategy for bit allocation based on sensitivity
            use_iterative: Whether to use iterative refinement based on perplexity
            max_perplexity_increase: Max allowed perplexity increase (for iterative mode)
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

        # Calculate average bits for reporting
        # total_bits = sum(bits for bits in mp_config.values())
        # avg_bits = total_bits / len(mp_config)

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



import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Callable
import copy
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration for gradient-based bit allocation optimization."""
    max_ppl_increase: float = 0.05  # Maximum allowed perplexity increase (5%)
    min_bits: int = 4              # Minimum bits per layer
    max_bits: int = 32             # Maximum bits per layer
    bit_candidates: List[int] = None  # Allowed bit widths
    optimization_method: str = "scipy"  # "scipy", "evolutionary", or "gradient_descent"
    max_iterations: int = 100      # Maximum optimization iterations
    tolerance: float = 1e-4        # Convergence tolerance
    regularization_weight: float = 0.01  # L2 regularization on bit allocation
    
    def __post_init__(self):
        if self.bit_candidates is None:
            self.bit_candidates = [4, 8, 16, 32]


class GradientBasedBitAllocator:
    """
    Gradient-based optimization for mixed precision bit allocation.
    
    Formulates the problem as:
    minimize: average_bits + λ * regularization_term
    subject to: perplexity_increase ≤ max_allowed_increase
    """
    
    def __init__(self, analyzer, config: OptimizationConfig = None):
        self.analyzer = analyzer
        self.config = config or OptimizationConfig()
        self.sensitivity_scores = analyzer.sensitivity_scores
        self.layer_names = list(self.sensitivity_scores.keys())
        self.n_layers = len(self.layer_names)
        
        # Cache baseline perplexity
        self.baseline_ppl = self._compute_baseline_perplexity()
        print(f"Baseline perplexity: {self.baseline_ppl:.4f}")
        
        # Precompute perplexity functions for each layer at different bit widths
        self._precompute_layer_perplexities()
    
    def _compute_baseline_perplexity(self) -> float:
        """Compute baseline perplexity for the original model."""
        from model_utils import evaluate_model
        return evaluate_model(
            self.analyzer.model,
            self.analyzer.tokenizer,
            self.analyzer.eval_data,
            self.analyzer.eval_num_samples,
            self.analyzer.device,
        )
    
    def _precompute_layer_perplexities(self):
        """
        Precompute perplexity impact for each layer at different bit widths.
        This creates a lookup table to avoid repeated model evaluations during optimization.
        """
        print("Precomputing layer perplexity impacts...")
        self.layer_ppl_impacts = {}
        
        for i, layer_name in enumerate(self.layer_names):
            print(f"Analyzing layer {i+1}/{self.n_layers}: {layer_name}")
            self.layer_ppl_impacts[layer_name] = {}
            
            for bits in self.config.bit_candidates:
                if bits == 32:  # FP32 baseline
                    self.layer_ppl_impacts[layer_name][bits] = 0.0
                    continue
                
                # Create model with only this layer quantized
                config = {layer_name: bits}
                # Set all other layers to FP32
                for other_layer in self.layer_names:
                    if other_layer != layer_name:
                        config[other_layer] = 32
                
                try:
                    quantized_model = self.analyzer.apply_mixed_precision(config)
                    ppl = evaluate_model(
                        quantized_model,
                        self.analyzer.tokenizer,
                        self.analyzer.eval_data,
                        self.analyzer.eval_num_samples,
                        self.analyzer.device,
                    )
                    
                    # Store the perplexity increase caused by quantizing this layer
                    ppl_increase = (ppl - self.baseline_ppl) / self.baseline_ppl
                    self.layer_ppl_impacts[layer_name][bits] = ppl_increase
                    
                    del quantized_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Warning: Failed to evaluate {layer_name} at {bits} bits: {e}")
                    self.layer_ppl_impacts[layer_name][bits] = 1.0  # High penalty
    
    def _estimate_total_perplexity_increase(self, bit_allocation: Dict[str, int]) -> float:
        """
        Estimate total perplexity increase from individual layer contributions.
        Uses a combination of linear superposition and interaction terms.
        """
        total_increase = 0.0
        
        # Linear contribution from each layer
        for layer_name, bits in bit_allocation.items():
            if layer_name in self.layer_ppl_impacts and bits in self.layer_ppl_impacts[layer_name]:
                layer_contribution = self.layer_ppl_impacts[layer_name][bits]
                
                # Weight by sensitivity score
                sensitivity = self.sensitivity_scores.get(layer_name, 1.0)
                weighted_contribution = layer_contribution * sensitivity
                total_increase += weighted_contribution
        
        # Add interaction term (quadratic penalty for multiple low-bit layers)
        low_bit_layers = sum(1 for bits in bit_allocation.values() if bits <= 8)
        if low_bit_layers > 1:
            interaction_penalty = 0.1 * (low_bit_layers - 1) ** 1.5
            total_increase += interaction_penalty
            
        return total_increase
    
    def _objective_function(self, x: np.ndarray) -> float:
        """
        Objective function to minimize: weighted combination of compression and regularization.
        
        Args:
            x: Continuous variables representing bit allocation (will be mapped to discrete bits)
        """
        # Map continuous variables to discrete bit allocations
        bit_allocation = self._continuous_to_discrete_bits(x)
        
        # Calculate average bits (lower is better for compression)
        avg_bits = np.mean(list(bit_allocation.values()))
        
        # Regularization term to prefer smoother allocations
        bit_values = np.array(list(bit_allocation.values()))
        regularization = self.config.regularization_weight * np.var(bit_values)
        
        return avg_bits + regularization
    
    def _constraint_function(self, x: np.ndarray) -> float:
        """
        Constraint function: perplexity increase should be ≤ max_allowed.
        Returns negative value when constraint is satisfied.
        """
        bit_allocation = self._continuous_to_discrete_bits(x)
        estimated_ppl_increase = self._estimate_total_perplexity_increase(bit_allocation)
        
        # Return negative when constraint is satisfied (ppl_increase ≤ max_allowed)
        return self.config.max_ppl_increase - estimated_ppl_increase
    
    def _continuous_to_discrete_bits(self, x: np.ndarray) -> Dict[str, int]:
        """Map continuous optimization variables to discrete bit allocations."""
        bit_allocation = {}
        
        for i, layer_name in enumerate(self.layer_names):
            # Map continuous value [0, 1] to discrete bit candidates
            continuous_val = np.clip(x[i], 0, 1)
            idx = int(continuous_val * (len(self.config.bit_candidates) - 1))
            idx = min(idx, len(self.config.bit_candidates) - 1)
            bit_allocation[layer_name] = self.config.bit_candidates[idx]
            
        return bit_allocation
    
    def optimize_scipy(self) -> Dict[str, int]:
        """Use SciPy's constrained optimization."""
        from scipy.optimize import minimize
        
        print("Running SciPy-based optimization...")
        
        # Initial guess: start with moderate precision
        x0 = np.full(self.n_layers, 0.5)  # Middle of the range
        
        # Bounds: each variable in [0, 1]
        bounds = [(0, 1) for _ in range(self.n_layers)]
        
        # Constraint: perplexity increase ≤ max_allowed
        constraint = {
            'type': 'ineq',
            'fun': self._constraint_function
        }
        
        result = minimize(
            self._objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if result.success:
            print("Optimization converged successfully!")
        else:
            print(f"Optimization warning: {result.message}")
        
        return self._continuous_to_discrete_bits(result.x)
    
    def optimize_evolutionary(self) -> Dict[str, int]:
        """Use evolutionary/genetic algorithm for global optimization."""
        print("Running evolutionary optimization...")
        
        bounds = [(0, 1) for _ in range(self.n_layers)]
        
        def combined_objective(x):
            # Combine objective and constraint into single function
            obj = self._objective_function(x)
            constraint_violation = max(0, -self._constraint_function(x))
            penalty = 1000 * constraint_violation  # Large penalty for constraint violation
            return obj + penalty
        
        result = differential_evolution(
            combined_objective,
            bounds,
            maxiter=self.config.max_iterations,
            tol=self.config.tolerance,
            seed=42
        )
        
        return self._continuous_to_discrete_bits(result.x)
    
    def optimize_gradient_descent(self) -> Dict[str, int]:
        """
        Custom gradient descent using PyTorch for automatic differentiation.
        This provides the most flexible approach for complex objective functions.
        """
        print("Running gradient descent optimization...")
        
        # Use continuous relaxation of discrete bit allocation
        # x[i] represents the "softmax" weights over bit candidates for layer i
        x = torch.randn(self.n_layers, len(self.config.bit_candidates), requires_grad=True, device='cpu')
        optimizer = torch.optim.Adam([x], lr=0.01)
        
        best_allocation = None
        best_objective = float('inf')
        
        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()
            
            # Softmax to get probability distribution over bit candidates
            bit_probs = torch.softmax(x, dim=1)
            
            # Expected bit allocation (continuous relaxation)
            expected_bits = torch.sum(
                bit_probs * torch.tensor(self.config.bit_candidates, dtype=torch.float32), dim=1
            )
            
            # Objective: minimize average bits
            avg_bits = torch.mean(expected_bits)
            regularization = self.config.regularization_weight * torch.var(expected_bits)
            objective = avg_bits + regularization
            
            # Estimate perplexity constraint using expected allocations
            discrete_allocation = {}
            for i, layer_name in enumerate(self.layer_names):
                # Use expected bit value or sample from distribution
                expected_bit = expected_bits[i].item()
                # Map to nearest discrete value
                closest_bit = min(self.config.bit_candidates, 
                                key=lambda b: abs(b - expected_bit))
                discrete_allocation[layer_name] = closest_bit
            
            ppl_increase = self._estimate_total_perplexity_increase(discrete_allocation)
            
            # Add penalty for constraint violation
            constraint_penalty = max(0, ppl_increase - self.config.max_ppl_increase) * 1000
            total_loss = objective + constraint_penalty
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_objective and ppl_increase <= self.config.max_ppl_increase:
                best_objective = total_loss.item()
                best_allocation = discrete_allocation.copy()
            
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}: Loss={total_loss.item():.4f}, "
                      f"Avg Bits={avg_bits.item():.2f}, PPL Increase={ppl_increase:.4f}")
        
        return best_allocation or discrete_allocation
    
    def optimize(self) -> Dict[str, int]:
        """Main optimization function that selects the appropriate method."""
        if self.config.optimization_method == "scipy":
            return self.optimize_scipy()
        elif self.config.optimization_method == "evolutionary":
            return self.optimize_evolutionary()
        elif self.config.optimization_method == "gradient_descent":
            return self.optimize_gradient_descent()
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
    
    def validate_allocation(self, bit_allocation: Dict[str, int]) -> Tuple[float, bool]:
        """Validate the final bit allocation by actually evaluating the quantized model."""
        print("Validating optimized bit allocation...")
        
        quantized_model = self.analyzer.apply_mixed_precision(bit_allocation)
        actual_ppl = evaluate_model(
            quantized_model,
            self.analyzer.tokenizer,
            self.analyzer.eval_data,
            self.analyzer.eval_num_samples,
            self.analyzer.device,
        )
        
        actual_ppl_increase = (actual_ppl - self.baseline_ppl) / self.baseline_ppl
        constraint_satisfied = actual_ppl_increase <= self.config.max_ppl_increase
        
        print(f"Validation results:")
        print(f"  Actual perplexity: {actual_ppl:.4f}")
        print(f"  Actual increase: {actual_ppl_increase:.4f} (limit: {self.config.max_ppl_increase:.4f})")
        print(f"  Constraint satisfied: {constraint_satisfied}")
        
        del quantized_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return actual_ppl_increase, constraint_satisfied


# Enhanced LayerSensitivityAnalyzer with gradient-based optimization
class EnhancedLayerSensitivityAnalyzer(LayerSensitivityAnalyzer):
    """Extended analyzer with gradient-based bit allocation optimization."""
    
    def get_gradient_based_config(
        self, 
        max_ppl_increase: float = 0.05,
        optimization_method: str = "scipy",
        bit_candidates: List[int] = None
    ) -> Dict[str, int]:
        """
        Get optimized bit allocation using gradient-based optimization.
        
        Args:
            max_ppl_increase: Maximum allowed perplexity increase (default 5%)
            optimization_method: "scipy", "evolutionary", or "gradient_descent"
            bit_candidates: List of allowed bit widths (default [4, 8, 16, 32])
        """
        if not self.sensitivity_scores:
            raise ValueError("No sensitivity scores available. Run analyze_layer_sensitivity first.")
        
        config = OptimizationConfig(
            max_ppl_increase=max_ppl_increase,
            optimization_method=optimization_method,
            bit_candidates=bit_candidates or [4, 8, 16, 32]
        )
        
        optimizer = GradientBasedBitAllocator(self, config)
        optimal_allocation = optimizer.optimize()
        
        # Validate the result
        actual_increase, constraint_satisfied = optimizer.validate_allocation(optimal_allocation)
        
        if not constraint_satisfied:
            print("Warning: Optimized allocation violates perplexity constraint!")
            print("Consider increasing max_ppl_increase or using a more conservative approach.")
        
        # Print allocation summary
        self._print_allocation_summary(optimal_allocation)
        
        return optimal_allocation
    
    def _print_allocation_summary(self, allocation: Dict[str, int]):
        """Print summary of bit allocation."""
        bit_counts = {32: 0, 16: 0, 8: 0, 4: 0}
        for bits in allocation.values():
            if bits in bit_counts:
                bit_counts[bits] += 1
        
        total_layers = len(allocation)
        print(f"\nOptimized bit allocation:")
        for bits, count in bit_counts.items():
            if count > 0:
                percentage = count / total_layers * 100
                print(f"  {bits}-bit: {count} layers ({percentage:.1f}%)")
        
        avg_bits = sum(allocation.values()) / len(allocation)
        compression_ratio = 32 / avg_bits
        print(f"  Average bits: {avg_bits:.2f}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")


# Usage example
def run_gradient_based_analysis():
    """Example of how to use the gradient-based bit allocation."""
    
    # Initialize analyzer
    analyzer = EnhancedLayerSensitivityAnalyzer(
        model_name="facebook/opt-125m",
        calibration_num_samples=50,
        eval_num_samples=50
    )
    
    # Analyze layer sensitivities first
    sensitivity_scores = analyzer.analyze_layer_sensitivity(sensitivity_method="divergence")
    
    # Get gradient-based optimal configuration
    optimal_config = analyzer.get_gradient_based_config(
        max_ppl_increase=0.05,  # 5% max perplexity increase
        optimization_method="scipy",  # or "evolutionary", "gradient_descent"
        bit_candidates=[4, 8, 16, 32]
    )
    
    print("\nOptimal bit allocation:")
    for layer_name, bits in optimal_config.items():
        sensitivity = sensitivity_scores[layer_name]
        print(f"  {layer_name}: {bits} bits (sensitivity: {sensitivity:.4f})")
    
    return optimal_config

if __name__ == "__main__":
    optimal_config = run_gradient_based_analysis()