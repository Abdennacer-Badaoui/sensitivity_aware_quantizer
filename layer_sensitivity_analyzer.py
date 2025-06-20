import os
import json
import time
from typing import Dict, List
import copy
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sensitivity_metrics import SensitivityMetrics
from reporter import run_full_analysis_report
from model_utils import perplexity, get_model_size_mb
from benchmarker import evaluate_llm_benchmark
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
        calibration_data=None,
        eval_data=None,
        calibration_num_samples: int = 100,
        eval_num_samples: int = 100,
        batch_size: int = 128,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        sensitivity_method="divergence",
        config_strategy="adaptive_threshold",
        use_iterative=True,
        max_perplexity_increase=0.1,
        layers_per_iteration=3,
        max_iterations=50,
        benchmarking_tasks: List[str] = None,
        device: str = "auto",
    ):
        """Initialize a LayerSensitivityAnalyzer for quantizing transformer models.

        This class analyzes the sensitivity of different layers in a transformer model
        and determines the optimal quantization strategy while maintaining model performance.

        Args:
            model_name (str): HuggingFace model identifier. Defaults to "microsoft/DialoGPT-small".
            calibration_data: Pre-prepared calibration dataset. If None, will be created from dataset_name.
            eval_data: Pre-prepared evaluation dataset. If None, will be created from dataset_name.
            calibration_num_samples (int): Number of samples for calibration. Defaults to 100.
            eval_num_samples (int): Number of samples for evaluation. Defaults to 100.
            batch_size (int): Batch size for processing. Defaults to 128.
            dataset_name (str): HuggingFace dataset name. Defaults to "wikitext".
            dataset_config (str): Dataset configuration. Defaults to "wikitext-2-raw-v1".
            sensitivity_method (str): Method for computing layer sensitivity ("divergence" or "hessian").
            config_strategy (str): Strategy for quantization configuration
                ("adaptive_threshold", "percentile", "exponential", "conservative", "aggressive",
                "int8_only", or "int4_only").
            use_iterative (bool): Whether to use iterative refinement. Defaults to True.
            max_perplexity_increase (float): Maximum allowed perplexity degradation. Defaults to 0.1.
            layers_per_iteration (int): Number of layers to upgrade per iteration. Defaults to 3.
            max_iterations (int): Maximum number of refinement iterations. Defaults to 50.
            benchmarking_tasks (List[str]): List of tasks for benchmarking the model performance.
            device (str): Computing device ("cuda", "cpu", or "auto"). Defaults to "auto".


        Note:
            The analyzer supports various quantization strategies and can be configured
            for different trade-offs between model size and performance.
        """
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        print(f"Loading model {model_name}  ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None
        ).to(self.device)
        self.model.eval()
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
        self.batch_size = batch_size
        self.sensitivity_scores = {}
        self.sensitivity_method = sensitivity_method
        self.config_strategy = config_strategy
        self.use_iterative = use_iterative
        self.max_perplexity_increase = max_perplexity_increase
        self.layers_per_iteration = layers_per_iteration
        self.max_iterations = max_iterations
        self.benchmarking_tasks = benchmarking_tasks

    def _prepare_hf_dataset(
        self,
        split: str,
        num_samples: int,
        dataset_name: str,
        dataset_config: str,
        split_name: str,
    ):
        """Prepare and tokenize dataset from HuggingFace datasets.

        Args:
            split (str): Dataset split ("train", "validation", "test").
            num_samples (int): Number of samples to prepare.
            dataset_name (str): HuggingFace dataset name.
            dataset_config (str): Dataset configuration name.
            split_name (str): Name for logging purposes.

        Returns:
            dict: Dictionary containing tokenized input_ids and attention_mask tensors.
        """
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
        """Extract all quantizable linear layers from the model.

        Returns:
            dict: Dictionary mapping layer names to their corresponding nn.Linear modules.
        """
        layers = {}
        for name, module in self.model.named_modules():
            # Only include layers that have weights
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                layers[name] = module
        return layers

    def _compute_activation_statistics(self):
        """Compute statistical metrics for layer activations.

        Analyzes layer activations during forward pass to compute mean, standard deviation,
        and magnitude statistics for each layer.

        Returns:
            dict: Dictionary containing activation statistics for each layer.
        """
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

    def analyze_layer_sensitivity(self, bits: int = 8):
        """Analyze each layer's sensitivity to quantization.

        Quantizes each layer individually and measures its impact on model outputs
        using the specified sensitivity method.

        Args:
            bits (int): Number of bits for quantization testing. Defaults to 8.

        Returns:
            dict: Dictionary mapping layer names to their sensitivity scores.
        """
        print(f"Analyzing layer sensitivity with {bits}-bit quantization ...")
        baseline_outputs = self._compute_baseline()
        activation_stats = self._compute_activation_statistics()
        layers = self._get_layer_modules()
        sensitivity_scores = {}

        from tqdm import tqdm

        for layer_name, layer_module in tqdm(
            layers.items(), desc="Sensitivity Analysis"
        ):
            model_copy = copy.deepcopy(self.model)
            replace_single_linear_with_target(
                model_copy,
                W8A16LinearLayer,
                layer_name,
            )

            quantized_outputs = self._compute_quantized(model_copy)
            if self.sensitivity_method == "divergence":
                sensitivity_scores[
                    layer_name
                ] = SensitivityMetrics.compute_divergence_based_sensitivities(
                    baseline_outputs, quantized_outputs
                )
            elif self.sensitivity_method == "hessian":
                sensitivity_scores[
                    layer_name
                ] = SensitivityMetrics.compute_hessian_based_sensitivities(
                    self.model,
                    self.tokenizer,
                    layer_module.weight,
                    self.calibration_data,
                    self.batch_size,
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

    def cached_sensitivity_scores(self, results_dir: str):
        """Load previously computed sensitivity scores from cache.

        Args:
            results_dir (str): Directory containing cached sensitivity results.

        Returns:
            dict: Cached sensitivity scores if found, None otherwise.
        """
        for file in os.listdir(results_dir):
            with open(os.path.join(results_dir, file), "r") as f:
                data = json.load(f)
                if (
                    data["model_name"] == self.model_name
                    and data["calibration_num_samples"] == self.calibration_num_samples
                    and data["sensitivity_method"] == self.sensitivity_method
                ):
                    self.sensitivity_scores = data["sensitivity_scores"]
                    print(
                        f"Loaded cached sensitivity scores from {file} (model_name: {self.model_name}, sensitivity_method: {self.sensitivity_method}, calibration_num_samples: {self.calibration_num_samples})"
                    )
                    return self.sensitivity_scores
        return None

    def _compute_baseline(self):
        """Compute and cache baseline model outputs.

        Returns:
            list: List of dictionaries containing baseline model outputs (logits and hidden states).
        """
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
        """Compute outputs from a quantized version of the model.

        Args:
            quantized_model: The quantized model to evaluate.

        Returns:
            list: List of dictionaries containing quantized model outputs.
        """
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

    def get_sensitivity_based_config(self):
        """Generate mixed precision configuration based on sensitivity scores.

        Uses the selected configuration strategy to determine optimal bit-width
        for each layer based on its sensitivity score.

        Returns:
            dict: Layer-wise quantization configuration mapping layer names to bit-widths.
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

        if self.config_strategy == "adaptive_threshold":
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

        elif self.config_strategy == "percentile":
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

        elif self.config_strategy == "exponential":
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

        elif self.config_strategy == "conservative":
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

        elif self.config_strategy == "aggressive":
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

        elif self.config_strategy == "int8_only":
            # INT8 only: all layers quantized to INT8
            for layer_name in self.sensitivity_scores.keys():
                config[layer_name] = 8

        elif self.config_strategy == "int4_only":
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

    def get_iterative_config(self):
        """Iteratively refine quantization configuration to meet performance constraints.

        Progressively upgrades layer precision based on sensitivity scores while
        monitoring model performance, aiming to find the optimal trade-off between
        model size and accuracy.

        Returns:
            dict: Optimized layer-wise quantization configuration.
        """
        print(f"Starting progressive iterative configuration:")
        print(f"  - Max perplexity increase: {self.max_perplexity_increase}")
        print(f"  - Layers per iteration: {self.layers_per_iteration}")
        print(f"  - Max total iterations: {self.max_iterations}")

        # Get baseline perplexity
        baseline_ppl = perplexity(
            self.model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        print(f"Baseline perplexity: {baseline_ppl:.4f}")

        # Start with initial configuration
        current_config = self.get_sensitivity_based_config()

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
                (layer_name, sensitivity)
                for layer_name, sensitivity in sorted_layers
                if current_config[layer_name] == from_bits
            ]

            if not layers_at_current_level:
                print(f"No layers at {from_bits} bits. Skipping this level.")
                continue

            print(f"Found {len(layers_at_current_level)} layers at {from_bits} bits")

            layers_upgraded_this_level = 0

            # Continue upgrading layers at this level until all are upgraded or constraints are met
            while layers_at_current_level and total_iteration < self.max_iterations:
                total_iteration += 1

                # Check current perplexity
                quantized_model = self.apply_mixed_precision(current_config)
                current_ppl = perplexity(
                    quantized_model,
                    self.tokenizer,
                    self.eval_data,
                    self.eval_num_samples,
                    self.device,
                )

                perplexity_increase = (current_ppl - baseline_ppl) / baseline_ppl
                print(f"\nLevel {from_bits}→{to_bits}, Iteration {total_iteration}:")
                print(
                    f"  Current perplexity: {current_ppl:.4f} (increase: {perplexity_increase:.3f})"
                )

                # Check if constraint is satisfied
                if perplexity_increase <= self.max_perplexity_increase:
                    print(f"  ✓ Perplexity constraint satisfied!")
                    del quantized_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return current_config

                # Get next batch of layers to upgrade (up to layers_per_iteration)
                layers_to_upgrade = layers_at_current_level[: self.layers_per_iteration]

                # Upgrade the selected layers
                print(f"  Upgrading {len(layers_to_upgrade)} layers:")
                for layer_name, sensitivity in layers_to_upgrade:
                    current_config[layer_name] = to_bits
                    layers_upgraded_this_level += 1
                    print(f"    - {layer_name} (sensitivity: {sensitivity:.4f})")

                # Remove upgraded layers from the current level list
                layers_at_current_level = layers_at_current_level[
                    self.layers_per_iteration :
                ]

                del quantized_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"\nCompleted {from_bits}→{to_bits} bit level:")
            print(f"  - Layers upgraded: {layers_upgraded_this_level}")
            print(
                f"  - Remaining layers at {from_bits} bits: {len(layers_at_current_level)}"
            )

            # Check if we've reached max iterations
            if total_iteration >= self.max_iterations:
                print(f"  - Reached maximum iterations ({self.max_iterations})")
                break

            # Print current distribution
            bit_counts = {32: 0, 16: 0, 8: 0, 4: 0}
            for bits in current_config.values():
                bit_counts[bits] += 1

            total_layers = len(current_config)
            print(f"  - Current distribution:")
            print(
                f"    FP32: {bit_counts[32]} ({bit_counts[32]/total_layers*100:.1f}%)"
            )
            print(
                f"    BF16: {bit_counts[16]} ({bit_counts[16]/total_layers*100:.1f}%)"
            )
            print(f"    INT8: {bit_counts[8]} ({bit_counts[8]/total_layers*100:.1f}%)")
            print(f"    INT4: {bit_counts[4]} ({bit_counts[4]/total_layers*100:.1f}%)")

        # Final evaluation
        print(f"\n{'='*50}")
        print("FINAL EVALUATION")
        print(f"{'='*50}")

        quantized_model = self.apply_mixed_precision(current_config)
        final_ppl = perplexity(
            quantized_model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )

        final_perplexity_increase = (final_ppl - baseline_ppl) / baseline_ppl
        print(
            f"Final perplexity: {final_ppl:.4f} (increase: {final_perplexity_increase:.3f})"
        )
        print(f"Total iterations used: {total_iteration}")

        if final_perplexity_increase <= self.max_perplexity_increase:
            print("✓ Final configuration meets perplexity constraint!")
        else:
            print("⚠ Final configuration exceeds perplexity constraint.")
            print(
                "  Consider using a less aggressive initial strategy or higher constraint."
            )

        del quantized_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return current_config

    def apply_mixed_precision(self, config: Dict[str, int]):
        """Apply mixed precision quantization to the model.

        Args:
            config (Dict[str, int]): Layer-wise quantization configuration.

        Returns:
            torch.nn.Module: Quantized model with mixed precision layers.
        """
        print("\n Applying mixed precision quantization...")
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

    def run_full_analysis(self):
        """Run complete mixed precision quantization analysis pipeline.

        Performs sensitivity analysis, generates quantization configuration,
        and evaluates the quantized model's performance on various metrics.

        Returns:
            dict: Comprehensive results including:
                - Sensitivity scores
                - Quantization configuration
                - Performance metrics (perplexity, model size)
                - Benchmark results
                - Quantized model
        """
        print("=" * 60)
        print("RUNNING FULL MIXED PRECISION ANALYSIS")
        print("=" * 60)

        # Get original model metrics
        print("\nEvaluating original model...")
        original_ppl = perplexity(
            self.model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        original_size = get_model_size_mb(self.model)

        # Run original model benchmarks
        print("\nRunning original model benchmarks...")
        original_benchmark_results = {}
        for benchmark_name in self.benchmarking_tasks:
            try:
                bench_name = benchmark_name.split("/")[-1]
                results = evaluate_llm_benchmark(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    benchmark_name=benchmark_name.split("/")[0],
                    task_name=bench_name,
                    num_samples=200,
                    device=self.device,
                )
                if results:
                    original_benchmark_results[bench_name] = results
            except Exception as e:
                print(
                    f"Error running benchmark {benchmark_name} on original model: {e}"
                )

        # Analyze layer sensitivity
        start_time = time.time()
        sensitivity_scores = self.cached_sensitivity_scores(results_dir="results")
        if not sensitivity_scores:
            print("No cached sensitivity scores found. Running analysis...")
            sensitivity_scores = self.analyze_layer_sensitivity()
        end_time = time.time()
        print(f"Sensitivity analysis completed in {end_time - start_time:.2f} seconds.")

        # Get mixed precision config based on sensitivity
        print("\nGenerating mixed precision configuration...")
        if self.use_iterative:
            mp_config = self.get_iterative_config()
        else:
            mp_config = self.get_sensitivity_based_config()

        # Apply quantization
        quantized_model = self.apply_mixed_precision(mp_config)

        # Evaluate quantized model
        print("\nEvaluating quantized model...")
        quantized_ppl = perplexity(
            quantized_model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        quantized_size = get_model_size_mb(quantized_model)

        # Run quantized model benchmarks
        print("\nRunning quantized model benchmarks...")
        quantized_benchmark_results = {}
        for benchmark_name in self.benchmarking_tasks:
            try:
                bench_name = benchmark_name.split("/")[-1]
                results = evaluate_llm_benchmark(
                    model=quantized_model,
                    tokenizer=self.tokenizer,
                    benchmark_name=benchmark_name.split("/")[0],
                    task_name=bench_name,
                    num_samples=200,
                    device=self.device,
                )
                if results:
                    quantized_benchmark_results[bench_name] = results
            except Exception as e:
                print(
                    f"Error running benchmark {benchmark_name} on quantized model: {e}"
                )

        # Package benchmark results
        benchmark_results = {
            "original": original_benchmark_results,
            "quantized": quantized_benchmark_results,
        }

        # Generate report
        results = run_full_analysis_report(
            sensitivity_scores,
            mp_config,
            original_ppl,
            quantized_ppl,
            original_size,
            quantized_size,
            benchmark_results,
        )

        # Add quantized model to results for further use
        results["quantized_model"] = quantized_model

        return results
