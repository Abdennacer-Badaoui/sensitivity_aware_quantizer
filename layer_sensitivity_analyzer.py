import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sensitivity_metrics import SensitivityMetrics
from reporter import run_full_analysis_report
from model_utils import perplexity, get_model_size_mb
from quantizer import (
    W4A16LinearLayer,
    W8A16LinearLayer,
    W16A16LinearLayer,
    dequantize_model_to_standard_format,
    replace_single_linear_with_target,
)


class LayerSensitivityAnalyzer:
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        profiling_data=None,
        eval_data=None,
        profiling_num_samples: int = 100,
        eval_num_samples: int = 100,
        mode: str = "per_layer",
        batch_size: int = 128,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        sensitivity_method="divergence",
        config_strategy="adaptive_threshold",
        use_iterative=True,
        max_perplexity_increase=0.1,
        layers_per_iteration=3,
        max_iterations=50,
        device: str = "auto",
    ):
        """Initialize a LayerSensitivityAnalyzer for quantizing transformer models.

        This class analyzes the sensitivity of different layers in a transformer model
        and determines the optimal quantization strategy while maintaining model performance.

        Args:
            model_name (str): HuggingFace model identifier. Defaults to "microsoft/DialoGPT-small".
            profiling_data: Pre-prepared profiling dataset. If None, will be created from dataset_name.
            eval_data: Pre-prepared evaluation dataset. If None, will be created from dataset_name.
            profiling_num_samples (int): Number of samples for profiling. Defaults to 100.
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
            device (str): Computing device ("cuda", "cpu", or "auto"). Defaults to "auto".


        Note:
            The analyzer supports various quantization strategies and can be configured
            for different trade-offs between model size and performance.
        """
        # Model loading
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

        # Profiling dataset preparation (profile model behavior under quantization)
        self.profiling_data = profiling_data or self._prepare_hf_dataset(
            split="train",
            num_samples=profiling_num_samples,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split_name="profiling",
        )
        self.eval_data = eval_data or self._prepare_hf_dataset(
            split="validation",
            num_samples=eval_num_samples,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split_name="evaluation",
        )
        self.profiling_num_samples = profiling_num_samples
        self.eval_num_samples = eval_num_samples
        self.batch_size = batch_size
        self.mode = mode
        self.sensitivity_method = sensitivity_method
        self.config_strategy = config_strategy
        self.use_iterative = use_iterative
        self.max_perplexity_increase = max_perplexity_increase
        self.layers_per_iteration = layers_per_iteration
        self.max_iterations = max_iterations
        self.sensitivity_scores = {}

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
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    def _get_modules(self):
        """Extract all quantizable linear layers from the model.

        Returns:
            dict: Dictionary mapping layer names to their corresponding nn.Linear modules.
        """
        layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                layers[name] = module
        return layers
    
    def _get_block_groups(self):
        """Extract block groups from the model based on layer names.
        
        Returns:
            dict: Dictionary mapping block names to lists of layer names in that block.
        """
        layers = self._get_modules()
        block_groups = {}
        
        for layer_name in layers.keys():
            # Extract block identifier from layer name
            parts = layer_name.split('.')
            block_name = None
            
            # Look for patterns like "model.layers.0", "model.decoder.layers.8", etc.
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0:
                    # Found a numeric part, construct block name
                    block_name = '.'.join(parts[:i+1])
                    break
            
            if block_name:
                if block_name not in block_groups:
                    block_groups[block_name] = []
                block_groups[block_name].append(layer_name)
        
        return block_groups

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
                    activations[name] = output.detach().cpu()
                elif isinstance(output, tuple) and len(output) > 0:
                    activations[name] = output[0].detach().cpu()

            return hook

        layers = self._get_modules()
        for name, module in layers.items():
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
        with torch.no_grad():
            for i in range(0, len(self.profiling_data["input_ids"]), self.batch_size):
                end_idx = min(
                    i + self.batch_size, len(self.profiling_data["input_ids"])
                )
                batch_inputs = {
                    "input_ids": self.profiling_data["input_ids"][i:end_idx],
                    "attention_mask": self.profiling_data["attention_mask"][i:end_idx],
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

        # Clear activation data
        del activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return stats
    
    def _apply_block_quantization(self, model, block_name, layer_names, bits):
        """Apply quantization to all layers in a block.
        
        Args:
            model: Model to quantize
            block_name: Name of the block
            layer_names: List of layer names in the block
            bits: Number of bits for quantization
        """
        if bits == 32:
            return  # No quantization needed
        
        target_layer_class = {
            16: W16A16LinearLayer,
            8: W8A16LinearLayer,
            4: W4A16LinearLayer
        }[bits]
        
        for layer_name in layer_names:
            try:
                replace_single_linear_with_target(model, target_layer_class, layer_name)
            except Exception as e:
                print(f"Warning: Could not quantize layer {layer_name} in block {block_name}: {str(e)}")

    @staticmethod
    def _sensitivity_worker(
        gpu_id: int,
        model_state_dict: Dict,
        model_config,
        profiling_data: Dict,
        layer_names: List[str],
        batch_size: int,
        method: str,
        output_queue: mp.Queue,
    ):
        try:
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Worker on GPU {gpu_id} started with {len(layer_names)} layers")

            # Load model on this GPU
            model = AutoModelForCausalLM.from_config(model_config)
            model.load_state_dict(model_state_dict)
            model = model.to(device)
            model.eval()

            # Move profiling data to this GPU
            calib_data = {
                "input_ids": profiling_data["input_ids"].to(device),
                "attention_mask": profiling_data["attention_mask"].to(device),
            }

            # Compute baseline outputs
            baseline_outputs = []
            with torch.no_grad():
                for i in range(0, len(calib_data["input_ids"]), batch_size):
                    end_idx = min(i + batch_size, len(calib_data["input_ids"]))
                    batch_inputs = {
                        "input_ids": calib_data["input_ids"][i:end_idx],
                        "attention_mask": calib_data["attention_mask"][i:end_idx],
                    }
                    outputs = model(**batch_inputs)
                    baseline_outputs.append(
                        {
                            "logits": outputs.logits.cpu(),
                            "last_hidden_state": outputs.last_hidden_state.cpu()
                            if hasattr(outputs, "last_hidden_state")
                            else None,
                        }
                    )

            # Process assigned layers
            layer_scores = {}

            for layer_name in tqdm(layer_names, desc="Sensitivity Analysis"):
                model_copy = copy.deepcopy(model)
                replace_single_linear_with_target(
                    model_copy, W8A16LinearLayer, layer_name
                )

                quantized_outputs = []
                with torch.no_grad():
                    for i in range(0, len(calib_data["input_ids"]), batch_size):
                        end_idx = min(i + batch_size, len(calib_data["input_ids"]))
                        batch_inputs = {
                            "input_ids": calib_data["input_ids"][i:end_idx],
                            "attention_mask": calib_data["attention_mask"][i:end_idx],
                        }
                        outputs = model_copy(**batch_inputs)
                        quantized_outputs.append(
                            {
                                "logits": outputs.logits.cpu(),
                                "last_hidden_state": outputs.last_hidden_state.cpu()
                                if hasattr(outputs, "last_hidden_state")
                                else None,
                            }
                        )

                # Compute sensitivity
                score = SensitivityMetrics.compute_divergence_based_sensitivities(
                    baseline_outputs, quantized_outputs
                )
                layer_scores[layer_name] = score

                del model_copy
                torch.cuda.empty_cache()

            output_queue.put(layer_scores)
            print(f"Worker on GPU {gpu_id} completed successfully")

            # Clean up GPU memory
            del model
            del calib_data
            del baseline_outputs
            del quantized_outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in worker on GPU {gpu_id}: {str(e)}")
            output_queue.put({})
            torch.cuda.empty_cache()

    def analyze_layer_sensitivity(self, bits: int = 8):
        """Analyze sensitivity at layer or block level based on mode setting."""
        if self.mode == "per_layer":
            return self._analyze_per_layer_sensitivity(bits)
        elif self.mode == "per_block":
            return self._analyze_per_block_sensitivity(bits)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'per_layer' or 'per_block'")
        
    def _analyze_per_layer_sensitivity(self, bits: int = 8):
        """Original per-layer analysis (renamed from analyze_layer_sensitivity)."""
        if bits != 8:
            raise ValueError("Parallel processing only supports 8-bit quantization")

        print(f"Parallel layer sensitivity analysis with {bits}-bit quantization")
        baseline_outputs = self._compute_baseline()
        activation_stats = self._compute_activation_statistics()
        layers = list(self._get_modules().keys())

        # Get available GPUs (exclude display GPU)
        available_gpus = []
        for i in range(
            0, torch.cuda.device_count()
        ):  # excluding 0 for the moment (Should be fixed because we usually get CUDA OOM on device 0)
            total_mem = torch.cuda.get_device_properties(i).total_memory
            if total_mem >= 10 * 1024**3:  # At least 10GB memory
                available_gpus.append(i)

        if not available_gpus:
            print("No suitable GPUs found, falling back to sequential processing")
            return self._analyze_layer_sensitivity_sequential(bits)

        print(f"Using GPUs: {available_gpus}")
        num_gpus = len(available_gpus)
        layers_per_gpu = (len(layers) + num_gpus - 1) // num_gpus
        layer_groups = [
            layers[i : i + layers_per_gpu]
            for i in range(0, len(layers), layers_per_gpu)
        ]

        # Prepare data for workers
        profiling_data_cpu = {
            "input_ids": self.profiling_data["input_ids"].cpu(),
            "attention_mask": self.profiling_data["attention_mask"].cpu(),
        }

        # Setup multiprocessing
        mp.set_start_method("spawn", force=True)
        output_queue = mp.Queue()
        processes = []

        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        model_config = self.model.config

        # Start worker processes
        for i, gpu_id in enumerate(available_gpus):
            if i < len(layer_groups):
                p = mp.Process(
                    target=self._sensitivity_worker,
                    args=(
                        gpu_id,
                        model_state_dict,
                        model_config,
                        profiling_data_cpu,
                        layer_groups[i],
                        self.batch_size,
                        self.sensitivity_method,
                        output_queue,
                    ),
                )
                p.start()
                processes.append(p)

        # Collect results
        sensitivity_scores = {}
        for _ in range(len(processes)):
            result = output_queue.get()
            sensitivity_scores.update(result)

        # Clean up
        for p in processes:
            p.join()

        # Clean up profiling data in main process
        del profiling_data_cpu
        torch.cuda.empty_cache()

        # Normalize with activation statistics
        max_magnitude = max(stat["magnitude"] for stat in activation_stats.values())
        for layer_name, score in sensitivity_scores.items():
            if layer_name in activation_stats:
                normalized_magnitude = (
                    activation_stats[layer_name]["magnitude"] / max_magnitude
                )
                sensitivity_scores[layer_name] *= 1.0 + 0.5 * normalized_magnitude

        self.sensitivity_scores = sensitivity_scores
        return sensitivity_scores

    def _analyze_per_block_sensitivity(self, bits: int = 8):
        """Analyze sensitivity at block level instead of individual layers."""
        print(f"Block-level sensitivity analysis with {bits}-bit quantization")
        
        baseline_outputs = self._compute_baseline()
        activation_stats = self._compute_activation_statistics()
        block_groups = self._get_block_groups()
        
        print(f"Found {len(block_groups)} blocks to analyze")
        for block_name, layer_names in block_groups.items():
            print(f"  Block {block_name}: {len(layer_names)} layers")
        
        sensitivity_scores = {}
        
        for block_name, layer_names in tqdm(block_groups.items(), desc="Block Sensitivity Analysis"):
            # Create a copy of the model and quantize the entire block
            model_copy = copy.deepcopy(self.model)
            self._apply_block_quantization(model_copy, block_name, layer_names, bits)
            
            # Compute outputs from the quantized model
            quantized_outputs = self._compute_quantized(model_copy)
            
            # Compute sensitivity for the block
            if self.sensitivity_method == "divergence":
                block_sensitivity = SensitivityMetrics.compute_divergence_based_sensitivities(
                    baseline_outputs, quantized_outputs
                )
            elif self.sensitivity_method == "hessian":
                # For hessian, we'll use the average of all layers in the block
                block_sensitivity = 0.0
                layer_count = 0
                layers = self._get_modules()
                
                input_data = {
                    "input_ids": self.profiling_data["input_ids"][:1],
                    "attention_mask": self.profiling_data["attention_mask"][:1],
                }
                
                for layer_name in layer_names:
                    if layer_name in layers:
                        layer_sensitivity = SensitivityMetrics.compute_hessian_based_sensitivities(
                            self.model,
                            self.tokenizer,
                            layers[layer_name].weight,
                            input_data,
                        )
                        block_sensitivity += layer_sensitivity
                        layer_count += 1
                
                if layer_count > 0:
                    block_sensitivity /= layer_count
            
            # Normalize with activation statistics (use average of block layers)
            block_magnitude = 0.0
            valid_layers = 0
            for layer_name in layer_names:
                if layer_name in activation_stats:
                    block_magnitude += activation_stats[layer_name]["magnitude"]
                    valid_layers += 1
            
            if valid_layers > 0:
                block_magnitude /= valid_layers
                max_magnitude = max(stat["magnitude"] for stat in activation_stats.values())
                normalized_magnitude = block_magnitude / max_magnitude
                block_sensitivity *= 1.0 + 0.5 * normalized_magnitude
            
            sensitivity_scores[block_name] = block_sensitivity
            
            # Clean up
            del model_copy
            del quantized_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.sensitivity_scores = sensitivity_scores
        return sensitivity_scores

    def _analyze_layer_sensitivity_sequential(self, bits: int = 8):
        """Analyze each layer's sensitivity to quantization (Sequantial).

        Quantizes each layer individually and measures its impact on model outputs
        using the specified sensitivity method.

        Args:
            bits (int): Number of bits for quantization testing. Defaults to 8.

        Returns:
            dict: Dictionary mapping layer names to their sensitivity scores.
        """
        print(f"Sequential layer sensitivity analysis with {bits}-bit quantization")
        baseline_outputs = self._compute_baseline()
        activation_stats = self._compute_activation_statistics()
        layers = self._get_modules()
        sensitivity_scores = {}

        input_data = {
            "input_ids": self.profiling_data["input_ids"][:1],
            "attention_mask": self.profiling_data["attention_mask"][:1],
        }

        for layer_name, layer_module in tqdm(
            layers.items(), desc="Sensitivity Analysis"
        ):
            model_copy = copy.deepcopy(self.model)
            replace_single_linear_with_target(
                model_copy,
                W8A16LinearLayer if bits == 8 else W4A16LinearLayer,
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
                    input_data,
                )

            # Normalize activation magnitude
            if layer_name in activation_stats:
                max_magnitude = max(
                    stat["magnitude"] for stat in activation_stats.values()
                )
                normalized_magnitude = (
                    activation_stats[layer_name]["magnitude"] / max_magnitude
                )
                sensitivity_scores[layer_name] *= 1.0 + 0.5 * normalized_magnitude

            del model_copy
            del quantized_outputs
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
                    and data["profiling_num_samples"] == self.profiling_num_samples
                    and data["sensitivity_method"] == self.sensitivity_method
                ):
                    self.sensitivity_scores = data["sensitivity_scores"]
                    print(
                        f"Loaded cached sensitivity scores from {file} (model_name: {self.model_name}, sensitivity_method: {self.sensitivity_method}, profiling_num_samples: {self.profiling_num_samples})"
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
            for i in range(0, len(self.profiling_data["input_ids"]), self.batch_size):
                end_idx = min(
                    i + self.batch_size, len(self.profiling_data["input_ids"])
                )
                batch_inputs = {
                    "input_ids": self.profiling_data["input_ids"][i:end_idx],
                    "attention_mask": self.profiling_data["attention_mask"][i:end_idx],
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
        self._cached_baseline_outputs = baseline_outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            for i in range(0, len(self.profiling_data["input_ids"]), self.batch_size):
                end_idx = min(
                    i + self.batch_size, len(self.profiling_data["input_ids"])
                )
                batch_inputs = {
                    "input_ids": self.profiling_data["input_ids"][i:end_idx],
                    "attention_mask": self.profiling_data["attention_mask"][i:end_idx],
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return quantized_outputs

    def get_sensitivity_based_config(self):
        """Generate mixed precision configuration based on sensitivity scores."""
        if not self.sensitivity_scores:
            raise ValueError(
                "No sensitivity scores available. Run analyze_layer_sensitivity first."
            )

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
            high_threshold = mean_sens + 0.5 * std_sens
            medium_threshold = mean_sens
            low_threshold = mean_sens - 0.5 * std_sens

            for name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= high_threshold:
                    config[name] = 32
                elif sensitivity >= medium_threshold:
                    config[name] = 16
                elif sensitivity >= low_threshold:
                    config[name] = 8
                else:
                    config[name] = 4

        elif self.config_strategy == "percentile":
            p75 = np.percentile(sensitivities, 75)
            p50 = np.percentile(sensitivities, 50)
            p25 = np.percentile(sensitivities, 25)

            for name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= p75:
                    config[name] = 32
                elif sensitivity >= p50:
                    config[name] = 16
                elif sensitivity >= p25:
                    config[name] = 8
                else:
                    config[name] = 4

        elif self.config_strategy == "exponential":
            norm_sensitivities = {}
            sens_range = max_sens - min_sens
            if sens_range == 0:
                sens_range = 1

            for name, sensitivity in self.sensitivity_scores.items():
                normalized = (sensitivity - min_sens) / sens_range
                exp_score = np.exp(2 * normalized) / np.exp(2)
                norm_sensitivities[name] = exp_score

            for name, norm_sens in norm_sensitivities.items():
                if norm_sens >= 0.75:
                    config[name] = 32
                elif norm_sens >= 0.5:
                    config[name] = 16
                elif norm_sens >= 0.25:
                    config[name] = 8
                else:
                    config[name] = 4

        elif self.config_strategy == "conservative":
            low_threshold = mean_sens - 0.25 * std_sens
            medium_threshold = mean_sens + 0.25 * std_sens
            high_threshold = mean_sens + 0.75 * std_sens

            for name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= high_threshold:
                    config[name] = 32
                elif sensitivity >= medium_threshold:
                    config[name] = 32
                elif sensitivity >= low_threshold:
                    config[name] = 16
                else:
                    config[name] = 8

        elif self.config_strategy == "aggressive":
            high_threshold = mean_sens + 1.0 * std_sens
            medium_threshold = mean_sens + 0.25 * std_sens
            low_threshold = mean_sens - 0.25 * std_sens

            for name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= high_threshold:
                    config[name] = 32
                elif sensitivity >= medium_threshold:
                    config[name] = 16
                elif sensitivity >= low_threshold:
                    config[name] = 8
                else:
                    config[name] = 4

        elif self.config_strategy == "int8_only":
            for name in self.sensitivity_scores.keys():
                config[name] = 8

        elif self.config_strategy == "int4_only":
            for name in self.sensitivity_scores.keys():
                config[name] = 4

        # Count configurations
        if self.mode == "per_block":
            # For block mode, count the total layers affected
            block_groups = self._get_block_groups()
            bit_counts = {32: 0, 16: 0, 8: 0, 4: 0}
            total_layers = 0
            
            for block_name, bits in config.items():
                if block_name in block_groups:
                    layer_count = len(block_groups[block_name])
                    bit_counts[bits] += layer_count
                    total_layers += layer_count
            
            print(f"\nBit allocation distribution (block mode):")
            print(f"FP32: {bit_counts[32]} layers ({bit_counts[32]/total_layers*100:.1f}%)")
            print(f"BF16: {bit_counts[16]} layers ({bit_counts[16]/total_layers*100:.1f}%)")
            print(f"INT8: {bit_counts[8]} layers ({bit_counts[8]/total_layers*100:.1f}%)")
            print(f"INT4: {bit_counts[4]} layers ({bit_counts[4]/total_layers*100:.1f}%)")
        else:
            # Original layer counting for per_layer mode
            bit_counts = {32: 0, 16: 0, 8: 0, 4: 0}
            for bits in config.values():
                bit_counts[bits] += 1

            total_layers = len(config)
            print(f"\nBit allocation distribution (layer mode):")
            print(f"FP32: {bit_counts[32]} layers ({bit_counts[32]/total_layers*100:.1f}%)")
            print(f"BF16: {bit_counts[16]} layers ({bit_counts[16]/total_layers*100:.1f}%)")
            print(f"INT8: {bit_counts[8]} layers ({bit_counts[8]/total_layers*100:.1f}%)")
            print(f"INT4: {bit_counts[4]} layers ({bit_counts[4]/total_layers*100:.1f}%)")

        return config

    def get_iterative_config(self):
        """Iteratively refine quantization configuration to meet performance constraints."""
        print(f"Starting progressive iterative configuration in {self.mode} mode:")
        print(f"  - Max perplexity increase: {self.max_perplexity_increase}")
        
        if self.mode == "per_layer":
            print(f"  - Layers per iteration: {self.layers_per_iteration}")
        else:  # per_block mode
            print(f"  - Blocks per iteration: 1 (block-by-block)")
        
        print(f"  - Max total iterations: {self.max_iterations}")

        baseline_ppl = perplexity(
            self.model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        print(f"Baseline perplexity: {baseline_ppl:.4f}")

        current_config = self.get_sensitivity_based_config()
        sorted_items = sorted(
            self.sensitivity_scores.items(), key=lambda x: x[1], reverse=True
        )
        upgrade_levels = [(4, 8), (8, 16), (16, 32)]
        total_iteration = 0

        for from_bits, to_bits in upgrade_levels:
            print(f"\n{'='*50}")
            print(f"UPGRADING FROM {from_bits} TO {to_bits} BITS")
            print(f"{'='*50}")

            items_at_current_level = [
                (name, sensitivity)
                for name, sensitivity in sorted_items
                if current_config[name] == from_bits
            ]

            if not items_at_current_level:
                print(f"No items at {from_bits} bits. Skipping this level.")
                continue

            items_upgraded_this_level = 0
            
            if self.mode == "per_block":
                block_groups = self._get_block_groups()
                total_layers_at_level = sum(
                    len(block_groups.get(name, [])) for name, _ in items_at_current_level
                )
                print(f"Found {len(items_at_current_level)} blocks at {from_bits} bits ({total_layers_at_level} total layers)")
            else:
                print(f"Found {len(items_at_current_level)} layers at {from_bits} bits")

            while items_at_current_level and total_iteration < self.max_iterations:
                total_iteration += 1
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

                if perplexity_increase <= self.max_perplexity_increase:
                    print(f"  ✓ Perplexity constraint satisfied!")
                    del quantized_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return current_config

                # Determine how many items to upgrade based on mode
                if self.mode == "per_block":
                    # For per_block mode: upgrade one block at a time
                    items_to_upgrade = items_at_current_level[:1]
                    upgrade_description = "block"
                else:
                    # For per_layer mode: upgrade layers_per_iteration layers at a time
                    items_to_upgrade = items_at_current_level[:self.layers_per_iteration]
                    upgrade_description = f"{len(items_to_upgrade)} layers"
                
                print(f"  Upgrading {upgrade_description}:")
                
                for name, sensitivity in items_to_upgrade:
                    current_config[name] = to_bits
                    items_upgraded_this_level += 1
                    if self.mode == "per_block":
                        block_groups = self._get_block_groups()
                        layer_count = len(block_groups.get(name, []))
                        print(f"    - {name} ({layer_count} layers, sensitivity: {sensitivity:.4f})")
                    else:
                        print(f"    - {name} (sensitivity: {sensitivity:.4f})")

                # Remove upgraded items from the list
                if self.mode == "per_block":
                    items_at_current_level = items_at_current_level[1:]  # Remove 1 block
                else:
                    items_at_current_level = items_at_current_level[self.layers_per_iteration:]  # Remove layers_per_iteration layers

                del quantized_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"\nCompleted {from_bits}→{to_bits} bit level:")
            print(f"  - Items upgraded: {items_upgraded_this_level}")
            print(f"  - Remaining items at {from_bits} bits: {len(items_at_current_level)}")

            if total_iteration >= self.max_iterations:
                print(f"  - Reached maximum iterations ({self.max_iterations})")
                break

            # Show current distribution
            if self.mode == "per_block":
                block_groups = self._get_block_groups()
                bit_counts = {32: 0, 16: 0, 8: 0, 4: 0}
                total_layers = 0
                
                for block_name, bits in current_config.items():
                    if block_name in block_groups:
                        layer_count = len(block_groups[block_name])
                        bit_counts[bits] += layer_count
                        total_layers += layer_count
            else:
                bit_counts = {32: 0, 16: 0, 8: 0, 4: 0}
                for bits in current_config.values():
                    bit_counts[bits] += 1
                total_layers = len(current_config)

            print(f"  - Current distribution:")
            print(f"    FP32: {bit_counts[32]} ({bit_counts[32]/total_layers*100:.1f}%)")
            print(f"    BF16: {bit_counts[16]} ({bit_counts[16]/total_layers*100:.1f}%)")
            print(f"    INT8: {bit_counts[8]} ({bit_counts[8]/total_layers*100:.1f}%)")
            print(f"    INT4: {bit_counts[4]} ({bit_counts[4]/total_layers*100:.1f}%)")

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
        print(f"Final perplexity: {final_ppl:.4f} (increase: {final_perplexity_increase:.3f})")
        print(f"Total iterations used: {total_iteration}")

        if final_perplexity_increase <= self.max_perplexity_increase:
            print("✓ Final configuration meets perplexity constraint!")
        else:
            print("⚠ Final configuration exceeds perplexity constraint.")
            print("  Consider using a less aggressive initial strategy or higher constraint.")

        del quantized_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return current_config

    def apply_mixed_precision(self, config: Dict[str, int]):
        """Apply mixed precision quantization to the model."""
        print(f"\nApplying mixed precision quantization in {self.mode} mode...")
        quantized_model = copy.deepcopy(self.model)
        
        if self.mode == "per_layer":
            # Original per-layer quantization
            for layer_name, bits in config.items():
                if not layer_name.strip():
                    continue

                try:
                    parts = layer_name.split(".")
                    current = self.model
                    for part in parts:
                        if not part:
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
                    else:
                        replace_single_linear_with_target(
                            quantized_model, W4A16LinearLayer, layer_name
                        )

                except Exception as e:
                    print(f"Warning: Could not quantize layer {layer_name}: {str(e)}")
                    continue
        
        elif self.mode == "per_block":
            # Block-level quantization
            block_groups = self._get_block_groups()
            
            for block_name, bits in config.items():
                if block_name in block_groups:
                    layer_names = block_groups[block_name]
                    print(f"  Quantizing block {block_name} to {bits} bits ({len(layer_names)} layers)")
                    self._apply_block_quantization(quantized_model, block_name, layer_names, bits)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                - Quantized model
        """
        print("=" * 60)
        print("RUNNING FULL MIXED PRECISION ANALYSIS")
        print("=" * 60)

        print("\nEvaluating original model...")
        original_ppl = perplexity(
            self.model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        original_size = get_model_size_mb(self.model)

        start_time = time.time()
        sensitivity_scores = self.cached_sensitivity_scores(results_dir="results")
        if not sensitivity_scores:
            print("No cached sensitivity scores found. Running analysis...")
            if self.sensitivity_method == "divergence":
                sensitivity_scores = self.analyze_layer_sensitivity()
            elif self.sensitivity_method == "hessian":
                sensitivity_scores = self._analyze_layer_sensitivity_sequential(bits=8)
            else:
                raise ValueError(
                    f"Unknown sensitivity method: {self.sensitivity_method}"
                )
        end_time = time.time()
        print(f"Sensitivity analysis completed in {end_time - start_time:.2f} seconds.")

        print("\nGenerating mixed precision configuration...")
        if self.use_iterative:
            mp_config = self.get_iterative_config()
        else:
            mp_config = self.get_sensitivity_based_config()

        quantized_model = self.apply_mixed_precision(mp_config)

        print("\nEvaluating quantized model...")
        quantized_ppl = perplexity(
            quantized_model,
            self.tokenizer,
            self.eval_data,
            self.eval_num_samples,
            self.device,
        )
        quantized_size = get_model_size_mb(quantized_model)

        results = run_full_analysis_report(
            sensitivity_scores,
            self.mode,
            mp_config,
            original_ppl,
            quantized_ppl,
            original_size,
            quantized_size,
        )

        results["quantized_model"] = quantized_model

        # Clean up before saving
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        quantized_model_standard_format = dequantize_model_to_standard_format(
            quantized_model
        )
        output_path = f"quantized_models/{self.model_name}_{self.sensitivity_method}_{self.profiling_num_samples}_{self.config_strategy}_{self.use_iterative}"
        quantized_model_standard_format.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Final cleanup
        del quantized_model_standard_format
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
