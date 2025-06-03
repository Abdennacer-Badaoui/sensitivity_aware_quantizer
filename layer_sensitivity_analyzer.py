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
from quantizer import replace_single_linear_with_target, W8A16LinearLayer, W16A16LinearLayer


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
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                layers[name] = module
        return layers
    
    def _replace_layer_in_model(self, model, layer_name, new_layer):
        """Replace a layer in the model with a new layer instance."""
        parts = layer_name.split(".")
        current = model
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], new_layer)

    def _quantize(
        self,
        tensor: torch.Tensor,
        bits: int = 8,
        per_channel: bool = True,
        symmetric: bool = True,
    ):
        """Quantize a tensor to specified bit precision using symmetric or asymmetric quantization."""
        if bits == 32:
            return tensor
        if bits == 16:
            # Simulate float16 quantization but keep float32 storage
            return tensor.to(torch.float16).to(torch.float32)
        if symmetric:
            return self._symmetric_quantize(tensor, bits, per_channel)
        else:
            return self._asymmetric_quantize(tensor, bits, per_channel)

    def _symmetric_quantize(
        self, tensor: torch.Tensor, bits: int = 8, per_channel: bool = True, unsigned: bool = False
    ):
        """Perform symmetric quantization on a tensor to specified bit precision."""

        if unsigned:
            qmin = 0
            qmax = 2 ** bits - 1
        else:
            qmin = -2 ** (bits - 1)
            qmax = 2 ** (bits - 1) - 1
        if per_channel and tensor.dim() >= 2:
            dims = tuple(range(1, tensor.dim()))
            max_vals = tensor.abs().amax(dim=dims, keepdim=True)
            scale_denominator = (qmax - qmin) / 2
            scale = max_vals / scale_denominator
            zero_point = 0  # Symmetric quantization uses zero_point = 0
            q_tensor = torch.round(tensor / scale + zero_point)
            q_tensor = torch.clamp(q_tensor, qmin, qmax)
            return (q_tensor - zero_point) * scale
        else:
            max_val = tensor.abs().max().item()
            scale = max_val / ((qmax - qmin) / 2) if (qmax - qmin) != 0 else 1.0
            zero_point = 0 if not unsigned else qmin
            q_tensor = torch.round(tensor / scale + zero_point)
            q_tensor = torch.clamp(q_tensor, qmin, qmax)
            return (q_tensor - zero_point) * scale

    def _asymmetric_quantize(self, tensor: torch.Tensor, bits: int = 8, per_channel: bool = True, unsigned: bool = False):
        """Perform asymmetric quantization using only bits (no dtype/iinfo)."""

        if unsigned:
            qmin = 0
            qmax = 2 ** bits - 1
        else:
            qmin = -2 ** (bits - 1)
            qmax = 2 ** (bits - 1) - 1
        if per_channel and tensor.dim() >= 2:
            dims = tuple(range(1, tensor.dim()))
            rmin = tensor.amin(dim=dims, keepdim=True)
            rmax = tensor.amax(dim=dims, keepdim=True)
            scale = (rmax - rmin) / (qmax - qmin)
            valid_scale = scale != 0
            zero_point = torch.where(
                valid_scale,
                qmin - (rmin / scale),
                torch.tensor(0.0, device=tensor.device)
            )
            zero_point = zero_point.round().to(torch.int32)
            q_tensor = torch.where(
                valid_scale,
                torch.round(tensor / scale + zero_point),
                zero_point  # When scale=0, all values are rmin, quantized to zero_point
            )
            q_tensor = torch.clamp(q_tensor, qmin, qmax)
            return (q_tensor - zero_point) * scale
        else:
            rmin, rmax = tensor.min().item(), tensor.max().item()
            scale = (rmax - rmin) / (qmax - qmin) if (qmax - qmin) != 0 else 1.0
            zero_point = qmin - rmin / scale if scale != 0 else 0
            zero_point = int(round(zero_point))
            q_tensor = torch.round(tensor / scale + zero_point)
            q_tensor = torch.clamp(q_tensor, qmin, qmax)
            return (q_tensor - zero_point) * scale

    def _quantize_layer(
        self, layer: nn.Module, bits: int = 8, preserve_magnitude: bool = False
    ):
        """Quantize a neural network layer's weights and biases to specified bit precision."""
        if not hasattr(layer, "weight"):
            return layer
        quantized_layer = copy.deepcopy(layer)
        original_weight = layer.weight.data
        quantized_weight = self._quantize(
            original_weight, bits, per_channel=True, symmetric=False
        )
        if preserve_magnitude:
            original_norm = torch.norm(original_weight)
            quantized_norm = torch.norm(quantized_weight)
            if quantized_norm > 1e-8:
                scale_factor = original_norm / quantized_norm
                quantized_weight = quantized_weight * scale_factor
        quantized_layer.weight.data = quantized_weight
        if hasattr(layer, "bias") and layer.bias is not None:
            bias_bits = min(bits + 4, 16)
            quantized_layer.bias.data = self._quantize(
                layer.bias.data, bias_bits, per_channel=False, symmetric=False
            )
        return quantized_layer

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

    def analyze_layer_sensitivity(self, bits: int = 8, sensitivity_method: str = "divergence"):
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
                layer_name
            )

            quantized_outputs = self._compute_quantized(model_copy)
            if sensitivity_method == "divergence":
                sensitivity_scores[layer_name] = SensitivityMetrics.compute_divergence_based_sensitivities(baseline_outputs, quantized_outputs)
            elif sensitivity_method == "hessian":
                sensitivity_scores[layer_name] = SensitivityMetrics.compute_hessian_based_sensitivities(model_copy, layer_name, layer_module, self.calibration_data) 

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

    def get_mixed_precision_config(
        self, target_bits: float = 6.0, bit_options: list = None
    ):
        """Generate mixed precision configuration based on layer sensitivity scores."""
        if bit_options is None:
            bit_options = [32, 16, 8]

        # Sort layers by sensitivity (most sensitive first)
        sorted_layers = sorted(
            self.sensitivity_scores.items(), key=lambda x: x[1], reverse=True
        )
        n_layers = len(sorted_layers)

        # Initialize all layers with minimum bit
        min_bit = min(bit_options)
        config = {layer: min_bit for layer, _ in sorted_layers}
        current_total = min_bit * n_layers
        target_total = target_bits * n_layers

        # Sort bit options descending for priority checking
        sorted_bits = sorted(bit_options, reverse=True)

        # Distribute bits
        for layer_name, _ in sorted_layers:
            if current_total >= target_total:
                break

            current_bit = config[layer_name]
            remaining_budget = target_total - current_total

            # Try highest bits first
            for bit in sorted_bits:
                if bit <= current_bit:
                    continue

                bit_increase = bit - current_bit
                if bit_increase <= remaining_budget:
                    # Apply upgrade
                    config[layer_name] = bit
                    current_total += bit_increase
                    break

        return config

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

                if bits==16:
                    replace_single_linear_with_target(
                        quantized_model,
                        W16A16LinearLayer,  
                        layer_name
                    )

                elif bits==8:
                    replace_single_linear_with_target(
                        quantized_model,
                        W8A16LinearLayer,  
                        layer_name
                    )

            except Exception as e:
                print(f"Warning: Could not quantize layer {layer_name}: {str(e)}")
                continue
        return quantized_model

    def run_full_analysis(self, target_avg_bits: float = 6.0, sensitivity_method: str = "divergence"):
        """Run full analysis including sensitivity analysis, mixed precision configuration, and evaluation."""
        sensitivity_scores = self.analyze_layer_sensitivity(
            sensitivity_method=sensitivity_method
        )  # analyze layer sensitivity
        mp_config = self.get_mixed_precision_config(
            target_bits=target_avg_bits
        )  # get mixed precision config
        original_ppl = evaluate_model(self.model, self.tokenizer, self.eval_data, self.eval_num_samples, self.device)  # evaluate original model
        original_size = get_model_size_mb(self.model)  # get original model size
        quantized_model = self.apply_mixed_precision(
            mp_config
        )  # apply mixed precision quantization
        quantized_ppl = evaluate_model(quantized_model, self.tokenizer, self.eval_data, self.eval_num_samples, self.device)  # evaluate quantized model
        quantized_size = get_model_size_mb(
            quantized_model, mp_config
        )  # get quantized model size

        results = run_full_analysis_report(
            sensitivity_scores,
            mp_config,
            original_ppl,
            quantized_ppl,
            original_size,
            quantized_size,
            target_avg_bits,
            sensitivity_method=sensitivity_method,
        )

        return results