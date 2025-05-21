from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import copy
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json
from utils import get_model_size_mb

class LayerSensitivityAnalyzer:
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        device: str = "auto",
        calibration_data=None,
        eval_data=None,
        calibration_num_samples: int = 1,
        eval_num_samples: int = 1,
        batch_size: int = 256,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
    ):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        ).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.calibration_data = calibration_data or self._prepare_hf_dataset(
            split="train", num_samples=calibration_num_samples, dataset_name=dataset_name, dataset_config=dataset_config, split_name="calibration"
        )
        self.eval_data = eval_data or self._prepare_hf_dataset(
            split="validation", num_samples=eval_num_samples, dataset_name=dataset_name, dataset_config=dataset_config, split_name="evaluation"
        )
        self.sensitivity_scores = {}

    def _prepare_hf_dataset(self, split: str, num_samples: int, dataset_name: str, dataset_config: str, split_name: str):
        print(f"Preparing {split_name} data from HuggingFace dataset: {dataset_name}/{dataset_config} [{split}] ...")
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
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device)
        }

    def _get_layer_modules(self):
        layers = {}
        for name, module in self.model.named_modules():
            if ("transformer.h." in name and any(x in name for x in [".c_attn", ".c_proj", ".c_fc"])) or \
                "lm_head" in name:
                layers[name] = module
        return layers

    def _symmetric_quantize(self, tensor: torch.Tensor, bits: int = 8, per_channel: bool = True):
        if bits >= 16:
            return tensor
        if per_channel and tensor.dim() >= 2:
            axis = 0
            shape_for_scale = [1] * tensor.dim()
            shape_for_scale[axis] = tensor.shape[axis]
        else:
            axis = None
            shape_for_scale = [1] * tensor.dim()
        if axis is not None:
            abs_max = tensor.abs().amax(dim=tuple(i for i in range(tensor.dim()) if i != axis), keepdim=True)
        else:
            abs_max = tensor.abs().max()
        eps = 1e-8
        abs_max = torch.clamp(abs_max, min=eps)
        qmax = 2 ** (bits - 1) - 1
        scale = abs_max / qmax
        quantized = torch.round(tensor / scale).clamp(-qmax - 1, qmax)
        dequantized = quantized * scale
        return dequantized

    def _quantize_layer(self, layer: nn.Module, bits: int = 8, preserve_magnitude: bool = False):
        if not hasattr(layer, 'weight'):
            return layer
        quantized_layer = copy.deepcopy(layer)
        original_weight = layer.weight.data
        quantized_weight = self._symmetric_quantize(original_weight, bits, per_channel=True)
        if preserve_magnitude:
            original_norm = torch.norm(original_weight)
            quantized_norm = torch.norm(quantized_weight)
            if quantized_norm > 1e-8:
                scale_factor = original_norm / quantized_norm
                quantized_weight = quantized_weight * scale_factor
        quantized_layer.weight.data = quantized_weight
        if hasattr(layer, 'bias') and layer.bias is not None:
            bias_bits = min(bits + 4, 16)
            quantized_layer.bias.data = self._symmetric_quantize(
                layer.bias.data, bias_bits, per_channel=False
            )
        return quantized_layer

    def _compute_activation_statistics(self):
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
                end_idx = min(i + self.batch_size, len(self.calibration_data["input_ids"]))
                batch_inputs = {
                    "input_ids": self.calibration_data["input_ids"][i:end_idx],
                    "attention_mask": self.calibration_data["attention_mask"][i:end_idx]
                }
                outputs = self.model(**batch_inputs)
        for hook in hooks:
            hook.remove()
        stats = {}
        for name, activation in activations.items():
            stats[name] = {
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'magnitude': activation.norm().item()
            }
        return stats

    def analyze_layer_sensitivity(self, bits: int = 8, metric: str = "kl_div"):
        print(f"Analyzing layer sensitivity with {bits}-bit quantization...")
        baseline_outputs = self._compute_baseline_detailed()
        activation_stats = self._compute_activation_statistics()
        layers = self._get_layer_modules()
        sensitivity_scores = {}
        from tqdm import tqdm
        for layer_name, layer_module in tqdm(layers.items(), desc="Analyzing layers"):
            model_copy = copy.deepcopy(self.model)
            self._replace_layer_in_model(model_copy, layer_name, 
                                       self._quantize_layer(layer_module, bits))
            quantized_outputs = self._compute_quantized_detailed(model_copy)
            scores = {}
            for baseline_batch, quantized_batch in zip(baseline_outputs, quantized_outputs):
                baseline_logits = baseline_batch["logits"]
                quantized_logits = quantized_batch["logits"]
                if metric == "kl_div" or metric == "all":
                    baseline_probs = torch.softmax(baseline_logits / 1.0, dim=-1)
                    quantized_probs = torch.softmax(quantized_logits / 1.0, dim=-1)
                    kl_score = torch.nn.functional.kl_div(
                        torch.log(quantized_probs + 1e-8),
                        baseline_probs,
                        reduction='batchmean'
                    ).item()
                    scores['kl_div'] = scores.get('kl_div', []) + [kl_score]
                if metric == "cosine" or metric == "all":
                    cos_sim = torch.nn.functional.cosine_similarity(
                        baseline_logits.flatten(),
                        quantized_logits.flatten(),
                        dim=0
                    )
                    scores['cosine'] = scores.get('cosine', []) + [1 - cos_sim.item()]
                if metric == "mse" or metric == "all":
                    mse = torch.nn.functional.mse_loss(baseline_logits, quantized_logits).item()
                    baseline_mag = baseline_logits.norm().item()
                    normalized_mse = mse / (baseline_mag ** 2 + 1e-8)
                    scores['mse'] = scores.get('mse', []) + [normalized_mse]
            if metric == "all":
                final_score = 0
                weights = {'kl_div': 0.5, 'cosine': 0.3, 'mse': 0.2}
                for metric_name, weight in weights.items():
                    if metric_name in scores:
                        final_score += weight * np.mean(scores[metric_name])
                sensitivity_scores[layer_name] = final_score
            else:
                sensitivity_scores[layer_name] = np.mean(scores[metric])
            if layer_name in activation_stats:
                act_weight = 1.0 + 0.1 * activation_stats[layer_name]['magnitude'] / 1000
                sensitivity_scores[layer_name] *= act_weight
            del model_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.sensitivity_scores = sensitivity_scores
        return sensitivity_scores

    def _compute_baseline_detailed(self):
        baseline_outputs = []
        with torch.no_grad():
            for i in range(0, len(self.calibration_data["input_ids"]), self.batch_size):
                end_idx = min(i + self.batch_size, len(self.calibration_data["input_ids"]))
                batch_inputs = {
                    "input_ids": self.calibration_data["input_ids"][i:end_idx],
                    "attention_mask": self.calibration_data["attention_mask"][i:end_idx]
                }
                outputs = self.model(**batch_inputs)
                baseline_outputs.append({
                    "logits": outputs.logits.cpu(),
                    "last_hidden_state": outputs.last_hidden_state.cpu() if hasattr(outputs, 'last_hidden_state') else None
                })
        return baseline_outputs

    def _compute_quantized_detailed(self, quantized_model):
        quantized_outputs = []
        with torch.no_grad():
            for i in range(0, len(self.calibration_data["input_ids"]), self.batch_size):
                end_idx = min(i + self.batch_size, len(self.calibration_data["input_ids"]))
                batch_inputs = {
                    "input_ids": self.calibration_data["input_ids"][i:end_idx],
                    "attention_mask": self.calibration_data["attention_mask"][i:end_idx]
                }
                outputs = quantized_model(**batch_inputs)
                quantized_outputs.append({
                    "logits": outputs.logits.cpu(),
                    "last_hidden_state": outputs.last_hidden_state.cpu() if hasattr(outputs, 'last_hidden_state') else None
                })
        return quantized_outputs

    def _replace_layer_in_model(self, model, layer_name, new_layer):
        parts = layer_name.split('.')
        current = model
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], new_layer)

    def get_mixed_precision_config(self, 
                             target_bits: float = 6.0,
                             bit_options: list = None):
        if bit_options is None:
            bit_options = [16, 8, 4]
        else:
            # Ensure we have at least the minimum bit
            bit_options = sorted(list(set(bit_options)), reverse=True)
        
        # Sort layers by sensitivity (most sensitive first)
        sorted_layers = sorted(self.sensitivity_scores.items(), 
                            key=lambda x: x[1], reverse=True)
        n_layers = len(sorted_layers)
        
        # Initialize all layers with minimum bit
        min_bit = min(bit_options)
        config = {layer: min_bit for layer, _ in sorted_layers}
        current_total = min_bit * n_layers
        target_total = target_bits * n_layers
        
        # Sort bit options descending for priority checking
        sorted_bits = sorted(bit_options, reverse=True)
        
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
        print("Applying mixed precision quantization...")
        quantized_model = copy.deepcopy(self.model)
        for layer_name, bits in config.items():
            parts = layer_name.split('.')
            current = self.model
            for part in parts:
                current = getattr(current, part)
            quantized_layer = self._quantize_layer(current, bits)
            self._replace_layer_in_model(quantized_model, layer_name, quantized_layer)
        return quantized_model

    def evaluate_model(self, model, num_samples: int = 30):
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        try:
            with torch.no_grad():
                # Get model's maximum sequence length
                max_length = model.config.max_position_embeddings
                stride = 512  # Sliding window stride
                
                # Process each sample in the evaluation data
                for i in range(min(num_samples, len(self.eval_data["input_ids"]))):
                    input_ids = self.eval_data["input_ids"][i:i+1]  # Keep batch dimension
                    attention_mask = self.eval_data["attention_mask"][i:i+1]
                    seq_len = input_ids.size(1)
                    
                    prev_end_loc = 0
                    for begin_loc in range(0, seq_len, stride):
                        end_loc = min(begin_loc + max_length, seq_len)
                        trg_len = end_loc - prev_end_loc
                        
                        # Prepare input and target
                        batch_inputs = {
                            "input_ids": input_ids[:, begin_loc:end_loc],
                            "attention_mask": attention_mask[:, begin_loc:end_loc]
                        }
                        target_ids = batch_inputs["input_ids"].clone()
                        target_ids[:, :-trg_len] = -100  # Mask out tokens not used for loss
                        
                        # Forward pass
                        outputs = model(**batch_inputs, labels=target_ids)
                        
                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            loss = outputs.loss
                            if torch.isfinite(loss):
                                # Calculate number of valid tokens for loss
                                num_valid_tokens = (target_ids != -100).sum().item()
                                batch_size = target_ids.size(0)
                                num_loss_tokens = num_valid_tokens - batch_size  # Account for label shift
                                
                                total_loss += loss.item() * num_loss_tokens
                                total_tokens += num_loss_tokens
                        
                        prev_end_loc = end_loc
                        if end_loc == seq_len:
                            break
                            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return float('inf')
            
        if total_tokens == 0:
            return float('inf')
            
        avg_loss = total_loss / total_tokens
        avg_loss = min(avg_loss, 50)  # Cap the loss to prevent numerical issues
        perplexity = math.exp(avg_loss)
        return perplexity

    def run_full_analysis(self, target_avg_bits: float = 6.0):
        print("=" * 70)
        print("IMPROVED LAYER-WISE SENSITIVITY ANALYSIS")
        print("=" * 70)
        sensitivity_scores = self.analyze_layer_sensitivity(bits=8, metric="kl_div")
        print(f"\nLayer Sensitivity Scores (KL Divergence, higher = more sensitive):")
        print("-" * 70)
        sorted_scores = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        for layer_name, score in sorted_scores:
            print(f"{layer_name:<45} {score:.8f}")
        mp_config = self.get_mixed_precision_config(target_bits=target_avg_bits)
        print(f"\nMixed Precision Configuration (target: {target_avg_bits:.1f} bits avg):")
        print("-" * 70)
        bit_counts = {}
        for layer_name, bits in mp_config.items():
            bit_counts[bits] = bit_counts.get(bits, 0) + 1
            sensitivity = sensitivity_scores[layer_name]
            print(f"{layer_name:<45} {bits:2d} bits (sensitivity: {sensitivity:.8f})")
        print(f"\nBit Distribution:")
        for bits in sorted(bit_counts.keys(), reverse=True):
            print(f"  {bits:2d} bits: {bit_counts[bits]:2d} layers")
        actual_avg_bits = sum(mp_config.values()) / len(mp_config)
        print(f"  Actual average: {actual_avg_bits:.2f} bits")
        quantized_model = self.apply_mixed_precision(mp_config)
        print(f"\nModel Evaluation:")
        print("-" * 70)
        try:
            original_ppl = self.evaluate_model(self.model)
            original_size = get_model_size_mb(self.model)  # Original model is 32-bit
            print(f"Original Model Size:            {original_size:.2f} MB")
            print(f"Original Model Perplexity:      {original_ppl:.4f}")
        except Exception as e:
            print(f"Error computing original model metrics: {e}")
            original_ppl = float('inf')
            original_size = 0
            print(f"Original Model Size:            Failed to compute")
            print(f"Original Model Perplexity:      Failed to compute")
        try:
            quantized_ppl = self.evaluate_model(quantized_model)
            quantized_size = get_model_size_mb(quantized_model, mp_config)  # Pass bit configuration
            print(f"Quantized Model Size:           {quantized_size:.2f} MB")
            print(f"Mixed Precision Perplexity:     {quantized_ppl:.4f}")
            if original_ppl != float('inf') and quantized_ppl != float('inf'):
                degradation = quantized_ppl - original_ppl
                rel_degradation = (quantized_ppl / original_ppl - 1) * 100
                size_reduction = ((original_size - quantized_size) / original_size) * 100
                print(f"Perplexity Degradation:         {degradation:.4f}")
                print(f"Relative Degradation:           {rel_degradation:.2f}%")
                print(f"Size Reduction:                 {size_reduction:.2f}%")
            else:
                print(f"Degradation analysis:           Not available")
        except Exception as e:
            print(f"Error computing quantized model metrics: {e}")
            quantized_ppl = float('inf')
            quantized_size = 0
            print(f"Quantized Model Size:           Failed to compute")
            print(f"Mixed Precision Perplexity:     Failed to compute")
        results = {
            "sensitivity_scores": sensitivity_scores,
            "mixed_precision_config": mp_config,
            "original_perplexity": original_ppl if original_ppl != float('inf') else None,
            "quantized_perplexity": quantized_ppl if quantized_ppl != float('inf') else None,
            "original_model_size_mb": original_size,
            "quantized_model_size_mb": quantized_size,
            "target_avg_bits": target_avg_bits,
            "actual_avg_bits": actual_avg_bits,
            "bit_distribution": bit_counts
        }
        return results 