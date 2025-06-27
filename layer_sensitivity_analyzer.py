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
        benchmark: bool = False,
        device: str = "auto",
    ):
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
        self.benchmark = benchmark

    def _prepare_hf_dataset(
        self,
        split: str,
        num_samples: int,
        dataset_name: str,
        dataset_config: str,
        split_name: str,
    ):
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
        layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                layers[name] = module
        return layers

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
                end_idx = min(
                    i + self.batch_size, len(self.calibration_data["input_ids"])
                )
                batch_inputs = {
                    "input_ids": self.calibration_data["input_ids"][i:end_idx],
                    "attention_mask": self.calibration_data["attention_mask"][i:end_idx],
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

    @staticmethod
    def _sensitivity_worker(
        gpu_id: int,
        model_name: str,
        calibration_data: Dict,
        layer_names: List[str],
        batch_size: int,
        method: str,
        output_queue: mp.Queue,
    ):
        try:
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Worker on GPU {gpu_id} started with {len(layer_names)} layers")
            
            # Load model on this GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name
            ).to(device)
            model.eval()
            
            # Move calibration data to this GPU
            calib_data = {
                "input_ids": calibration_data["input_ids"].to(device),
                "attention_mask": calibration_data["attention_mask"].to(device),
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
                    baseline_outputs.append({
                        "logits": outputs.logits.cpu(),
                        "last_hidden_state": outputs.last_hidden_state.cpu()
                        if hasattr(outputs, "last_hidden_state")
                        else None,
                    })

            # Process assigned layers
            layer_scores = {}
            from tqdm import tqdm
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
                        quantized_outputs.append({
                            "logits": outputs.logits.cpu(),
                            "last_hidden_state": outputs.last_hidden_state.cpu()
                            if hasattr(outputs, "last_hidden_state")
                            else None,
                        })
                
                # Compute sensitivity
                score = SensitivityMetrics.compute_divergence_based_sensitivities(
                    baseline_outputs, quantized_outputs
                )
                layer_scores[layer_name] = score
                #print(f"GPU {gpu_id} processed {layer_name}: {score:.4f}")
                
                del model_copy
                torch.cuda.empty_cache()

            output_queue.put(layer_scores)
            print(f"Worker on GPU {gpu_id} completed successfully")
            
        except Exception as e:
            print(f"Error in worker on GPU {gpu_id}: {str(e)}")
            output_queue.put({})

    def analyze_layer_sensitivity(self, bits: int = 8):
        if bits != 8:
            raise ValueError("Parallel processing only supports 8-bit quantization")
        
        print(f"Parallel layer sensitivity analysis with {bits}-bit quantization")
        baseline_outputs = self._compute_baseline()
        activation_stats = self._compute_activation_statistics()
        layers = list(self._get_layer_modules().keys())
        
        # Get available GPUs (exclude display GPU)
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            if total_mem >= 10 * 1024**3:  # At least 10GB memory
                available_gpus.append(i)
        
        if not available_gpus:
            print("No suitable GPUs found, falling back to sequential processing")
            return self._analyze_layer_sensitivity_sequential(bits)
        
        print(f"Using GPUs: {available_gpus}")
        num_gpus = len(available_gpus)
        layers_per_gpu = (len(layers) + num_gpus - 1) // num_gpus
        layer_groups = [layers[i:i+layers_per_gpu] 
                       for i in range(0, len(layers), layers_per_gpu)]
        
        # Prepare data for workers
        calibration_data_cpu = {
            "input_ids": self.calibration_data["input_ids"].cpu(),
            "attention_mask": self.calibration_data["attention_mask"].cpu(),
        }
        
        # Setup multiprocessing
        mp.set_start_method('spawn', force=True)
        output_queue = mp.Queue()
        processes = []
        
        # Start worker processes
        for i, gpu_id in enumerate(available_gpus):
            if i < len(layer_groups):
                p = mp.Process(
                    target=self._sensitivity_worker,
                    args=(
                        gpu_id,
                        self.model_name,
                        calibration_data_cpu,
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

    def _analyze_layer_sensitivity_sequential(self, bits: int = 8):
        print(f"Sequential layer sensitivity analysis with {bits}-bit quantization")
        baseline_outputs = self._compute_baseline()
        activation_stats = self._compute_activation_statistics()
        layers = self._get_layer_modules()
        sensitivity_scores = {}

        input_data = {
            "input_ids": self.calibration_data["input_ids"][:1],
            "attention_mask": self.calibration_data["attention_mask"][:1]
        }

        from tqdm import tqdm
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
                sensitivity_scores[layer_name] = SensitivityMetrics.compute_hessian_based_sensitivities(
                    self.model, self.tokenizer, layer_module.weight, input_data,
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        self.sensitivity_scores = sensitivity_scores
        return sensitivity_scores

    def cached_sensitivity_scores(self, results_dir: str):
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
                    "attention_mask": self.calibration_data["attention_mask"][i:end_idx],
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
        return baseline_outputs

    def _compute_quantized(self, quantized_model):
        quantized_outputs = []
        with torch.no_grad():
            for i in range(0, len(self.calibration_data["input_ids"]), self.batch_size):
                end_idx = min(
                    i + self.batch_size, len(self.calibration_data["input_ids"])
                )
                batch_inputs = {
                    "input_ids": self.calibration_data["input_ids"][i:end_idx],
                    "attention_mask": self.calibration_data["attention_mask"][i:end_idx],
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

            for layer_name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= high_threshold:
                    config[layer_name] = 32
                elif sensitivity >= medium_threshold:
                    config[layer_name] = 16
                elif sensitivity >= low_threshold:
                    config[layer_name] = 8
                else:
                    config[layer_name] = 4

        elif self.config_strategy == "percentile":
            p75 = np.percentile(sensitivities, 75)
            p50 = np.percentile(sensitivities, 50)
            p25 = np.percentile(sensitivities, 25)

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
            norm_sensitivities = {}
            sens_range = max_sens - min_sens
            if sens_range == 0:
                sens_range = 1

            for layer_name, sensitivity in self.sensitivity_scores.items():
                normalized = (sensitivity - min_sens) / sens_range
                exp_score = np.exp(2 * normalized) / np.exp(2)
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
            low_threshold = mean_sens - 0.25 * std_sens
            medium_threshold = mean_sens + 0.25 * std_sens
            high_threshold = mean_sens + 0.75 * std_sens

            for layer_name, sensitivity in self.sensitivity_scores.items():
                if sensitivity >= high_threshold:
                    config[layer_name] = 32
                elif sensitivity >= medium_threshold:
                    config[layer_name] = 32
                elif sensitivity >= low_threshold:
                    config[layer_name] = 16
                else:
                    config[layer_name] = 8

        elif self.config_strategy == "aggressive":
            high_threshold = mean_sens + 1.0 * std_sens
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
            for layer_name in self.sensitivity_scores.keys():
                config[layer_name] = 8

        elif self.config_strategy == "int4_only":
            for layer_name in self.sensitivity_scores.keys():
                config[layer_name] = 4

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
        print(f"Starting progressive iterative configuration:")
        print(f"  - Max perplexity increase: {self.max_perplexity_increase}")
        print(f"  - Layers per iteration: {self.layers_per_iteration}")
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
        sorted_layers = sorted(
            self.sensitivity_scores.items(), key=lambda x: x[1], reverse=True
        )
        upgrade_levels = [(4, 8), (8, 16), (16, 32)]
        total_iteration = 0

        for from_bits, to_bits in upgrade_levels:
            print(f"\n{'='*50}")
            print(f"UPGRADING LAYERS FROM {from_bits} TO {to_bits} BITS")
            print(f"{'='*50}")

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

            while layers_at_current_level and total_iteration < self.max_iterations:
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

                layers_to_upgrade = layers_at_current_level[: self.layers_per_iteration]
                print(f"  Upgrading {len(layers_to_upgrade)} layers:")
                for layer_name, sensitivity in layers_to_upgrade:
                    current_config[layer_name] = to_bits
                    layers_upgraded_this_level += 1
                    print(f"    - {layer_name} (sensitivity: {sensitivity:.4f})")

                layers_at_current_level = layers_at_current_level[self.layers_per_iteration :]
                del quantized_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"\nCompleted {from_bits}→{to_bits} bit level:")
            print(f"  - Layers upgraded: {layers_upgraded_this_level}")
            print(
                f"  - Remaining layers at {from_bits} bits: {len(layers_at_current_level)}"
            )

            if total_iteration >= self.max_iterations:
                print(f"  - Reached maximum iterations ({self.max_iterations})")
                break

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
        print("\n Applying mixed precision quantization...")
        quantized_model = copy.deepcopy(self.model)
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
        return quantized_model

    def run_full_analysis(self):
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

        if self.benchmark:
            print("\nRunning original model benchmarks...")
            original_benchmark_results = {}
            try:
                results = evaluate_llm_benchmark(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                )
                if results:
                    original_benchmark_results = results
            except Exception as e:
                print(f"Error running MMLU benchmark on original model: {e}")

        start_time = time.time()
        sensitivity_scores = self.cached_sensitivity_scores(results_dir="results")
        if not sensitivity_scores:
            print("No cached sensitivity scores found. Running analysis...")
            sensitivity_scores = self.analyze_layer_sensitivity()
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

        benchmark_results = None
        if self.benchmark:
            print("\nRunning quantized model benchmarks...")
            quantized_benchmark_results = {}
            try:
                results = evaluate_llm_benchmark(
                    model=quantized_model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                )
                if results:
                    quantized_benchmark_results = results
            except Exception as e:
                print(f"Error running MMLU benchmark on quantized model: {e}")

            benchmark_results = {
                "original": original_benchmark_results,
                "quantized": quantized_benchmark_results,
            }

        results = run_full_analysis_report(
            sensitivity_scores,
            mp_config,
            original_ppl,
            quantized_ppl,
            original_size,
            quantized_size,
            benchmark_results,
        )

        results["quantized_model"] = quantized_model

        return results