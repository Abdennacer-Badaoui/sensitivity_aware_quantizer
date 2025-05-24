# Sensitivity Aware Quantizer (WIP)

A tool for analyzing and quantizing transformer models using sensitivity-aware mixed precision quantization.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sensitivity_aware_quantizer

# Install dependencies
pip install -r requirements.txt
```

## Usage

The tool provides various command-line options to customize the quantization process. Here are some common usage examples:

### Basic Usage

Run with default settings (uses EleutherAI/gpt-neo-125M model, 12.0 target bits, and JSD metric):

```bash
python quantizer.py
```

### Multiple Models Analysis

Analyze multiple models with custom target bits and metric:

```bash
python quantizer.py \
    --models EleutherAI/gpt-neo-125M gpt2 \
    --target_bits 8.0 \
    --metric jsd
```

### Custom Dataset Configuration

Use a different dataset and adjust sample sizes:

```bash
python quantizer.py \
    --dataset wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --calibration_samples 200 \
    --eval_samples 200
```

### Custom Output Directories

Specify custom directories for results and plots:

```bash
python quantizer.py \
    --results_dir custom_results \
    --plots_dir custom_plots
```

### Full Configuration Example

Complete example with all major parameters customized:

```bash
python quantizer.py \
    --models EleutherAI/gpt-neo-125M \
    --target_bits 8.0 \
    --metric all \
    --batch_size 32 \
    --device cuda \
    --calibration_samples 150 \
    --eval_samples 150
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--models` | List of model names to analyze | `["EleutherAI/gpt-neo-125M"]` |
| `--dataset` | Dataset name for calibration and evaluation | `"wikitext"` |
| `--dataset_config` | Dataset configuration | `"wikitext-2-raw-v1"` |
| `--calibration_samples` | Number of samples for calibration | `100` |
| `--eval_samples` | Number of samples for evaluation | `100` |
| `--target_bits` | Target average bits for mixed precision | `12.0` |
| `--metric` | Sensitivity analysis metric (`jsd`, `cosine`, `mse`, or `all`) | `"jsd"` |
| `--batch_size` | Batch size for processing | `16` |
| `--device` | Device to use (`auto`, `cuda`, or `cpu`) | `"auto"` |
| `--results_dir` | Directory to save results | `"results"` |
| `--plots_dir` | Directory to save plots | `"plots"` |

## Output

The tool generates:
- Quantization analysis results in JSON format (saved in `results_dir`)
- Comparison plots showing model sizes and perplexity (saved in `plots_dir`)
- Detailed sensitivity scores for each layer
- Mixed precision configuration based on layer sensitivity