# Sensitivity Aware Mixed Precision Quantization V1 

## 🗝️ TL;DR

**Problem:** Uniform ultra-low-precision quantization causes severe accuracy loss, demanding per-layer approaches.  
**Solution:** Smart mixed-precision quantization adapts bit-widths layer by layer—protecting sensitive parts while aggressively compressing the rest.  

**How It Works:**  
- 🔍 **Sensitivity Scoring** – Measures how much each layer suffers from quantization  
- ⚖️ **Precision Allocation** – Automatically assigns higher bits to fragile layers, lower bits to robust ones  
- 🔄 **Refinement** – Iteratively adjusts to meet accuracy targets  

**Why It Matters:** Delivers leaner, faster models without sacrificing critical performance.  
*(Dive into the methodology for the nitty-gritty! 🧠)*  

## Table of Contents:

- 📖 [Introduction](#introduction)
- 📚 [Background and Motivation](#background-and-motivation)
- 🧠 [Methodology](#methodology)
  - 🔢 [Quantization Options](#quantization-options)
  - 🧪 [Layer Sensitivity Estimation](#layer-sensitivity-estimation)
  - ⚖️ [Layer-Wise Quantization Bit Assignment](#layer-wise-quantization-bit-assignment)
- 📊 [Results](#results)
- ✅ [Conclusion](#conclusion)
- 🧭 [Next Steps](#next-steps) 

## Introduction

Deploying large neural networks like Transformers and diffusion models efficiently remains a major challenge due to their high memory and compute requirements. Quantization helps by reducing precision, but applying it uniformly across all layers often hurts performance.

In this work, we explore a sensitivity-aware mixed-precision quantization (MPQ) approach that allocates precision levels based on how sensitive each block is to quantization. More fragile blocks are kept at higher precision, while more robust ones are quantized more aggressively. This method adapts to different architectures — from LLMs to diffusion models — and leads to better trade-offs between efficiency and accuracy compared to uniform quantization.

## Background and Motivation

Recent advancements in mixed-precision quantization have highlighted the importance of assigning different bit-widths to various components of neural networks based on their sensitivity to quantization. For instance, [**Qua²SeDiMo**](https://arxiv.org/pdf/2412.14628) introduces a graph neural network (GNN) to predict sensitivities in diffusion models, enabling effective mixed-precision quantization. However, this approach requires training a GNN for each model, which can be computationally intensive and limits its practicality for rapid deployment across diverse architectures.

Similarly, [**SensiMix**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265621) applies sensitivity-aware mixed-precision quantization to BERT by combining 8-bit index quantization with 1-bit value quantization. While effective, this method is tailored specifically to BERT, making it less adaptable to other architectures. Other approaches, such as those utilizing cluster-based tree-structured Parzen estimators, offer automated bit-width selection but often rely on model-specific characteristics or require substantial computational resources. 

These methods underscore a common challenge: the lack of a fast, model-agnostic technique for sensitivity analysis that can be applied across various neural network architectures without extensive training or fine-tuning.

Our goal is to develop a **fast**,  **model-agnostic** sensitivity analysis method that enables efficient mixed-precision quantization across diverse neural network architectures. By swiftly identifying sensitive components within a network, we aim to allocate higher precision where necessary and lower precision elsewhere, achieving an optimal balance between model performance and computational efficiency.


## Methodology 

This sensitivity-aware automatic quantization framework formulates a set of optimization problems whose careful resolution is crucial to achieving the highest level of quantization with minimal degradation in model performance. The core idea is to adapt the quantization strategy to each layer based on its sensitivity to precision loss.

#### Quantization Options

We begin by traversing the model to identify all `nn.Linear` layers (except `lm_head`). After computing a sensitivity score for each layer (or block), we assign an appropriate quantization strategy. The available options include:

- **Unchanged (`nn.Linear`)**: Keeps the original FP32 weights.
- **W16A16LinearLayer**: Weight quantization to BF16 (16-bit), activation remains FP16.
- **W8A16LinearLayer**: Weight quantization to INT8, activation remains FP16.
- **W4A16LinearLayer**: Weight quantization to INT4 (4-bit), activation remains FP16.

Each selected quantization scheme replaces the standard `nn.Linear` layer with a corresponding custom layer class that supports the chosen precision level. These custom classes handle the quantization and dequantization processes, including weight packing (for W4A16), scale and zero-point computation, and buffer management.

<div style="display: flex; justify-content: center; gap: 40px; text-align: center; font-family: Arial, sans-serif;">

  <div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/MoiIpvN4WyzBJ6T3mwjPE.png" alt="W8A16 Quantization" width="300" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    <div align="center" style="margin-top: 10px; font-weight: 600;">Original Model</div>
  </div>

  <div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/GESgY8DCx-hju-qLh3EB6.png" alt="W4A16 Quantization" width="300" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    <div align="center"  style="margin-top: 10px; font-weight: 600;">MPQ version of the model</div>
  </div>

</div>

<p align="center" style="font-family: Arial, sans-serif; font-style: italic; font-size: 13px; margin-top: 25px;">
  Figure 1: EleutherAI/gpt-neo-125M model overview before and after MPQ 
</p>


#### Layer Sensitivity Estimation

As presented in the background section, there are multiple methods for estimating the sensitivity of a layer or block to quantization. However, most of these methods are either computationally expensive or model-specific.
In this work, our goal is to develop methods that are both effective and computationally efficient for estimating such 
sensitivities. A simple yet promising idea is divergence-based sensitivity estimation. The core idea is straightforward: to estimate the sensitivity of a given layer or block, we compute the Jensen-Shannon Divergence (JSD) between the outputs of the original model and those of a model where only that specific layer or block is quantized. A high divergence indicates a highly sensitive layer, which corresponds to a high sensitivity score—this layer should then be allocated more bits during the mixed-precision quantization phase.
We also incorporate a small boost to the sensitivity scores based on the activation magnitude, to further refine the estimation.

The divergence-based sensitivity score is computed using the Jensen-Shannon Divergence (JSD) between the baseline and quantized output probability distributions. The sensitivity score averaged over all batches \\( B\\) and normalized by \\( \log 2 \\) is:

$$
\text{Sensitivity}_{\text{div}} = \frac{1}{B} \sum_{b=1}^{B} \frac{\mathrm{JSD}(P_b \| Q_b)}{\log 2}
$$

The final sensitivity score is adjusted using the normalized activation magnitude:

$$
\text{Final Score}_\ell = \text{Sensitivity}_\ell \cdot \left(1 +  \lambda \cdot \frac{\text{magnitude}_\ell}{\max\limits_{k} \text{magnitude}_k} \right)
$$

The variable \\( \lambda \\) is a hyperparameter that can be tuned on the validation dataset.

The dataset used to calculate these scores is a subset of the `wikitext` dataset. The number of samples used also needs to be studied in order to determine the minimum size that provides a representative subset and yields reliable sensitivity estimates.

Here is the layer sensitivity analysis for three different models:

<div style="display: flex; justify-content: center; gap: 40px; text-align: center; font-family: Arial, sans-serif;">

  <div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/O93gAgEq6N0C5UDdfHv20.png" width="300" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    <div align="center" style="margin-top: 10px; font-weight: 600;"></div>
  </div>

  <div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/uWumpG2OrAGFzC7bapcYt.png" width="375" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    <div align="center"  style="margin-top: 10px; font-weight: 600;"></div>
  </div>

</div>

<p align="center" style="font-family: Arial, sans-serif; font-style: italic; font-size: 13px; margin-top: 25px;">
  Figure 2: Layer Sensitivities of EleutherAI/gpt-neo-125M
</p>


<div style="display: flex; justify-content: center; gap: 40px; text-align: center; font-family: Arial, sans-serif;">

  <div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/rDpK2IUsfH3XmhjKK7x25.png" width="300" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    <div align="center" style="margin-top: 10px; font-weight: 600;"></div>
  </div>

  <div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/zw_WrS8sVNvsxmgcmEuld.png" width="375" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    <div align="center"  style="margin-top: 10px; font-weight: 600;"></div>
  </div>

</div>

<p align="center" style="font-family: Arial, sans-serif; font-style: italic; font-size: 13px; margin-top: 25px;">
  Figure 3: Layer Sensitivities of facebook/opt-125m
</p>


<div style="display: flex; justify-content: center; gap: 40px; text-align: center; font-family: Arial, sans-serif;">

  <div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/bzwYMwBPfkXarqCPe8SSA.png" width="300" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    <div align="center" style="margin-top: 10px; font-weight: 600;"></div>
  </div>

  <div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/SqgJQh7CpA3o_85EgYQyM.png" width="375" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    <div align="center"  style="margin-top: 10px; font-weight: 600;"></div>
  </div>

</div>

<p align="center" style="font-family: Arial, sans-serif; font-style: italic; font-size: 13px; margin-top: 25px;">
  Figure 4: Layer Sensitivities of TinyLlama/TinyLlama-1.1B-Chat-v1.0
</p>

Looking at the sensitivity analysis results across three different language models (EleutherAI GPT-Neo-125M, Facebook OPT-125M, and TinyLlama 1.1B), several consistent patterns emerge that provide valuable insights into quantization behavior. The most striking observation is the consistently high sensitivity of the final projection layers (c_proj) across all models, with sensitivity scores ranging from 1.0e-04 to 1.4e-04 in GPT-Neo and OPT models, and reaching 3.3e-05 in TinyLlama's down_proj layers. This pattern reflects these layers' critical role in projecting high-dimensional representations back to the vocabulary space, where even small quantization errors can significantly impact final output probabilities. Additionally, value projection layers (v_proj) demonstrate notable sensitivity, particularly in GPT-Neo (1.0e-04) and TinyLlama (1.1e-05), suggesting that attention value computations are particularly susceptible to precision loss due to their role in determining information flow within the attention mechanism.

The analysis reveals significant architectural differences in sensitivity patterns between models. Facebook OPT exhibits a distinctive pattern with its fully connected layers (fc1 and fc2) showing relatively high sensitivity scores (3.87e-04 and 5.76e-04), which differs markedly from the other models and likely reflects architectural differences in feed-forward network implementation. Conversely, key and query projection layers (k_proj and q_proj) consistently demonstrate lower sensitivity across all models, with scores around 2.7e-05 in GPT-Neo and even lower values of 5.1e-06 and 9.5e-06 in OPT. This robustness to quantization suggests that attention mechanisms can tolerate some noise in key-query similarity computations without substantial performance degradation.

The layer-wise sensitivity distribution reveals non-uniform patterns across model depth, with clear peaks occurring at specific layers rather than gradual transitions. Both GPT-Neo and OPT models show highest sensitivities in early-to-middle layers and at the model's end, while TinyLlama exhibits a more distributed pattern reflecting its different architecture and training methodology. These findings provide clear guidance for mixed-precision quantization strategies: projection layers, particularly c_proj/down_proj components, should receive higher bit allocations to maintain performance, while k_proj and q_proj layers can tolerate more aggressive quantization. The model-specific patterns underscore the importance of per-architecture sensitivity analysis, as optimal quantization strategies may vary significantly between different model families even when they share similar parameter counts and overall structures.


##### 🧭 Next Steps:

Next, we will implement more robust methods for estimating layer or block sensitivities. In particular, we will explore Hessian-based approaches, as introduced in the HAWQ paper. These methods leverage second-order information to more accurately assess how sensitive a layer is to quantization by examining the curvature of the loss landscape with respect to the weights.

While computationally more intensive than divergence-based methods, Hessian-aware techniques offer higher fidelity in estimating sensitivity and can better guide precision allocation in mixed-precision quantization. We plan to investigate approximations of the Hessian (e.g., block-diagonal or diagonal) to reduce the computational overhead and make these methods more scalable.

Additionally, we will compare the effectiveness and efficiency of divergence-based and Hessian-based sensitivity estimators across different model architectures and tasks, in order to assess trade-offs and guide method selection in practice.


#### Layer-Wise Quantization Bit Assignment

Once layer-wise sensitivity scores have been computed, the next challenge is to determine the appropriate quantization configuration based on these scores. This presents a non-trivial optimization problem that depends on various factors, including model architecture, accuracy constraints, and memory or latency budgets.

In the absence of sensitivity estimates, one would need to perform an exhaustive search over all possible bitwidth assignments. However, this becomes computationally intractable for deep neural networks, as the search space for mixed-precision quantization grows exponentially with the number of layers. To address this, we propose a heuristic initialization strategy that leverages sensitivity scores to guide the initial bit allocation, followed by iterative refinement to satisfy specific performance constraints.

We consider the following three initialization strategies:

- `adaptive_threshold`: Derives bit allocations from sensitivity statistics using predefined thresholds.
- `int8_only`: Assigns INT8 precision uniformly to all layers.
- `int4_only`: Assigns INT4 precision to all layers (typically suboptimal unless followed by refinement).

In the `adaptive_threshold` method, we define three threshold values to segment the sensitivity distribution into four quantization intervals: FP32, BF16, INT8, and INT4. These thresholds are calculated as follows:

$$
\alpha_1 = \mu + 0.5\sigma, \quad \alpha_2 = \mu, \quad \alpha_3 = \mu - 0.5\sigma
$$

where \\( \mu \\) and \\( \sigma \\) denote the mean and standard deviation of the sensitivity scores across all layers.

These strategies act as initialization points in the search for an optimal mixed-precision configuration—one that maximizes model compression while minimizing accuracy degradation, typically measured in terms of perplexity (smaller is better). Compared to brute-force search, this approach is significantly more scalable and practical.

Following initialization, we iteratively refine the bit allocation by monitoring the perplexity gap between the quantized and original models. If the gap exceeds a predefined tolerance (e.g., `max_perplexity_increase`), we systematically upgrade layers level by level: first upgrading all INT4 layers to INT8, then all INT8 layers to BF16, and finally all BF16 layers to FP32. This level-by-level approach is motivated by the principle of diminishing returns in quantization—upgrading from INT4 to INT8 typically provides greater accuracy improvements than upgrading from INT8 to BF16 or BF16 to FP32, making it more efficient to exhaust lower-precision improvements before moving to higher-precision upgrades. Within each bit level, we upgrade the most sensitive layers in batches (defined by `layers_per_iteration`) until either all layers at that level are upgraded, the perplexity constraint is satisfied, or the maximum iteration limit is reached. This ensures complete exploration of each quantization level before moving to higher precision levels.

##### 🧭 Next Steps:

Moving forward, we plan to explore alternative bit allocation strategies that go beyond heuristic thresholding. In particular, gradient-based optimization methods will be investigated to dynamically adjust bitwidth assignments by directly minimizing a loss function that balances model accuracy and compression. Such approaches could leverage sensitivity scores as initialization but refine bit allocations through backpropagation or reinforcement learning techniques.


## Results 

Using 100 samples for both sensitivity estimation and model evaluation (perplexity), and 50 iterations for the iterative refinement with 3 layers updated per iteration and with a 5% permitted perplexity increase, we obtain the following results:

<p align="center"><em>Comparison of Strategies with and without refinement</em></p>

<p align="center">

<table>
  <thead>
    <tr>
      <th style="text-align:center">Strategy</th>
      <th style="text-align:center">w/o refinement</th>
      <th style="text-align:center">with refinement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">`adaptive_threshold`</td>
      <td align="center"><img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/czBk5YFVZ0jhbKKLYdB09.png" width="300"/></td>
      <td align="center"><img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/XpJ5egRDMX6V7b1v157fY.png" width="300"/></td>
    </tr>
    <tr>
      <td align="center">`int8_only`</td>
      <td align="center"><img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/KTHRIlOBT9eQ-Q8_8mTKt.png" width="300"/></td>
      <td align="center"><img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/cqSYoGJLRGqiQM0GKYyvU.png" width="300"/></td>
    </tr>
    <tr>
      <td align="center">`int4_only`</td>
      <td align="center"><img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/l13uZrXnDvqgJDcW7CzQQ.png" width="300"/></td>
      <td align="center"><img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/CMWpFbj5vNvrdvIiPl6-7.png" width="300"/></td>
    </tr>
  </tbody>
</table>

</p>

The `adaptive_threshold` strategy demonstrates a **conservative approach** that prioritizes model quality preservation. This method maintains model performance with minimal perplexity degradation (0.1-2.3% across tested models) while achieving size reductions of approximately 32-37% for smaller models and 62.7% for TinyLlama. This conservative behavior stems from the strategy's reliance on statistical thresholds derived from sensitivity distributions, which assign higher precision to layers that fall above the mean sensitivity score. The absence of differences between refined and unrefined versions indicates that the initial bit allocation already satisfies the perplexity constraint, demonstrating the `adaptive_threshold`'s built-in safety margins for quality preservation.

The mixed-precision approach reveals important insights about layer-wise sensitivity patterns. While uniform INT8 quantization shows competitive results in terms of compression ratios, the **mixed-precision framework provides crucial flexibility** for handling diverse model architectures and varying sensitivity requirements. The `adaptive_threshold` strategy's ability to automatically allocate precision based on empirical sensitivity measurements represents a significant advancement over fixed quantization schemes, particularly for models where layer importance varies substantially. This approach enables targeted precision allocation that can be fine-tuned based on specific performance requirements and computational constraints.

The INT4-only strategy provides compelling evidence of the **iterative refinement mechanism's recovery capabilities**. Initial aggressive quantization to 4-bit precision resulted in performance degradation, with perplexity increases ranging from 17.4% to 416.8% across tested models. The refinement process achieved substantial recovery, transforming the EleutherAI GPT-Neo model from a 416.8% perplexity increase to 3.6% while maintaining significant compression (39.8% model size reduction). This recovery demonstrates the refinement algorithm's ability to systematically identify and upgrade critical layers, validating both the sensitivity estimation method's layer importance ranking and the iterative upgrade strategy's effectiveness in meeting performance constraints while preserving compression benefits. We can choose a stricter constraint on the allowed perplexity increase (e.g., 1%) if we are more interested in performance. Here are the obtained results (`int4_only` with refinement):

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/65baa31607366d903890bcf4/IBCZ4BNcGGbpwu-YuBywo.png" width="600"/>
    <em>Figure 5: Comparative results (Strategy: int4_only with refinement, Max permitted PPL increase: 1%) </em>
</p>

This offers a good trade-off between compression and performance, as it achieves greater size reduction than the `int8_only` model while maintaining almost no performance degradation.

## Conclusion

In this work, we proposed a sensitivity-aware mixed-precision quantization method that dynamically assigns precision levels to different blocks of a Transformer-based language model. Our approach improves the trade-off between compression and performance by leveraging internal sensitivity signals, achieving competitive results without retraining. Experiments conducted on decoder-only autoregressive models demonstrate the effectiveness of our method, with promising gains in accuracy under tight bit constraints.

## Next Steps
So far, we have evaluated our quantization strategy on decoder-only models for causal language modeling. As future work, we plan to extend our method to other architectural families and tasks, to assess its generalizability. Additionally, we aim to validate our sensitivity metric by comparing it with alternative measures and conduct a more comprehensive evaluation by benchmarking the quantized models on standard downstream datasets such as MMLU and HellaSwag. This will help confirm the robustness of our approach and its applicability to real-world tasks.








