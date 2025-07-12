import os
import glob
import json
import torch
import matplotlib.pyplot as plt

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))]
    )
else:
    accelerator = None


def evaluate_llm_benchmark(model_name):
    evaluation_tracker = EvaluationTracker(
        output_dir="./benchmark_results",
        save_details=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE, max_samples=10
    )

    model_config = VLLMModelConfig(
        model_name=model_name,
        # dtype="float16",
        use_chat_template=True,
    )

    task = "leaderboard|mmlu|5|1"

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

    del pipeline, model_config, evaluation_tracker, pipeline_params
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Define full precision and quantized model names
    model1 = "Qwen/Qwen3-4B"
    model2 = "quantized_models/Qwen/Qwen3-4B_divergence_100_int4_only_True"

    # Evaluate each model (comment out if already done)
    evaluate_llm_benchmark(model1)
    evaluate_llm_benchmark(model2)
