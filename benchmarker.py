import torch
import numpy as np
from datasets import load_dataset
from evaluate import load
import re
from typing import Dict, List, Optional, Union, Any


def evaluate_llm_benchmark(
    model,
    tokenizer,
    benchmark_name: str = "glue",
    task_name: str = "mrpc",
    num_samples: int = 1000,
    device: str = "cuda",
    few_shot_examples: int = 5,
    max_length: int = 512,
    temperature: float = 0.1,
    do_sample: bool = True,
    batch_size: int = 256, 
) -> Optional[Dict[str, Any]]:
    """
    Evaluate LLM performance on various benchmark datasets using prompt-based evaluation with batch processing.

    Args:
        model: The language model
        tokenizer: The tokenizer
        benchmark_name: Name of benchmark ("glue", "super_glue", "mmlu", etc.)
        task_name: Specific task within benchmark
        num_samples: Number of samples to evaluate
        device: Device to run on
        few_shot_examples: Number of few-shot examples to include in prompt
        max_length: Maximum generation length
        temperature: Generation temperature
        do_sample: Whether to use sampling for generation
        batch_size: Number of examples to process in each batch

    Returns:
        Dictionary containing evaluation results
    """

    try:
        # Load dataset and metric
        if benchmark_name == "glue":
            dataset = load_dataset("glue", task_name, split="validation")
            metric = load("glue", task_name)
        elif benchmark_name == "super_glue":
            dataset = load_dataset("super_glue", task_name, split="validation")
            metric = load("super_glue", task_name)
        elif benchmark_name == "mmlu":
            print(f"Evaluating MMLU task: {task_name}")
            dataset = load_dataset("cais/mmlu", task_name, split="test")
            metric = load("accuracy")
        else:
            print(f"Benchmark {benchmark_name} not supported yet")
            return None

        # Limit number of samples
        if num_samples and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))

        print(f"Evaluating {len(dataset)} samples from {benchmark_name}/{task_name} with batch size {batch_size}")

        # Get task configuration
        task_config = get_task_config(benchmark_name, task_name)
        if not task_config:
            print(f"Task configuration not found for {benchmark_name}/{task_name}")
            return None

        # Get few-shot examples if requested
        few_shot_prompt = ""
        if few_shot_examples > 0:
            few_shot_prompt = create_few_shot_prompt(
                dataset, task_config, few_shot_examples
            )

        model.eval()
        all_predictions = []
        all_labels = []

        # Process in batches
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_examples = dataset.select(range(batch_start, batch_end))
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
            
            # Create prompts for the entire batch
            batch_prompts = []
            batch_labels = []
            
            for example in batch_examples:
                prompt = create_prompt(example, task_config, few_shot_prompt)
                label = get_label(example, task_config)
                
                if label is not None:
                    batch_prompts.append(prompt)
                    batch_labels.append(label)
            
            if not batch_prompts:
                continue
                
            # Generate responses for the batch
            batch_responses = generate_batch_responses(
                model,
                tokenizer,
                batch_prompts,
                device,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
            )
            
            # Extract predictions from batch responses
            for i, response in enumerate(batch_responses):
                if i < len(batch_examples):
                    example = batch_examples[i]
                    prediction = extract_prediction(response, task_config, example)
                    
                    if prediction is not None and i < len(batch_labels):
                        all_predictions.append(prediction)
                        all_labels.append(batch_labels[i])
                    else:
                        print(f"Skipping example in batch: prediction={prediction}")

        print(f"Successfully processed {len(all_predictions)}/{len(dataset)} examples")

        if len(all_predictions) == 0:
            print("No valid predictions generated")
            return None

        # Compute metrics
        if benchmark_name in ["glue", "super_glue"]:
            results = metric.compute(predictions=all_predictions, references=all_labels)
        else:
            results = metric.compute(predictions=all_predictions, references=all_labels)

        return results

    except Exception as e:
        print(f"Error during benchmark evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_batch_responses(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_length: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> List[str]:
    """Generate responses from the model for a batch of prompts."""
    
    if not prompts:
        return []
    
    # Tokenize all prompts in the batch
    # We need to handle padding for batch processing
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length // 2,
        padding=True,  # Enable padding for batch processing
        pad_to_multiple_of=8,  # Optional: pad to multiple of 8 for efficiency
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    responses = []
    
    # Generate responses
    with torch.no_grad():
        if hasattr(model, "generate"):
            # For models with generate method
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode each generated sequence
            for i in range(outputs.shape[0]):
                # Get the generated part (excluding input tokens)
                input_length = inputs["input_ids"][i].shape[0]
                generated_tokens = outputs[i][input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response.strip())
                
        else:
            # For models without generate method, use forward pass
            # This is more complex for batch processing
            outputs = model(**inputs)
            
            for i in range(outputs.logits.shape[0]):
                # Get last non-padded token logits for each sequence
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    # Find the last non-padded position
                    last_pos = attention_mask[i].sum().item() - 1
                    logits = outputs.logits[i, last_pos, :]
                else:
                    logits = outputs.logits[i, -1, :]
                
                predicted_token_id = torch.argmax(logits).item()
                response = tokenizer.decode([predicted_token_id], skip_special_tokens=True)
                responses.append(response.strip())

    return responses


def get_task_config(benchmark_name: str, task_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for specific benchmark task."""

    configs = {
        "glue": {
            "cola": {
                "input_keys": ["sentence"],
                "label_key": "label",
                "labels": ["unacceptable", "acceptable"],
                "task_type": "classification",
                "prompt_template": "Sentence: {sentence}\nIs this sentence grammatically acceptable?\nAnswer:",
            },
            "sst2": {
                "input_keys": ["sentence"],
                "label_key": "label",
                "labels": ["negative", "positive"],
                "task_type": "classification",
                "prompt_template": "Review: {sentence}\nSentiment:",
            },
            "mrpc": {
                "input_keys": ["sentence1", "sentence2"],
                "label_key": "label",
                "labels": ["not equivalent", "equivalent"],
                "task_type": "classification",
                "prompt_template": "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre these sentences equivalent?\nAnswer:",
            },
            "qqp": {
                "input_keys": ["question1", "question2"],
                "label_key": "label",
                "labels": ["not duplicate", "duplicate"],
                "task_type": "classification",
                "prompt_template": "Question 1: {question1}\nQuestion 2: {question2}\nAre these questions asking the same thing?\nAnswer:",
            },
            "mnli": {
                "input_keys": ["premise", "hypothesis"],
                "label_key": "label",
                "labels": ["entailment", "neutral", "contradiction"],
                "task_type": "classification",
                "prompt_template": "Premise: {premise}\nHypothesis: {hypothesis}\nRelationship:",
            },
            "qnli": {
                "input_keys": ["question", "sentence"],
                "label_key": "label",
                "labels": ["not entailment", "entailment"],
                "task_type": "classification",
                "prompt_template": "Question: {question}\nSentence: {sentence}\nDoes the sentence answer the question?\nAnswer:",
            },
            "rte": {
                "input_keys": ["sentence1", "sentence2"],
                "label_key": "label",
                "labels": ["not entailment", "entailment"],
                "task_type": "classification",
                "prompt_template": "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nDoes sentence 1 entail sentence 2?\nAnswer:",
            },
            "wnli": {
                "input_keys": ["sentence1", "sentence2"],
                "label_key": "label",
                "labels": ["not entailment", "entailment"],
                "task_type": "classification",
                "prompt_template": "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nDoes sentence 1 entail sentence 2?\nAnswer:",
            },
            "stsb": {
                "input_keys": ["sentence1", "sentence2"],
                "label_key": "label",
                "labels": None,  # Regression task
                "task_type": "regression",
                "prompt_template": "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nSimilarity score (0-5):",
            },
        },
        "super_glue": {
            "boolq": {
                "input_keys": ["passage", "question"],
                "label_key": "label",
                "labels": ["False", "True"],
                "task_type": "classification",
                "prompt_template": "Passage: {passage}\nQuestion: {question}\nAnswer:",
            },
            "cb": {
                "input_keys": ["premise", "hypothesis"],
                "label_key": "label",
                "labels": ["entailment", "contradiction", "neutral"],
                "task_type": "classification",
                "prompt_template": "Premise: {premise}\nHypothesis: {hypothesis}\nRelationship:",
            },
            "copa": {
                "input_keys": ["premise", "choice1", "choice2", "question"],
                "label_key": "label",
                "labels": ["choice1", "choice2"],
                "task_type": "multiple_choice",
                "prompt_template": "Premise: {premise}\nWhat is the {question}?\nChoice 1: {choice1}\nChoice 2: {choice2}\nAnswer:",
            },
            "wic": {
                "input_keys": ["sentence1", "sentence2", "word"],
                "label_key": "label",
                "labels": ["False", "True"],
                "task_type": "classification",
                "prompt_template": "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nDoes the word '{word}' have the same meaning in both sentences?\nAnswer:",
            },
        },
        "mmlu": {
            # Generic template for all MMLU subjects
            "default": {
                "input_keys": ["question", "choices"],
                "label_key": "answer",
                "labels": ["A", "B", "C", "D"],
                "task_type": "multiple_choice",
                "prompt_template": "Question: {question}\nA. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\nAnswer:",
            }
        },
    }

    if benchmark_name in configs:
        if task_name in configs[benchmark_name]:
            return configs[benchmark_name][task_name]
        elif benchmark_name == "mmlu":
            return configs[benchmark_name]["default"]

    return None


def create_few_shot_prompt(
    dataset, task_config: Dict[str, Any], num_examples: int
) -> str:
    """Create few-shot examples for the prompt."""
    few_shot_examples = []

    # Get a few examples from the dataset 
    examples = dataset.select(range(min(num_examples, len(dataset))))

    for example in examples:
        prompt = create_prompt(example, task_config, "")
        label = get_label(example, task_config)

        if task_config["task_type"] == "classification" and task_config["labels"]:
            answer = task_config["labels"][label]
        elif task_config["task_type"] == "multiple_choice" and task_config["labels"]:
            answer = task_config["labels"][label]
        elif task_config["task_type"] == "regression":
            answer = str(label)
        else:
            answer = str(label)

        few_shot_examples.append(f"{prompt} {answer}")

    return "\n\n".join(few_shot_examples) + "\n\n"


def create_prompt(
    example: Dict[str, Any], task_config: Dict[str, Any], few_shot_prompt: str = ""
) -> str:
    """Create prompt for a single example."""
    prompt_template = task_config["prompt_template"]

    # Handle MMLU special case
    if "choices" in example and isinstance(example["choices"], list):
        prompt_data = {
            "question": example.get("question", ""),
            "choice_a": example["choices"][0] if len(example["choices"]) > 0 else "",
            "choice_b": example["choices"][1] if len(example["choices"]) > 1 else "",
            "choice_c": example["choices"][2] if len(example["choices"]) > 2 else "",
            "choice_d": example["choices"][3] if len(example["choices"]) > 3 else "",
        }
    else:
        # Regular case
        prompt_data = {key: example.get(key, "") for key in task_config["input_keys"]}

    formatted_prompt = prompt_template.format(**prompt_data)

    return few_shot_prompt + formatted_prompt


def extract_prediction(
    response: str, task_config: Dict[str, Any], example: Dict[str, Any]
) -> Optional[Union[int, float]]:
    """Extract prediction from model response."""

    response = response.lower().strip()

    if task_config["task_type"] == "classification":
        labels = task_config["labels"]
        if not labels:
            return None

        # Try to find exact label matches
        for i, label in enumerate(labels):
            if label.lower() in response:
                return i

        # Try common answer patterns
        if (
            "yes" in response
            or "true" in response
            or "equivalent" in response
            or "duplicate" in response
        ):
            return 1
        elif "no" in response or "false" in response or "not" in response:
            return 0

        # Default to first option if no clear answer
        return 0

    elif task_config["task_type"] == "multiple_choice":
        # Look for A, B, C, D in response
        for i, letter in enumerate(["a", "b", "c", "d"]):
            if f" {letter}" in f" {response}" or f"{letter}." in response or response.startswith(letter):
                return i

        # Try to find choice content matches
        if "choices" in example:
            choices = example["choices"]
            for i, choice in enumerate(choices):
                if choice.lower() in response:
                    return i

        # Default to first option if no clear answer
        return 0

    elif task_config["task_type"] == "regression":
        # Extract number from response
        numbers = re.findall(r"\d+\.?\d*", response)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        return None

    return None


def get_label(
    example: Dict[str, Any], task_config: Dict[str, Any]
) -> Optional[Union[int, float]]:
    """Get ground truth label from example."""
    label_key = task_config["label_key"]

    if label_key in example:
        label = example[label_key]

        # Handle different label formats
        if isinstance(label, (int, float)):
            return label
        elif isinstance(label, str):
            # Try to convert string labels to indices
            if task_config["labels"]:
                try:
                    return task_config["labels"].index(label.lower())
                except ValueError:
                    return None
            else:
                # Try to parse as number for regression
                try:
                    return float(label)
                except ValueError:
                    return None

    return None


# Example usage function with batch processing
def run_benchmark_suite(model, tokenizer, device="cuda", num_samples=100, batch_size=8):
    """Run a suite of benchmark evaluations with batch processing."""

    benchmarks = [
        ("glue", "mrpc"),
        ("glue", "sst2"),
        ("glue", "cola"),
        ("glue", "qqp"),
        ("super_glue", "boolq"),
        ("super_glue", "copa"),
        ("mmlu", "formal_logic"),  
    ]

    results = {}

    for benchmark_name, task_name in benchmarks:
        print(f"\n{'='*50}")
        print(f"Running {benchmark_name}/{task_name}")
        print(f"{'='*50}")

        result = evaluate_llm_benchmark(
            model=model,
            tokenizer=tokenizer,
            benchmark_name=benchmark_name,
            task_name=task_name,
            num_samples=num_samples,
            device=device,
            few_shot_examples=2,  # Use 2-shot prompting
            batch_size=batch_size,  # Use batch processing
        )

        if result:
            results[f"{benchmark_name}_{task_name}"] = result
            print(f"Results: {result}")
        else:
            print(f"Failed to evaluate {benchmark_name}/{task_name}")

    return results