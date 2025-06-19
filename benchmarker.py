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
    num_samples: int = 100,
    device: str = "cuda",
    few_shot_examples: int = 0,
    max_length: int = 512,
    temperature: float = 0.1,
    do_sample: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate LLM performance on various benchmark datasets using prompt-based evaluation.

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
            dataset = load_dataset("hendrycks_test", task_name, split="test")
            metric = load("accuracy")
        else:
            print(f"Benchmark {benchmark_name} not supported yet")
            return None

        # Limit number of samples
        if num_samples and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))

        print(f"Evaluating {len(dataset)} samples from {benchmark_name}/{task_name}")

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

        for i, example in enumerate(dataset):
            # Create prompt for this example
            prompt = create_prompt(example, task_config, few_shot_prompt)

            # Generate response
            response = generate_response(
                model,
                tokenizer,
                prompt,
                device,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
            )

            # Extract prediction from response
            prediction = extract_prediction(response, task_config, example)

            # Get ground truth label
            label = get_label(example, task_config)

            if prediction is not None and label is not None:
                all_predictions.append(prediction)
                all_labels.append(label)
            else:
                print(f"Skipping example {i}: prediction={prediction}, label={label}")

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
            # MMLU has many subjects, using a generic template
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
        elif "default" in configs[benchmark_name]:
            return configs[benchmark_name]["default"]

    return None


def create_few_shot_prompt(
    dataset, task_config: Dict[str, Any], num_examples: int
) -> str:
    """Create few-shot examples for the prompt."""
    few_shot_examples = []

    # Get a few examples from the dataset (use training split if available)
    examples = dataset.select(range(min(num_examples, len(dataset))))

    for example in examples:
        prompt = create_prompt(example, task_config, "")
        label = get_label(example, task_config)

        if task_config["task_type"] == "classification" and task_config["labels"]:
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


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_length: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """Generate response from the model."""

    # Tokenize input
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length // 2
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
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

            # Decode only the generated part
            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            # For models without generate method, use forward pass
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Get last token logits
            predicted_token_id = torch.argmax(logits).item()
            response = tokenizer.decode([predicted_token_id], skip_special_tokens=True)

    return response.strip()


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

        # For multiple choice, look for A, B, C, D
        if task_config["task_type"] == "multiple_choice":
            for letter in ["a", "b", "c", "d"]:
                if letter in response:
                    return ord(letter) - ord("a")

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

    # Handle MMLU format where answer is 0,1,2,3 for A,B,C,D
    if "answer" in example:
        return example["answer"]

    return None


# Example usage function
def run_benchmark_suite(model, tokenizer, device="cuda", num_samples=100):
    """Run a suite of benchmark evaluations."""

    benchmarks = [
        ("glue", "mrpc"),
        ("glue", "sst2"),
        ("glue", "cola"),
        ("glue", "qqp"),
        ("super_glue", "boolq"),
        ("super_glue", "copa"),
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
        )

        if result:
            results[f"{benchmark_name}_{task_name}"] = result
            print(f"Results: {result}")
        else:
            print(f"Failed to evaluate {benchmark_name}/{task_name}")

    return results
