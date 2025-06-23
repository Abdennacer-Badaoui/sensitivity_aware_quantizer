import torch
from datasets import load_dataset
from evaluate import load
from typing import Dict, List, Optional, Any


def evaluate_llm_benchmark(
    model,
    tokenizer,
    device: str = "cuda",
    few_shot_examples: int = 5,
    max_length: int = 512,
    temperature: float = 0.1,
    do_sample: bool = True,
    batch_size: int = 512, 
) -> Optional[Dict[str, Any]]:
    """
    Evaluate LLM performance on MMLU benchmark by subject.
    """
    try:
        print(f"Evaluating the model on the MMLU benchmark")
        dataset = load_dataset("cais/mmlu", "all", split="test")
        metric = load("accuracy")

        subjects = list(set(dataset["subject"]))
        subject_accuracies = {}

        config = {
            "input_keys": ["question", "choices"],
            "label_key": "answer",
            "labels": ["A", "B", "C", "D"],
            "task_type": "multiple_choice",
            "prompt_template": "Question: {question}\nA. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\nAnswer:",
        }

        for subject in subjects:
            subject_dataset = dataset.filter(lambda x: x["subject"] == subject)

            few_shot_prompt = ""
            if few_shot_examples > 0:
                few_shot_prompt = create_few_shot_prompt(
                    subject_dataset, config, few_shot_examples
                )

            model.eval()
            all_predictions = []
            all_labels = []

            for batch_start in range(0, len(subject_dataset), batch_size):
                batch_end = min(batch_start + batch_size, len(subject_dataset))
                batch_examples = subject_dataset.select(range(batch_start, batch_end))
                                
                batch_prompts = []
                batch_labels = []
                
                for example in batch_examples:
                    prompt = create_prompt(example, config, few_shot_prompt)
                    label = example[config["label_key"]]
                    
                    if label is not None:
                        batch_prompts.append(prompt)
                        batch_labels.append(label)
                
                if not batch_prompts:
                    continue
                    
                batch_responses = generate_batch_responses(
                    model,
                    tokenizer,
                    batch_prompts,
                    device,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                
                for i, response in enumerate(batch_responses):
                    if i < len(batch_examples):
                        prediction = extract_prediction(response)
                        if prediction is not None and i < len(batch_labels):
                            all_predictions.append(prediction)
                            all_labels.append(batch_labels[i])

            if len(all_predictions) > 0:
                subject_accuracy = metric.compute(predictions=all_predictions, references=all_labels)
                subject_accuracies[subject] = subject_accuracy["accuracy"]
            else:
                print(f"No valid predictions for subject: {subject}")

        if not subject_accuracies:
            print("No valid predictions for any subject")
            return None

        mean_accuracy = sum(subject_accuracies.values()) / len(subject_accuracies)
        print(f"\nMean accuracy across all subjects: {mean_accuracy:.4f}")

        return {
            "mean_accuracy": mean_accuracy,
            "subject_accuracies": subject_accuracies
        }

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
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length // 2,
        padding=True,
        pad_to_multiple_of=8,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    responses = []
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for i in range(outputs.shape[0]):
            input_length = inputs["input_ids"][i].shape[0]
            generated_tokens = outputs[i][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())

    return responses


def create_few_shot_prompt(dataset, config: Dict[str, Any], num_examples: int) -> str:
    """Create few-shot examples for the prompt."""
    few_shot_examples = []
    examples = dataset.select(range(min(num_examples, len(dataset))))

    for example in examples:
        prompt = create_prompt(example, config, "")
        answer = ["A", "B", "C", "D"][example[config["label_key"]]]
        few_shot_examples.append(f"{prompt} {answer}")

    return "\n\n".join(few_shot_examples) + "\n\n"


def create_prompt(example: Dict[str, Any], config: Dict[str, Any], few_shot_prompt: str = "") -> str:
    """Create prompt for a single example."""
    prompt_data = {
        "question": example["question"],
        "choice_a": example["choices"][0],
        "choice_b": example["choices"][1],
        "choice_c": example["choices"][2],
        "choice_d": example["choices"][3],
    }
    return few_shot_prompt + config["prompt_template"].format(**prompt_data)


def extract_prediction(response: str) -> Optional[int]:
    """Extract prediction from model response."""
    response = response.lower().strip()
    
    for i, letter in enumerate(["a", "b", "c", "d"]):
        if f" {letter}" in f" {response}" or f"{letter}." in response or response.startswith(letter):
            return i
            
    return 0  # Default to first option if no clear answer
