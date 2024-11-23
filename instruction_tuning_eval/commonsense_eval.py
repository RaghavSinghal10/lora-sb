import argparse
import json
import re
import sys
import torch
import gc
import wandb
from tqdm.auto import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
MAX_INT = sys.maxsize


def extract_answer(dataset: str, sentence: str) -> str:
    """Extract the answer from model output based on dataset type."""
    sentence_ = sentence.strip().lower()
    
    if dataset == 'boolq':
        pred_answers = re.findall(r'true|false', sentence_)
    elif dataset == 'piqa':
        pred_answers = re.findall(r'solution1|solution2', sentence_)
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
    elif dataset == 'hellaswag':
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
    elif dataset == 'winogrande':
        pred_answers = re.findall(r'option1|option2', sentence_)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    return pred_answers[0] if pred_answers else ""


def batch_data(data_list, batch_size=1):
    """Split data into batches."""
    n = len(data_list)
    return [data_list[i:i + batch_size] for i in range(0, n, batch_size)]


def generate_prompt(instruction, input=None):
    """Generate prompt in the standard format."""
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def generate_response(model, tokenizer, prompt, device):
    """Generate response using transformers pipeline."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def commonsense_test(model_path, dataset_name, data_path, start=0, end=MAX_INT, batch_size=1):
    """Main evaluation function for commonsense tasks."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load dataset
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    dataset = dataset[start:end]
    instructions = [data.get('instruction') for data in dataset]
    answers = [data.get('answer') for data in dataset]
    
    # Batch the instructions
    batch_instructions = batch_data(instructions, batch_size=batch_size)

    res_completions = []
    result = []
    invalid_outputs = []

    # Generate responses
    print("\nGenerating responses...")
    for prompts in tqdm(batch_instructions, desc="Generating responses", ncols=100):
        if not isinstance(prompts, list):
            prompts = [prompts]
            
        for prompt in prompts:
            formatted_prompt = generate_prompt(prompt)
            completion = generate_response(model, tokenizer, formatted_prompt, device)
            res_completions.append(completion)

    # Evaluate responses
    print("\nEvaluating responses...")
    for instruction, completion, answer in tqdm(
        zip(instructions, res_completions, answers),
        total=len(instructions),
        desc="Evaluating answers",
        ncols=100
    ):
        pred = extract_answer(dataset_name, completion)
        is_correct = (pred == answer)
        result.append(is_correct)
        
        if not is_correct and not pred:
            temp = {'instruction': instruction, 'output': completion, 'answer': answer}
            invalid_outputs.append(temp)

    # Calculate and log metrics
    acc = sum(result) / len(result)
    wandb.log({
        f"eval/{dataset_name}_acc": acc,
    })

    print(f'Invalid outputs count: {len(invalid_outputs)}')
    print(f'Evaluation range: start={start}, end={end}')
    print(f'Total evaluated: {len(result)}, Accuracy: {acc:.4f}')
    
    return acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                      help="Path to the model")
    parser.add_argument("--dataset", type=str, required=True,
                      choices=["boolq", "piqa", "social_i_qa", "hellaswag",
                              "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                      help="Dataset to evaluate on")
    parser.add_argument("--data_file", type=str, default=None,
                      help="Path to the dataset file")
    parser.add_argument("--start", type=int, default=0,
                      help="Start index for evaluation")
    parser.add_argument("--end", type=int, default=MAX_INT,
                      help="End index for evaluation")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for evaluation")
    parser.add_argument("--run_dir", type=str,
                      help="Directory containing the wandb run ID")

    args = parser.parse_args()
    
    # Set default data file path if not provided
    if args.data_file is None:
        args.data_file = f'data/commonsense/{args.dataset}/test.json'

    # Initialize wandb
    if args.run_dir:
        try:
            with open(os.path.join(args.run_dir, "wandb_run_id.txt"), "r") as f:
                wandb_run_id = f.read().strip()
            wandb.init(
                id=wandb_run_id,
                project="project-name",
                resume="must"
            )
        except FileNotFoundError:
            print("WandB run ID file not found, starting new run")
            wandb.init(project="project-name")

    return args


if __name__ == "__main__":
    args = parse_args()
    commonsense_test(
        model_path=args.model,
        dataset_name=args.dataset,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
    )