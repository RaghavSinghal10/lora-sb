import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import argparse
import warnings
import os
from datetime import datetime
import numpy as np
import wandb
from train_eval import *
import json
import yaml
import atexit

from utils.data_utils import *
from models import *
from utils.initialization_utils import *
from utils.gradient_utils import *
from utils.misc import *

parser = argparse.ArgumentParser(description="GLUE fine-tuning with LoRA SB")

parser.add_argument("--task", type=str, default="cola", help="GLUE task to fine-tune on")
parser.add_argument("--model", type=str, default="roberta-large", help="Model name")
parser.add_argument("--lora_r", type=int, default=24, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha value")
parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout value")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--epochs", type=int, default=30, help="Number of rounds")
parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument('--device', type=str, default='cuda', help='Device')


args = parser.parse_args()

args.lora_alpha = args.lora_r

wandb.init(project="project-name", config=args)

# Register cleanup function
def cleanup_wandb():
    if wandb.run is not None:
        wandb.finish()

atexit.register(cleanup_wandb)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.method == "ours":
    args.no_xs = False
    args.eg = True
    args.use_eff_lr = True
    args.lora_alpha = args.lora_r


def finetune(task):

    GLUE_TASK_NUM_LABELS = {
    'cola': 2,     # Binary: acceptable/unacceptable
    'mnli': 3,     # Three-way: entailment, contradiction, neutral
    'mrpc': 2,     # Binary: paraphrase/not paraphrase
    'qnli': 2,     # Binary: entailment/not entailment
    'qqp': 2,      # Binary: duplicate/not duplicate
    'rte': 2,      # Binary: entailment/not entailment
    'sst2': 2,     # Binary: positive/negative
    'stsb': 1,     # Regression task: similarity score from 0-5
    'wnli': 2      # Binary: entailment/not entailment
    }
    
    num_labels = GLUE_TASK_NUM_LABELS[task]

    ##### create model
    model, tokenizer = create_model_tokenizer(num_labels, args)

    ##### data handling

    train_data, val_data, _ = load_and_preprocess_data(task=task, tokenizer=tokenizer, args=args)

    train_loader, val_loader = create_dataloader(train_data, args, shuffle=True), create_dataloader(val_data, args, shuffle=False)

    max_metric_1 = 0
    max_metric_2 = 0

    named_grads = None

    total_training_steps = len(train_loader) * args.epochs
    eff_lr = args.lr/(args.warmup_ratio*total_training_steps)

    named_grads = estimate_and_process_grads_torch_2(
        model=model,
        dataloader=train_loader,
        lr=eff_lr,
        num_samples=2,
    )
    
    ##### create peft model
    model, lora_config = create_peft_model(model, args)
    model.to(args.device)


    with open("config/reconstruct_config.yaml", 'r') as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    
    adapter_name = "default"
    peft_config_dict = {adapter_name: lora_config}

    reconstr_config['svd']['rank'] = args.lora_r


    named_grads_new = {}
    for keys in named_grads.keys():
        keys_new = 'base_model.model.' + keys
        named_grads_new[keys_new] = named_grads[keys]

    find_and_initialize_grad(
        model=model,
        peft_config=peft_config_dict,
        adapter_name=adapter_name,
        reconstr_type='svd',
        reconstruct_config=reconstr_config,
        writer=None,
        named_grads=named_grads_new,
    )

    model.to(args.device)

    # Ensure contiguous parameters
    for param in model.parameters():
        param.data = param.data.contiguous()
    
    if named_grads is not None:
        del named_grads

    param_counts = count_parameters(model, verbose=False)

    total_params = param_counts['total_trainable_params']  # e.g., 45.67K
    classifier_params = param_counts['classifier_params']   # e.g., 12.34K
    non_classifier_params = param_counts['non_classifier_params']

    wandb.log({"total_params": total_params, "classifier_params": classifier_params, "non_classifier_params": non_classifier_params})

    ##### setting up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    total_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )


    for epoch in tqdm(range(args.epochs), desc='Epochs'):
        model.train()
        
        # Calculate total steps for this epoch
        total_steps = len(train_loader)
        running_loss = 0
        
        # Create progress bar for batches within epoch
        progress_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', leave=False, total=total_steps)
        
        for step, data in progress_bar:
            data = {k: v.to(args.device) for k, v in data.items()}
            
            outputs = model(**data)
            loss = outputs.loss
            
            # Update running loss and progress bar
            running_loss = loss.item()
            progress_bar.set_postfix({'loss': f'{running_loss:.4f}'})
            
            wandb.log({
                "train_loss": loss.detach().cpu().float().numpy(),
                "epoch": epoch,
                "step": step
            })
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        max_metric_1, max_metric_2 = evaluate_glue(
        model, val_loader, args, max_metric_1, max_metric_2
        )


# Main execution
if __name__ == "__main__":
    task = args.task
    model = finetune(task)