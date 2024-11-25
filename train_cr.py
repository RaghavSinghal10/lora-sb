import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import argparse
import warnings
import os
from datetime import datetime
import json
import yaml
import atexit
import wandb

from utils.data_utils import *
from models import *
from utils.initialization_utils import *
from utils.gradient_utils import *
from utils.misc import *
from utils.merge_adapter_to_base_model import *

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

def create_run_directory(args):
    """Create a directory structure for the current training run."""
    # Create base directory for all runs
    base_dir = "experiments/commonsense_reasoning"
    
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model name directory (simplified name)
    model_name = args.model.split('/')[-1]
    
    # Create run-specific directory with relevant parameters
    run_name = f"{model_name}__r{args.lora_r}__lr{args.lr}__train"
    
    # Final directory structure: experiments/model_name/YYYYMMDD_HHMMSS_parameters
    run_dir = os.path.join(base_dir, model_name, f"{timestamp}_{run_name}")
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    # Save run configuration
    config_dict = vars(args)
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    return run_dir

def finetune():
    run_dir = create_run_directory(args)
    
    # Initialize wandb with the run directory
    wandb_run_name = os.path.basename(run_dir)
    wandb_run = wandb.init(
        project="project-name",
        config=args,
        dir=os.path.join(run_dir, "logs")
    )

    # Save wandb run ID to a file
    with open(os.path.join(run_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb_run.id)
    
    # Create model and tokenizer
    model, tokenizer = create_model_tokenizer_it(args)
    
    # Data handling
    train_dataset = load_and_preprocess_cr(tokenizer=tokenizer, args=args)

    data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.eg_bs, 
        shuffle=True, 
        collate_fn=data_collator
    )
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)
    
    # Estimate gradients if selected
    named_grads = None

    total_training_steps = len(train_loader) * args.epochs
    eff_lr = args.lr/(args.warmup_ratio * total_training_steps)


    named_grads = estimate_and_process_grads_torch_3(
        model=model,
        dataloader=train_loader,
        lr=eff_lr,
        num_samples=170,
    )

    # Create peft model
    model, lora_config = create_peft_model_cr(model, args)
        
    reconstr_config_path = os.path.join(run_dir, "reconstruct_config.yaml")
    # Copy reconstruct config to run directory
    with open("config/reconstruct_config.yaml", 'r') as src, open(reconstr_config_path, 'w') as dst:
        reconstr_config = yaml.load(src, Loader=yaml.FullLoader)
        reconstr_config['svd']['rank'] = args.lora_r
        yaml.dump(reconstr_config, dst)

    # Save the required JSON file with the correct name
    json_path = os.path.join(run_dir, "reconstr_config.json")  # Note: reconstr not reconstruct
    with open(json_path, 'w') as f:
        json.dump(reconstr_config, f, indent=4)
    
    adapter_name = "default"
    peft_config_dict = {adapter_name: lora_config}

    
    named_grads_new = {f'base_model.model.{k}': v for k, v in named_grads.items()}

    find_and_initialize_grad(
        model=model,
        peft_config=peft_config_dict,
        adapter_name=adapter_name,
        reconstr_type='svd',
        reconstruct_config=reconstr_config,
        writer=None,
        named_grads=named_grads_new,
    )

    # Ensure contiguous parameters
    for param in model.parameters():
        param.data = param.data.contiguous()
    
    if named_grads is not None:
        del named_grads

    param_counts = count_parameters(model, verbose=False)

    total_params = param_counts['total_trainable_params']
    classifier_params = param_counts['classifier_params'] 
    non_classifier_params = param_counts['non_classifier_params']

    wandb.log({"total_params": total_params, "classifier_params": classifier_params, "non_classifier_params": non_classifier_params})


    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        seed=args.seed,
        report_to="wandb",
        gradient_accumulation_steps=32,
        save_strategy="no",
        bf16=True,
        tf32=False,
        fp16=False,
        logging_steps=1,
        logging_first_step=True,
        logging_dir=os.path.join(run_dir, "logs"),
    )
    
    # Save training arguments
    training_args_path = os.path.join(run_dir, "training_args.json")
    with open(training_args_path, 'w') as f:
        json.dump(training_args.to_dict(), f, indent=4)
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module,
        optimizers=(optimizer, None),
    )
    
    # # Save tokenizer
    tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))
    
    # Training
    model.config.use_cache = False
    trainer.train()
    
    # After training
    final_model_path = os.path.join(run_dir, "final_model")
    trainer.save_state()
    model.save_pretrained(final_model_path)

    
    return run_dir

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LoRA training with organized output")
    
    # Dataset arguments
    parser.add_argument("--data_path", type=str, default="data/commonsense/commonsense_170k.json", help="Path to the training data")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--lora_r", type=int, default=96, help="LoRA R value")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout value")
    parser.add_argument('--train_on_inputs', action='store_true', help='Train on inputs')
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=24, help="Gradient accumulation steps")
    parser.add_argument("--eg_bs", type=int, default=10, help="Batch size for gradient estimation")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--scheduler", type=str, default="linear", help="Learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # our method is hyperparam independent
    # however since lora scales by alpha/r, we need to set alpha to r for ease of implementation
    args.lora_alpha = args.lora_r
    
    # Run training
    run_dir = finetune()