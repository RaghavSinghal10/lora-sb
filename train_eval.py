import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from utils.data_utils import *
from models import *
import argparse
import warnings
from sklearn.metrics import matthews_corrcoef
import numpy as np
import wandb
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
from utils.data_utils import *
import os
from copy import deepcopy


def train_client(model, dataloader, optimizer, scheduler, args):

    scaler = GradScaler()
    model.train()

    for step, data in enumerate(tqdm(dataloader)):
        data = {k: v.to(args.device) for k, v in data.items()}

        with autocast():
            outputs = model(**data)
            loss = outputs.loss

        wandb.log({"client_loss": loss.detach().cpu().numpy()})

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    return model.state_dict()


def calculate_metrics(all_true_labels, all_predictions, task):
    if task == "cola":
        return accuracy_score(all_true_labels, all_predictions), matthews_corrcoef(
            all_true_labels, all_predictions
        )
    elif task in ["sst2", "qnli", "rte", "wnli"]:
        return accuracy_score(all_true_labels, all_predictions), None
    elif task == "mrpc":
        return f1_score(all_true_labels, all_predictions), accuracy_score(
            all_true_labels, all_predictions
        )
    elif task == "stsb":
        return (
            pearsonr(all_true_labels, all_predictions)[0],
            spearmanr(all_true_labels, all_predictions)[0],
        )
    elif task == "qqp":
        return accuracy_score(all_true_labels, all_predictions), f1_score(
            all_true_labels, all_predictions
        )
    elif task in ["mnli_matched", "mnli_mismatched"]:
        return accuracy_score(all_true_labels, all_predictions), None
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_glue(model, dataloader, args, max_metric1, max_metric2):

    model.eval()
    eval_loss = 0
    all_predictions = []
    all_true_labels = []

    for batch in dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():

            outputs = model(**batch)

            eval_loss += outputs.loss.detach().cpu().numpy()

            if args.task == "stsb":
                predictions = outputs.logits.squeeze().cpu().numpy()
            else:
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_true_labels.extend(batch["labels"].cpu().numpy())

    eval_loss /= len(dataloader)

    # Calculate the metrics for the specific task
    metric1, metric2 = calculate_metrics(all_true_labels, all_predictions, args.task)

    if metric1 > max_metric1:
        max_metric1 = metric1

    if metric2 is not None and metric2 > max_metric2:
        max_metric2 = metric2

    print(f"{args.task} - Eval Loss: {eval_loss:.4f}, Metric 1: {metric1:.4f}")
    if metric2 is not None:
        print(f"{args.task} - Metric 2: {metric2:.4f}")
    print(f"{args.task} - Max Metric 1: {max_metric1:.4f}")
    if max_metric2 is not None:
        print(f"{args.task} - Max Metric 2: {max_metric2:.4f}")

    wandb.log(
        {
            f"eval_loss": eval_loss,
            f"metric1": metric1,
            f"metric2": metric2 if metric2 is not None else 0,
            f"max_metric1": max_metric1,
            f"max_metric2": max_metric2 if max_metric2 is not None else 0,
        }
    )

    return max_metric1, max_metric2


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

