import torch
from tqdm.auto import tqdm
from copy import deepcopy
from typing import Dict, List
from accelerate import Accelerator
import math
import gc
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from .offload_utils_for_quant import show_gpu_and_cpu_memory, OffloadContext

def get_record_gradient_hook(model, record_dict):
    """
    Creates a hook to record the gradients of a model's parameters into a dictionary.

    Args:
        model (torch.nn.Module): The model whose gradients will be recorded.
        record_dict (dict): A dictionary to store the recorded gradients.
    """

    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.detach().cpu()
                else:
                    record_dict[n] += p.grad.detach().cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_and_process_grads_torch(
    model,
    dataloader,
    lr,
    num_samples=170,
    quant_flag=False,
    origin_type="bf16",
    quant_type="nf4",
    no_split_module_classes=None,
) -> Dict[str, torch.Tensor]:
    """
    Estimates and processes gradients using batch-wise computation.
    Returns a dictionary of processed gradients.
    
    Args:
        model: The PyTorch model
        dataloader: DataLoader instance
        lr: Learning rate
        num_samples: Total number of samples to process
        quant_flag: Whether to use quantization
        origin_type: Original data type
        quant_type: Quantization type
        no_split_module_classes: Module classes to not split
    
    Returns:
        Dict[str, torch.Tensor]: Processed gradients
    """
    batch_size = dataloader.batch_size
    accelerator = Accelerator()
    
    if accelerator and model.device.type != "cuda":
        if not quant_flag:
            model.to(accelerator.device)
        else:
            model.to("cpu")
    
    model.train()
    dataloader = accelerator.prepare(dataloader)
    
    running_grads_sum = {}
    named_grads = {}
    total_samples = 0
    
    with OffloadContext(
        model=model,
        named_grads=named_grads,
        quant_flag=quant_flag,
        origin_type=origin_type,
        quant_type=quant_type,
        no_split_module_classes=no_split_module_classes,
    ):
        for batch in tqdm(dataloader, desc="Computing gradients"):
            current_batch_size = len(batch['input_ids'])
            samples_to_process = min(current_batch_size, num_samples - total_samples)
            
            if samples_to_process <= 0:
                break
                
            # Process only the needed portion of the batch
            batch = {k: v[:samples_to_process].to(accelerator.device) for k, v in batch.items()}
            
            if accelerator.is_main_process:
                print(f"Processing batch with {samples_to_process} samples")
            
            # Forward pass
            outputs = model(**batch)
            
            # Normalize loss by batch size to maintain scale
            (outputs.loss / samples_to_process).backward()
            
            # Record gradients
            get_record_gradient_hook(model, named_grads)(None)
            
            # Accumulate gradients
            for name, grad in named_grads.items():
                if name not in running_grads_sum:
                    running_grads_sum[name] = grad.detach().cpu()
                else:
                    running_grads_sum[name] += grad.detach().cpu()
            
            # Clear gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
            
            total_samples += samples_to_process
            named_grads.clear()
            del outputs
            torch.cuda.empty_cache()

    # Process final gradients
    processed_grads = {}
    
    # Synchronize for distributed training
    if accelerator and accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("Processing final gradients")
        for name in running_grads_sum:
            grad = running_grads_sum[name].to(accelerator.device)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            running_grads_sum[name] = grad.cpu()
    
    # Process gradients
    for name, grad in running_grads_sum.items():
        processed_grads[name] = (-1 * lr * torch.sign(grad))
    
    if accelerator.is_main_process:
        print("Finished processing gradients")

    return processed_grads