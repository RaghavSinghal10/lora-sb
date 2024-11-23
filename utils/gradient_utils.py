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



def estimate_and_store_grads_torch(
    model,
    dataloader,
    lr,
    num_samples=170,
    save_checkpoints=[1, 50, 170],
    quant_flag=False,
    origin_type="bf16",
    quant_type="nf4",
    no_split_module_classes=None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Stores gradients with checkpoints at specific sample counts using PyTorch format.
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
    next_checkpoint_idx = 0
    
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
            
            for sample_idx in range(current_batch_size):
                if total_samples >= num_samples:
                    break
                
                named_grads.clear()
                
                sample_batch = {
                    k: v[sample_idx:sample_idx+1].to(accelerator.device) 
                    for k, v in batch.items()
                }
                
                if accelerator.is_main_process:
                    print(f"Processing sample {total_samples + 1}")
                
                outputs = model(**sample_batch)
                outputs.loss.backward()
                
                get_record_gradient_hook(model, named_grads)(None)
                
                for name, grad in named_grads.items():
                    if name not in running_grads_sum:
                        running_grads_sum[name] = grad.detach().cpu()
                    else:
                        running_grads_sum[name] += grad.detach().cpu()
                
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = None
                
                total_samples += 1
                
                if (next_checkpoint_idx < len(save_checkpoints) and 
                    total_samples == save_checkpoints[next_checkpoint_idx]):
                    
                    # Prepare checkpoint dictionary
                    checkpoint_dict = {
                        'num_samples': total_samples,
                        'gradients': {name: grad for name, grad in running_grads_sum.items()}
                    }
                    
                    # Synchronize for distributed training
                    if accelerator and accelerator.num_processes > 1:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(f"Saving checkpoint at sample {total_samples}")
                        for name in running_grads_sum:
                            grad = running_grads_sum[name].to(accelerator.device)
                            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                            running_grads_sum[name] = grad.cpu()
                            checkpoint_dict['gradients'][name] = grad.cpu()
                    
                    # Save checkpoint using PyTorch format
                    checkpoint_path = f"/shared/kponkshe/estimated_grads/llama-8b-cr/model_{total_samples}.pt"
                    if accelerator.is_main_process:
                        torch.save(checkpoint_dict, checkpoint_path)
                        print(f"Saved checkpoint at sample {total_samples}")
                    
                    next_checkpoint_idx += 1
                
                del outputs
                torch.cuda.empty_cache()
            
            if total_samples >= num_samples:
                break
            
        

    return total_samples


def estimate_and_process_grads_torch_2(
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
    Estimates and processes gradients for specified number of samples using PyTorch.
    Returns a dictionary of processed gradients.
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
            
            for sample_idx in range(current_batch_size):
                if total_samples >= num_samples:
                    break
                
                named_grads.clear()
                
                sample_batch = {
                    k: v[sample_idx:sample_idx+1].to(accelerator.device) 
                    for k, v in batch.items()
                }
                
                if accelerator.is_main_process:
                    print(f"Processing sample {total_samples + 1}")
                
                outputs = model(**sample_batch)
                outputs.loss.backward()
                
                get_record_gradient_hook(model, named_grads)(None)
                
                for name, grad in named_grads.items():
                    if name not in running_grads_sum:
                        running_grads_sum[name] = grad.detach().cpu()
                    else:
                        running_grads_sum[name] += grad.detach().cpu()
                
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = None
                
                total_samples += 1
                del outputs
                torch.cuda.empty_cache()
            
            if total_samples >= num_samples:
                break

    # Process final gradients
    processed_grads = {}
    
    # Synchronize for distributed training
    if accelerator and accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"Processing final gradients")
        for name in running_grads_sum:
            grad = running_grads_sum[name].to(accelerator.device)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            running_grads_sum[name] = grad.cpu()
    
    # Process gradients
    for name, grad in running_grads_sum.items():
        processed_grads[name] = (-1 * lr * torch.sign(grad))
    
    if accelerator.is_main_process:
        print(f"Finished processing gradients")

    return processed_grads