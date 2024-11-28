![Project Logo](assets/silver_bullet.png)

# Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning

## Introduction

Low-rank adapters have become a standard approach for efficiently fine-tuning large language models (LLMs), but they often fall short of achieving the performance of full fine-tuning. We propose a method, **LoRA Silver Bullet** or **LoRA-SB**, that approximates full fine-tuning within low-rank subspaces using a carefully designed initialization strategy. We theoretically demonstrate that the architecture of LoRA-XS — which inserts a trainable rxr matrix between B and A while keeping other matrices fixed — provides the precise conditions needed for this approximation. We leverage its constrained update space to achieve optimal scaling for high-rank gradient updates while removing the need for hyperparameter tuning. We prove that our initialization offers an optimal low-rank approximation of the initial gradient and preserves update directions throughout training. Extensive experiments across mathematical reasoning, commonsense reasoning, and language understanding tasks demonstrate that our approach exceeds the performance of standard LoRA while using **27-90x** fewer parameters, and comprehensively outperforms LoRA-XS. Our findings establish that it is possible to simulate full fine-tuning in low-rank subspaces, and achieve significant efficiency gains without sacrificing performance.


![LoRA-SB Image](assets/LoRA-SB.png)

LoRA-XS reduces parameter count compared to LoRA by inserting a trainable *r × r* matrix *R* between *B* and *A*, while keeping other matrices fixed, leading to *W = W<sub>0</sub> + sBRA*. Our method, LoRA-SB, leverages the same architecture. We find that updating *R* using its gradients *g<sup>R</sup>* is equivalent to updating the full-finetuning matrix *W* with an equivalent gradient *g̃<sub>SB</sub> = sBg<sup>R</sup>A*. We initialize *B*, *R*, and *A* such that the equivalent gradient *g̃<sub>SB</sub>* optimally approximates the full fine-tuning gradient *g* in low rank subspaces **at each training step**. In essence, we simulate the **entire full fine-tuning process** optimally within low-rank subspaces by **utilizing only the initial gradient *g<sub>1</sub>*** (shown in green) from full fine-tuning.

## Environment
We recommend using a Conda environment to run the Python scripts for this project. Follow these commands to set up the environment and install the required libraries:
```
conda create -n lora-sb python=3.10
conda activate lora-sb
pip install -r requirements.txt
```

## Quickstart

LoRA-SB is built on top of HuggingFace Transformers and PEFT libraries, making it incredibly easy to use. The following example demonstrates the minimal changes required to fine-tune a model using LoRA-SB.

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from utils.initialization_utils import find_and_initialize
from utils.gradient utils import estimate_and_process_grads_torch

model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype = torch.bfloat16
    ) 

# estimate update approximation for initialization
named_grads = estimate_and_process_grads_torch(
        model=model,
        dataloader=train_loader,
        num_samples=50,
    )

# set up a peft config
peft_config = LoraConfig(
        r=lora_rank,
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM", # assuming a decoder-only model
    )

# convert model to peft model
model = get_peft_model(model, peft_config)

with open("config/reconstruct_config.yaml", 'r') as stream:
    reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    
adapter_name = "default"  # assuming a single LoRA adapter per module to be transformed to LoRA-SB
peft_config_dict = {adapter_name: lora_config}

# specifying LoRA rank for the SVD initialization
reconstr_config['svd']['rank'] = lora_rank
    
named_grads_new = {f'base_model.model.{k}': v for k, v in named_grads.items()}

# convert to LoRA-SB model
find_and_initialize_grad(
    model=model,
    peft_config=peft_config_dict,
    adapter_name=adapter_name,
    reconstr_type='svd',
    reconstruct_config=reconstr_config,
    writer=None,
    named_grads=named_grads_new,
)

# perform training as usual

# You can merge LoRA-SB into the base model using `merge_and_unload` in PEFT
model = model.merge_and_unload() 
```

## Arithmetic Reasoning

To run the arithmetic reasoning experiments, execute:

```
bash scripts/run_arithmetic.sh
```

This script will fine-tune a model on the MetaMathQA dataset and evaluate its performance on the GSM8K and MATH benchmarks. You can modify the ``BASE_MODEL`` parameter to use a different model if desired.

## Commonsense Reasoning

To run the commonsense experiments, start by downloading the required datasets.

Begin by fetching the fine-tuning dataset available [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json). Place this file in the `data/commonsense` folder.

Next, for the evaluation phase, download the necessary datasets from [this link](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset). Ensure each dataset is saved in its appropriate subdirectory within `data/commonsense`.

To run the experiments, use:

```
bash scripts/run_cr.sh
```

This script will fine-tune a model on the Commonsense170K dataset and evaluate it across eight different datasets. You can modify the ``BASE_MODEL`` parameter to explore various models.



## Natural Language Understanding
To run experiments, run:

```
bash scripts/run_glue.sh
```

This script fine-tunes a RoBERTa-large model on the GLUE benchmark datasets. You can adjust the ``TASKS`` parameter to target different datasets as needed.
