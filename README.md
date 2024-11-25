![Project Logo](assets/silver_bullet.png)

# Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning

## Introduction

Low-rank adapters have become a standard approach for efficiently fine-tuning large language models (LLMs), but they often fall short of achieving the performance of full fine-tuning. We propose a method, **LoRA Silver Bullet** or **LoRA-SB**, that approximates full fine-tuning within low-rank subspaces using a carefully designed initialization strategy. We theoretically demonstrate that the architecture of LoRA-XS —w hich inserts a trainable rxr matrix between B and A while keeping other matrices fixed — provides the precise conditions needed for this approximation. We leverage its constrained update space to achieve optimal scaling for high-rank gradient updates while removing the need for hyperparameter tuning. We prove that our initialization offers an optimal low-rank approximation of the initial gradient and preserves update directions throughout training. Extensive experiments across mathematical reasoning, commonsense reasoning, and language understanding tasks demonstrate that our approach exceeds the performance of standard LoRA while using **27-90x** fewer parameters, and comprehensively outperforms LoRA-XS. Our findings establish that it is possible to simulate full fine-tuning in low-rank subspaces, and achieve significant efficiency gains without sacrificing performance.


## Environment
We recommend using a Conda environment to run the Python scripts for this project. Follow these commands to set up the environment and install the required libraries:
```
conda create -n lora-sb python=3.10
pip install -r requirements.txt
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

## Citation

If you use our work for your research, please cite our paper:

```

```