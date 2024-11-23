Install requirements

```
pip install -r requirements.txt
```

# Run NLU experiments on GLUE

```
CUDA_VISIBLE_DEVICES={device_indices} python3 fed_train_glue.py --model=roberta_base --task=cola --agg_type=ours --num_clients=3 --lora_r=4 --rounds 50 --lr 1e-3 --local_epochs 3
```

change agg_type = "ours" for FedEx-LoRa, "normal" for FedIT, and "ffa" for FFA-LoRA
change task as per dataset name ("cola", "mrpc", "rte", "stsb", "sst2", "qnli")
change model as per model ("roberta-base" or "roberta-large")
change lora_r as per rank

# Run NLG experiments on E2E

```
CUDA_VISIBLE_DEVICES={device_indices} python3 fed_train_e2e_new.py --agg_type=ours --log --lora_r=4 --task=e2e --lr=2e-3 --num_clients=3 --local_epochs=5
```

change agg_type = "ours" for FedEx-LoRa, "normal" for FedIT, and "ffa" for FFA-LoRA
change lora_r as per rank

Code for evaluating E2E: https://github.com/tuetschek/e2e-metrics