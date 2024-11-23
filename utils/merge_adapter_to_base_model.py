from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
import argparse
import torch
import gc
import os
import json
from pathlib import Path
from safetensors import safe_open
from .initialization_utils import find_and_initialize


def merge_xs(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    if "llama" in args.base_model:
        if "Llama-3" in args.base_model:
            tokenizer = AutoTokenizer.from_pretrained(
                args.base_model,
                use_fast=True,
                device_map={"": "cuda"},
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(
                args.base_model,
                use_fast=True,
                device_map={"": "cuda"},
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            use_fast=True,
            device_map={"": "cuda"},
        )

    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        lora_config_dict = json.load(f)
    lora_config = LoraConfig(**lora_config_dict)
    # lora_config = PeftConfig.from_pretrained(args.adapter)
    # model = PeftModel.from_pretrained(model, args.adapter, config=lora_config)
    model = get_peft_model(model, lora_config)

    adapter_name = "default"
    peft_config_dict = {adapter_name: lora_config}

    peft_conf_dir = str(Path(args.adapter).parents[0])
    with open(os.path.join(peft_conf_dir, 'reconstr_config.json')) as fp:
        reconstr_config = json.load(fp)
    reconstr_type = reconstr_config['reconstruction_type']

    # in order to accelerate model preparation, svd iterations will be set to 1.
    reconstr_config['svd']['n_iter'] = 1

    find_and_initialize(model, peft_config_dict, adapter_name=adapter_name, reconstr_type=reconstr_type,
                        writer=None, reconstruct_config=reconstr_config)

    peft_model_weights = {}
    with safe_open(os.path.join(args.adapter, "adapter_model.safetensors"),
                   framework="pt", device="cpu") as f:
        for key in f.keys():
            peft_model_weights[key] = f.get_tensor(key)
    renamed_state_dict = {
        k.replace(
            "lora_A", "lora_A.default"
        ).replace(
            "lora_B", "lora_B.default"
        ).replace(
            "_lora_latent", ".default_lora_latent"): v
        for (k, v) in peft_model_weights.items() if "classifier.out_proj" not in k
    }
    model.load_state_dict(renamed_state_dict, strict=False)
    print("merging the LoRA into the base model.")
    model = model.merge_and_unload()
    print("Saving the merged model to disk.")
    model.save_pretrained(args.output_path,
                          #safe_serialization=True,
                          )
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge Adapter to Base Model')
    parser.add_argument('--base_model', type=str)
    parser.add_argument('--adapter', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    merge_xs(args)
