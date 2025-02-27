import json
import os
import sys

import torch
from omegaconf import OmegaConf, DictConfig
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from src.diffuse.search import continuous
from src.diffuse.utils import (
    find_and_generate_prompts,
)
from src.diffuse.generate import generate
import hydra


@hydra.main(config_path="./", config_name="gemma", version_base="1.1")
def main(config: DictConfig):
    # Initialize wandb
    wandb.init(
        project=config.get("wandb_project", "gfn-od"),
        name=config.get("wandb_run_name", "reproduce_diffuse_paper"),
        config=OmegaConf.to_container(config, resolve=True),
    )

    # config = OmegaConf.load(sys.argv[1])
    pretrained_model_name = config.model

    tokenizer = AutoTokenizer.from_pretrained(
        config.cache_dir, use_fast=False, legacy=False, local_files_only=True
    )
    train_prompts = find_and_generate_prompts(config.train_prompts, tokenizer)
    test_prompts = find_and_generate_prompts(config.test_prompts, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        config.cache_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    if config.get("load_model", False):
        model = PeftModelForCausalLM.from_pretrained(
            config.cache_dir, model_id=config.ckpt_dir, is_trainable=True, local_files_only=True
        )
    else:
        peft_config = LoraConfig(
            r=4,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    continuous(model, tokenizer, train_prompts, **config.alg_config)

    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    with open(os.path.join(config.output_dir, "prompts.json"), "w") as f:
        json.dump(
            {p.alias: p.__dict__ for p in train_prompts + test_prompts}, f, indent=2
        )
    model.save_pretrained(config.output_dir)

    if config.get("generate", True):
        gen = generate(model, tokenizer, test_prompts, 50, 20, 64)
        with open(os.path.join(config.output_dir, "gen.json"), "w") as f:
            json.dump(gen, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
