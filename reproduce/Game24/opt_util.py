import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

def load_model(args, device):
    if args.checkpoint_path:  
        model = PeftModel.from_pretrained(args.pretrained_model,args.checkpoint_path,is_trainable = True)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True,
                                                cache_dir=args.cache_dir)
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, add_bos_token=False)

    if args.use_lora and not args.test_only:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
    elif args.test_only and args.load_checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, args.load_checkpoint_path)

    return model, tokenizer
