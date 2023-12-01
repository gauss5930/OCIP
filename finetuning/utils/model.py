import os
from typing import Dict
from typing import List, Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

from peft import LoraConfig, AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification, get_peft_model
from trl import AutoModelForCausalLMWithValueHead

def get_train_model(args, task_type: Literal["sft", "rm", "ppo", "dpo"]):
    if task_type in ["sft", "dpo"]:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            low_cpu_mem_usage=True if task_type == "dpo" else None,
            torch_dtype=torch.bfloat16,
            use_cache=not args.gradient_checkpointing,
            use_flash_attention_2=True
        )

    elif task_type == "rm":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            num_labels=1,
            use_cache=not args.gradient_checkpointing,
            use_flash_attention_2=True
        )

        base_model = get_peft_model(base_model, get_peft_config(args, "rm"))

    elif task_type == "ppo":
        base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            use_cache=not args.gradient_checkpointing,
            peft_config=get_peft_config(args, "ppo")
        )

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    return base_model

def get_merged_model(args, task_type: Literal["sft", "rm", "dpo"]):
    if task_type in["sft", "dpo"]:
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    elif task_type == "rm":
        model = AutoPeftModelForSequenceClassification.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"{task_type} does not support.")
    
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.hf_hub_path:
        model.push_to_hub(args.hf_hub_path)
        tokenizer.push_to_hub(args.hf_hub_path)
    else:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not tokenizer.padding_side:
        tokenizer.padding_side = "right"

    if not tokenizer.chat_template:
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

    return tokenizer

def get_peft_config(args, task_type: Literal["sft", "rm", "ppo", "dpo"]):
    if task_type == "rm":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            task_type="TaskType.SEQ_CLS",
            inference_mode=False
        )

    elif task_type == "sft" or task_type == "dpo" or task_type == "ppo":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )

    return peft_config