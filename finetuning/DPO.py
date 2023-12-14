import os

import torch
import argparse
from transformers import TrainingArguments

from trl import DPOTrainer

from utils import (
    create_datasets,
    process_datasets,
    get_train_model,
    get_tokenizer,
    get_peft_config
)

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--num_proc", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="finetuning/result/DPO/")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", type=bool, default=False)

    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--logging_steps", type=int, default=1)

    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)


if __name__ == "__main__":
    args = args_parse()

    gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size

    model = get_train_model(args, "dpo")

    model_ref = get_train_model(args, "dpo")

    tokenizer = get_tokenizer(args)

    peft_config = get_peft_config(args, "dpo")

    train_dataset, eval_dataset = create_datasets(args, "dpo")

    original_columns = train_dataset.column_names

    train_dataset = train_dataset.map(
        process_datasets,
        fn_kwargs={"tokenizer": tokenizer, "task_type": "dpo"},
        remove_columns=original_columns
    )
    eval_dataset = eval_dataset.map(
        process_datasets,
        fn_kwargs={"tokenizer": tokenizer, "task_type": "dpo"},
        remove_columns=original_columns
    )

    train_dataset = train_dataset.filter(
        lambda x: len(x["text_prompt"]) + len(x["text_chosen"]) <= args.max_length
        and len(x["text_prompt"]) + len(x["text_rejected"]) <= args.max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["text_prompt"]) + len(x["text_chosen"]) <= args.max_length
        and len(x["text_prompt"]) + len(x["text_rejected"]) <= args.max_length
    )

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    training_args = TrainingArguments(
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.eval_strategy,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else None,
        run_name=args.wandb_run_name if use_wandb else None,
    )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )

    dpo_trainer.train()