from trl import SFTTrainer
from transformers import TrainingArguments
import huggingface_hub

from utils import (
    create_datasets,
    process_datasets,
    get_train_model,
    get_tokenizer,
    get_peft_config
)

import argparse
import os

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_token", type=str, help="Required to upload models to hub.")

    parser.add_argument("--neft_noise_alpha", type=int, help="NEFTune noise alpha")

    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceH4/no_robots")
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--sample_size", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--group_by_length", type=bool, default=False)
    parser.add_argument("--packing", type=bool, default=False)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetuning/result/SFT/"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()

    huggingface_hub.login(args.hf_token)
    
    peft_config = get_peft_config(args, "sft")

    model = get_train_model(args, "sft")

    tokenizer = get_tokenizer(args)
    
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.eval_strategy,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else None,
        run_name=args.wandb_run_name if use_wandb else None,
    )

    train_dataset, eval_dataset = create_datasets(args, "sft")
    train_dataset = train_dataset.map(process_datasets, fn_kwargs={"tokenizer": tokenizer, "task_type": "sft"})
    eval_dataset = eval_dataset.map(process_datasets, fn_kwargs={"tokenizer": tokenizer, "task_type": "sft"})

    peft_config = get_peft_config(args, "sft")
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=args.packing,
        dataset_text_field="text",
        max_seq_length=args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        neftune_noise_alpha=args.neft_noise_alpha if args.neft_noise_alpha else None
    )

    trainer.train()