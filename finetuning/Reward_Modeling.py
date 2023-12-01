from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import os
import argparse
import evaluate
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase, TrainingArguments, Trainer, TrainerCallback
from utils import (
    create_datasets,
    process_datasets,
    get_train_model,
    get_tokenizer
)
from transformers.utils import PaddingStrategy

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str, default="yitingxie/rlhf-reward-datasets")
    parser.add_argument("--output_dir", type=str, default="finetuning/result/RM/")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()

    gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.microl_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        local_rank=args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=args.bf16,
        logging_steps=10,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="wandb" if use_wandb else None,
        run_name=args.wandb_run_name if use_wandb else None,
    )
    
    tokenizer = get_tokenizer(args)

    model = get_train_model(args, "rm")

    train_dataset, eval_dataset = create_datasets(args, "rm")
    train_dataset = train_dataset.map(
        process_datasets,
        fn_kwargs={"tokenizer": tokenizer, "task_type": "rm"}
    )
    eval_dataset = eval_dataset.map(
        process_datasets, 
        fn_kwargs={"tokenizer": tokenizer, "task_type": "rm"}
    )

    original_columns = train_dataset.column_names

    # Turn the dataset into pairs of post + summaries. Then tokenizer it.
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["text_chosen"], examples["text_rejected"]):
            tokenized_chosen = tokenizer(chosen, truncation=True)
            tokenized_rejected = tokenizer(rejected, truncation=True)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.max_length and len(x["input_ids_rjected"]) <= args.max_length
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.max_length and len(x["input_ids_chosen"]) <= args.max_length
    )


    @dataclass
    class RewardDataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            features_chosen = []
            features_rejected = []
            for feature in features:
                features_chosen.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"],
                    }
                )
                features_rejected.append(
                    {
                        "input_ids": feature["input_ids_k"],
                        "attention_mask": feature["attention_mask_k"],
                    }
                )
            batch_chosen = self.tokenizer.pad(
                features_chosen,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch_rejected = self.tokenizer.pad(
                features_rejected,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch = {
                "input_ids_chosen": batch_chosen["input_ids"],
                "attention_mask_chosen": batch_chosen["attention_mask"],
                "input_ids_rejected": batch_rejected["input_ids"],
                "attention_mask_rejected": batch_rejected["attention_mask"],
                "return_loss": True,
            }
            return batch

    # Define the metric that we'll use for validation.
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        # Here, predictions is rewards_chosen and rewards_rejected.
        # We want to see how much of the time rewards_chosen > rewards_rejected.
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)


    class RewardTrainer(Trainer):
        # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
        def compute_loss(self, model, inputs, return_outputs=False):
            rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
            rewards_rejected = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])[0]
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
            if return_outputs:
                return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
            return loss


    # Train the model, woohoo.
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_length),
    )


    if args.eval_first_step:

        class EvaluateFirstStepCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_evaluate = True

        trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()

    model.save_pretrained(args.output_dir + "RM_last_checkpoint/")