import argparse
import torch
import os
from tqdm import tqdm
from transformers import Adafactor, pipeline

from trl import PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from utils import (
    create_datasets,
    process_datasets,
    get_train_model,
    get_tokenizer,
    get_peft_config
)

tqdm.pandas()

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--reward_model_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="HuggingFaceH4/ultrafeedback_binarized")

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--ada_factor", type=bool, default=False)
    parser.add_argument("--early_stopping", type=bool, default=False)

    parser.add_argument("--adafactor", type=bool, default=False)
    parser.add_argument("--target_kl", type=float, default=0.1)
    parser.add_argument("--reward_baseline", type=float, default=0.0)
    parser.add_argument("--batched_gen", type=bool, default=False)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="finetuning/result/PPO/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_kl_coef", type=float, default=0.2)
    parser.add_argument("--adap_kl_ctrl", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--log_with", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    tokenizer = get_tokenizer(args)

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    reward_model_name = args.reward_model_path

    config = PPOConfig(
        ppo_epochs=args.num_epochs,
        model_name=args.model_path.split("/")[-1],
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        batch_size=args.batch_size,
        mini_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        seed=args.seed,
        target_kl=args.target_kl,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=args.adap_kl_ctrl,
    )

    train_dataset, eval_dataset = create_datasets(args, "ppo")

    original_columns = train_dataset.column_names

    train_dataset = train_dataset.map(
        process_datasets, 
        fn_kwargs={"tokenizer": tokenizer, "task_type": "ppo"},
        batched=True,
        remove_columns=original_columns
    )
    eval_dataset = eval_dataset.map(
        process_datasets, 
        fn_kwargs={"tokenizer": tokenizer, "task_type": "ppo"},
        batched=True,
        remove_columns=original_columns
    )

    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
        "truncation": True,
    }

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    set_seed(args.seed)

    model = get_train_model(args, task_type="ppo")

    optimizer = None
    if args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=args.learning_rate,
        )

    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then build the sentiment analysis pipeline using our reward model, passing the
    # model name and the sentiment analysis pipeline arguments. Let's also make sure to
    # set the device to the same device as the PPOTrainer.
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model_name,
        device_map="auto",
        model_kwargs={"load_in_8bit": True},
        tokenizer=tokenizer,
        return_token_type_ids=False,
    )

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 32
    output_max_length = args.max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= config.total_ppo_epochs:
            break

        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute reward score (using the sentiment analysis pipeline)
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"] - args.reward_baseline) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if args.save_freq and epoch and epoch % args.save_freq == 0:
            ppo_trainer.save_pretrained(args.output_dir + f"step_{epoch}")