import re
from typing import List, Literal, Optional

from datasets import load_dataset

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def create_datasets(args, task):
    task_dict = {"sft": ["train_sft", "test_sft"], "rm": ["train", "test"], "ppo": ["train_gen", "test_gen"], "dpo": ["train_prefs", "test_prefs"]}

    train_dataset = load_dataset(
        args.dataset_path,
        split=task_dict[task][0]
    )
    test_dataset = load_dataset(
        args.dataset_path,
        split=task_dict[task][1]
    )

    return train_dataset, test_dataset

def process_datasets(
        example, tokenizer, task_type: Literal["sft", "rm", "ppo", "dpo"]
):
    if task_type == "sft":
        messages = example["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation=False
        )

    elif task_type == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    elif task_type == "ppo":
        new_examples = {
            "query": [],
            "input_ids": []
        }
        for prompt in example["messages"]:
            if prompt[0]["role"] != "system":
                prompt.insert(0, {"role": "system", "content": ""})
            query = tokenizer.apply_chat_template(prompt[0:2], add_generation_prompt=True)
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question)

        return new_examples

    elif task_type == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]

            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
                
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

    else:
        raise ValueError(
            f"Task {task_type} not supported, please ensure that the provided task is one of {['sft', 'rm', 'dpo']}"
        )
    
    return example