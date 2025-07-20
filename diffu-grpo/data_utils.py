from datasets import load_dataset, Dataset
import pandas as pd
from reward_func import extract_hash_answer

import random
import numpy as np
import torch
import os


def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pad_sequence(lists, padding_value, cut_len):
    new_lists = []
    for l in lists:
        if len(l) >= cut_len:
            new_lists.append(l[:cut_len])
        else:
            new_lists.append(l+[padding_value]*(cut_len-len(l)))
    return new_lists


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_cd4(split="train") -> Dataset:
    if split == 'train':
        data_file = '/openbayes/home/d1/data/cd4_train.jsonl'
    else:
        data_file = '/openbayes/home/d1/data/cd4_test.jsonl'
    dataset = load_dataset(
            'json',
            data_files=data_file,
            split=split,
              
            cache_dir='./cache',
            streaming=False,
        )
    dataset = dataset.rename_column('input', "prompt").rename_column('output', "response")
    return dataset


def preprocess_dataset(
    dataset: "Dataset",
    tokenizer: "PreTrainedTokenizer",
    cutoff_len: int
):
    def preprocess_func(examples):
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"prompt": [], "nums": [], "numbers": [], "target": []}
        for src, tgt in zip(examples['prompt'], examples['response']):
            prompt = src+ tokenizer.sep_token + tgt +tokenizer.eos_token
            model_inputs["prompt"].append(prompt)
            parts = list(map(int, src.split(',')))   # 全部转为 tensor
            model_inputs['nums'].append(parts[:4])                        # tensor([90, 11, 37, 95])
            model_inputs['numbers'].append(parts[:4])
            model_inputs['target'].append(parts[4])                      # tensor(55)

        return model_inputs

    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    dataset = dataset.map(
        preprocess_func,
        batched=True,            
        remove_columns=column_names,
        **kwargs
    )
    return dataset