# ============================
# Part 1. Dataset loading & preprocessing
# ============================

import json
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/zengdaojian/zhangjia/BioLatent/Qwen4B"
)
tokenizer.pad_token = tokenizer.eos_token

MAX_LEN = 128
# --------------------------------
# 1. load raw dataset
# --------------------------------
def extract_fields(example):
    meta_dict = json.loads(example["meta"])

    # label priority: gt > reference > struct_cot
    if meta_dict.get("gt"):
        label_value = str(meta_dict["gt"])
    elif meta_dict.get("reference"):
        label_value = str(meta_dict["reference"])
    else:
        struct_cot = example.get("struct_cot", "")
        match = re.search(r'"output"\s*:\s*"(\w+)"', struct_cot)
        label_value = match.group(1) if match else ""

    return {
        "query": example.get("query", ""),
        "input_smiles": meta_dict.get("molecule", "none"),
        "label": label_value,
    }
def llm_tokenize(example):
        """
        Build causal LM training sample:
        [PROMPT][ANSWER]
        loss only on ANSWER tokens
        """
        prompt = example["query"]
        answer = example["label"]

        full_text = prompt + tokenizer.eos_token + answer

        enc = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # -------- build labels --------
        labels = input_ids.copy()

        prompt_ids = tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=MAX_LEN,
        )["input_ids"]

        prompt_len = len(prompt_ids)

        # mask prompt part
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "smiles": example["input_smiles"],
        }

def load_data(path):
    ds = load_dataset("/zengdaojian/zhangjia/BioLatent/ChemCotDataset")["train"]
    print("Raw dataset example:")
    print(ds[0])

    # --------------------------------
    # 2. extract smiles / query / label
    # --------------------------------


    dataset = ds.map(
        extract_fields,
        batched=False,
        remove_columns=ds.column_names
    )

    # print("After extract_fields:")
    # print(dataset[0])

    # --------------------------------
    # 3. tokenize for LLM (text only)
    # --------------------------------


    
    dataset = dataset.map(
        llm_tokenize,
        batched=False,
        remove_columns=["query", "label", "input_smiles"]
    )
    
    return dataset
# load_data("/zengdaojian/zhangjia/BioLatent/ChemCotDataset/chemcotbench-cot")
# print("Final tokenized dataset example:")
# print(dataset[0])
