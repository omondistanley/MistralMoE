"""MMLU dataset loading/formatting and the dense base model + tokenizer loader.

Ported from moe_complete.ipynb cells 6-12 (lines 156-341). The notebook loaded
these into bare globals (`df`, `dataset`, `eval_dataset`, `tokenizer`,
`ANSWER_TOKENS`, ...); here each step is a function returning its result
explicitly instead of mutating module/notebook globals.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from .config import ANSWER_TO_IDX, MAX_LENGTH, MODEL_ID, build_answer_tokens


def format_mmlu_prompt(row) -> str:
    """Format MMLU questions for EVALUATION (no answer)."""
    return f"""Question: {row['prompt']}

A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}

Answer:"""


def format_mmlu_prompt_with_answer(row) -> str:
    """Format MMLU questions for TRAINING (includes answer)."""
    return f"""Question: {row['prompt']}

A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}

Answer: {row['answer']}"""


@dataclass
class MMLUDatasets:
    dataset: DatasetDict
    train_dataset: Dataset
    eval_dataset: Dataset
    train_df: pd.DataFrame
    eval_df: pd.DataFrame


def download_mmlu_csv() -> str:
    """Download the MMLU CSV via kagglehub and return the local dataset path."""
    import kagglehub

    path = kagglehub.dataset_download("peiyuanliu2001/mmlu-dataset")
    print("Dataset path:", path)
    return path


def load_mmlu_dataset(csv_path: str | None = None, test_size: float = 0.3, random_state: int = 42) -> MMLUDatasets:
    """Load, format, and split the MMLU dataset.

    Reproduces notebook cells 7-9: stratified train/eval split by subject,
    prompt formatting (with and without answer), and conversion to HF Datasets.
    """
    if csv_path is None:
        csv_path = download_mmlu_csv()

    df = pd.read_csv(f"{csv_path}/train.csv")

    stratify_labels = df["subject"] if "subject" in df.columns else None
    train_indices, eval_indices = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels if stratify_labels is not None else None,
    )

    df["split"] = "train"
    df.loc[eval_indices, "split"] = "eval"

    df["formatted_prompt"] = df.apply(format_mmlu_prompt, axis=1)
    df["formatted_prompt_with_answer"] = df.apply(format_mmlu_prompt_with_answer, axis=1)
    df["answer_idx"] = df["answer"].map(ANSWER_TO_IDX)

    train_df = df[df["split"] == "train"].copy()
    eval_df = df[df["split"] == "eval"].copy()

    columns_to_use = ["prompt", "formatted_prompt", "formatted_prompt_with_answer", "answer", "answer_idx"]
    if "subject" in train_df.columns:
        columns_to_use.insert(0, "subject")
    if "A" in train_df.columns:
        columns_to_use.extend(["A", "B", "C", "D"])

    train_dataset_raw = Dataset.from_pandas(train_df[columns_to_use], preserve_index=False)
    eval_dataset_raw = Dataset.from_pandas(eval_df[columns_to_use], preserve_index=False)

    dataset = DatasetDict({"train": train_dataset_raw, "test": eval_dataset_raw})

    print("Dataset prepared:")
    print(f"  Training samples: {len(train_dataset_raw):,}")
    print(f"  Evaluation samples: {len(eval_dataset_raw):,}")

    return MMLUDatasets(
        dataset=dataset,
        train_dataset=train_dataset_raw,
        eval_dataset=eval_dataset_raw,
        train_df=train_df,
        eval_df=eval_df,
    )


def load_base_model_and_tokenizer(model_id: str = MODEL_ID, device_map: dict | None = None):
    """Load the dense Mistral-7B base model (bf16) and its tokenizer.

    Reproduces notebook cells 10-12. Returns (model, tokenizer, answer_tokens).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = device_map or {"": "cuda:0"}

    print("Loading model without quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    answer_tokens = build_answer_tokens(tokenizer)
    print("Answer token IDs:")
    for letter, token_id in answer_tokens.items():
        print(f"  {letter}: {token_id} -> '{tokenizer.decode([token_id])}'")

    return model, tokenizer, answer_tokens


def make_tokenize_fn(tokenizer, max_length: int = MAX_LENGTH, with_answer: bool = True):
    """Build a `datasets.map`-compatible tokenize function.

    Reproduces `tokenize_function_for_training`/`tokenize_function_for_eval`
    (notebook lines 1735-1751 and 3177-3192, identical in both call sites,
    deduplicated here into one parameterized factory).
    """
    field = "formatted_prompt_with_answer" if with_answer else "formatted_prompt"

    def tokenize_fn(examples):
        return tokenizer(
            examples[field],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return tokenize_fn
