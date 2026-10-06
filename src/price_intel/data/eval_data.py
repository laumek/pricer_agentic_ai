"""
eval_data.py

Loads held-out test items from the Hugging Face dataset created by upload_dataset_to_hf.py.
The test split keeps the order of amazon_items_test.pkl, so slices like [1000:1250] select
the same items as before - without unpickling Item (which downloads the Llama tokenizer).
"""

from dataclasses import dataclass
from typing import List, Optional

from price_intel.config import PREFIX, QUESTION
from price_intel.data.env_setup import setup_environment

DATASET_NAME = "laureen-ai/pricer-data"

# Test items used to fit the ensemble (train_ensemble.py). Keep them out of held-out evaluations.
ENSEMBLE_FIT_START, ENSEMBLE_FIT_END = 1000, 1250


@dataclass
class EvalItem:
    """
    A held-out product: its description (as the agents see it) and its true price.
    Exposes title and price like Item, so it works with Tester.
    """

    description: str
    price: float

    @property
    def title(self) -> str:
        return self.description.split("\n", 1)[0]


def description_from_prompt(prompt: str) -> str:
    """
    Strip the question and the "Price is $" suffix from a test prompt,
    leaving only the item description.
    """
    text = prompt.split(f"{QUESTION}\n\n", 1)[-1]
    return text.split(f"\n\n{PREFIX}", 1)[0]


def load_eval_items(start: int = 0, end: Optional[int] = None) -> List[EvalItem]:
    """
    Load test[start:end] from the Hugging Face dataset (needs HF_TOKEN in .env).
    """
    from datasets import load_dataset

    setup_environment()
    test = load_dataset(DATASET_NAME, split="test")
    end = len(test) if end is None else min(end, len(test))
    rows = test.select(range(start, end))
    items = [EvalItem(description_from_prompt(row["text"]), float(row["price"])) for row in rows]
    print(f"Loaded {len(items):,} test items [{start}:{end}] from {DATASET_NAME}")
    return items
