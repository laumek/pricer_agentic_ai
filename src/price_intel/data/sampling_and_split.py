"""
sampling_and_split.py

Samples and splits items into train/test datasets and saves them to disk.
"""

import random
import numpy as np
import pickle
from collections import defaultdict, Counter
from typing import List, Tuple
from price_intel.data.items import Item

def make_price_slots(items: List[Item]):
    """Group items by rounded price."""
    slots = defaultdict(list)
    for item in items:
        slots[round(item.price)].append(item)
    return slots

def balanced_sample(slots: dict) -> List[Item]:
    """Sample items to balance categories and prices."""
    np.random.seed(42)
    random.seed(42)
    sample = []
    for i in range(1, 1000):
        slot = slots[i]
        if i >= 240:
            sample.extend(slot)
        elif len(slot) <= 1200:
            sample.extend(slot)
        else:
            weights = np.array([1 if item.category == "Automotive" else 5 for item in slot])
            weights = weights / np.sum(weights)
            selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
            selected = [slot[idx] for idx in selected_indices]
            sample.extend(selected)
    print(f"✓ Created balanced sample of {len(sample):,} items.")
    return sample

def summarize_categories(sample: List[Item]):
    """Print a simple category distribution summary."""
    counts = Counter(item.category for item in sample)
    for cat, count in counts.items():
        print(f"{cat:<30} {count:>8}")
    return counts

def split_train_test(sample: List[Item], train_size: int = 400_000, test_size: int = 2_000):
    """Split into train and test sets with reproducibility."""
    random.seed(42)
    random.shuffle(sample)
    train = sample[:train_size]
    test = sample[train_size:train_size + test_size]
    print(f"✓ Split into {len(train):,} train / {len(test):,} test items.")
    return train, test

def save_pickle(train: List[Item], test: List[Item], prefix: str = "data"):
    """Save train/test splits as pickle files."""
    with open(f"{prefix}_train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open(f"{prefix}_test.pkl", "wb") as f:
        pickle.dump(test, f)
    print(f"✓ Saved {prefix}_train.pkl and {prefix}_test.pkl")
