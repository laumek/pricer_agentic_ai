"""
curate_data.py

Full pipeline orchestration: setup environment, load datasets, sample, split, and save.
"""

from price_intel.data.env_setup import setup_environment, login_huggingface
from price_intel.data.aggregate_items import load_all_items
from price_intel.data.sampling_and_split import (
    make_price_slots,
    balanced_sample,
    summarize_categories,
    split_train_test,
    save_pickle,
)

def main():
    # 1. Environment setup and Hugging Face login
    token = setup_environment()
    login_huggingface(token)

    # 2. Load all items
    items = load_all_items()

    # 3. Create balanced sample
    slots = make_price_slots(items)
    sample = balanced_sample(slots)

    # 4. Inspect category distribution
    summarize_categories(sample)

    # 5. Split into train/test
    train, test = split_train_test(sample)

    # 6. Save to disk
    save_pickle(train, test, prefix="amazon_items")

if __name__ == "__main__":
    main()
