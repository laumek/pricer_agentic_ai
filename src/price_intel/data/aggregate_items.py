"""
aggregate_items.py

Loads multiple datasets using ItemLoader and aggregates them into a single list of Items.
"""

from typing import List
from price_intel.data.items import Item
from price_intel.data.loaders import ItemLoader


DATASET_NAMES = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]

def load_all_items(dataset_names: List[str] = DATASET_NAMES) -> List[Item]:
    """Load all items from predefined datasets."""
    all_items = []
    for name in dataset_names:
        loader = ItemLoader(name)
        items = loader.load()
        all_items.extend(items)
    print(f"✓ Loaded total of {len(all_items):,} items across {len(dataset_names)} categories.")
    return all_items