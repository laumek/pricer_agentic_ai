"""
description.py
Converts Item objects into text suitable for embedding.
"""

from price_intel.data.items import Item

def extract_description(item: Item) -> str:
    """
    Remove the question and the price answer from the prompt,
    leaving only the item description text.
    """
    text = item.prompt.replace(
        "How much does this cost to the nearest dollar?\n\n", ""
    )
    return text.split("\n\nPrice is $")[0]
