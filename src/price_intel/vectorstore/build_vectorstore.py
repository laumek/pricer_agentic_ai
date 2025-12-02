"""
build_vectorstore.py
Orchestrates loading pickled items and building the Chroma vectorstore.
"""

import pickle

from price_intel.data.env_setup import setup_environment, login_huggingface
from price_intel.vectorstore.chroma_builder import ChromaBuilder

def load_train_items(path: str = "amazon_items_train.pkl"):
    with open(path, "rb") as f:
        items = pickle.load(f)
    print(f"Loaded {len(items):,} training items from {path}")
    return items

def main():
    setup_environment()
    login_huggingface()

    items = load_train_items()

    builder = ChromaBuilder(
        db_path="products_vectorstore",
        collection_name="products"
    )
    builder.reset_collection()
    builder.ingest_items(items)

if __name__ == "__main__":
    main()
