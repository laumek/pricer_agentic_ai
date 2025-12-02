"""
chroma_builder.py
Handles creation, deletion, and population of a Chroma vectorstore.
"""

import chromadb
from tqdm import tqdm
from typing import List

from price_intel.vectorstore.description import extract_description
from price_intel.vectorstore.embedder import Embedder
from price_intel.data.items import Item

class ChromaBuilder:
    def __init__(self, db_path: str = "products_vectorstore", collection_name: str = "products"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.db_path)

    def reset_collection(self):
        existing = [c.name for c in self.client.list_collections()]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        self.collection = self.client.create_collection(self.collection_name)
        print(f"Created new collection: {self.collection_name}")

    def ingest_items(self, items: List[Item], batch_size: int = 1000):
        embedder = Embedder()

        total = len(items)
        print(f"Ingesting {total:,} documents into Chroma...")

        for start in tqdm(range(0, total, batch_size)):
            batch = items[start : start + batch_size]

            documents = [extract_description(item) for item in batch]
            vectors = embedder.encode_batch(documents)
            metadatas = [{"category": item.category, "price": item.price} for item in batch]
            ids = [f"doc_{i}" for i in range(start, start + len(batch))]

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=vectors,
                metadatas=metadatas
            )

        print("✓ Ingestion complete.")
