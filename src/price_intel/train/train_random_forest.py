"""
train_random_forest.py

Train a RandomForestRegressor on Chroma embeddings and prices,
then save it as `random_forest_model.pkl`.
"""

from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import chromadb


DB_PATH = "products_vectorstore"
COLLECTION_NAME = "products"
MODEL_PATH = "src/price_intel/models/random_forest_model.pkl"


def load_chroma_vectors(db_path: str, collection_name: str):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)

    result = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(result["embeddings"], dtype=float)
    prices = np.array([m["price"] for m in result["metadatas"]], dtype=float)

    print(f"Loaded {embeddings.shape[0]:,} embeddings from Chroma.")
    return embeddings, prices


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X, y)
    return rf_model


def main():
    X, y = load_chroma_vectors(DB_PATH, COLLECTION_NAME)
    rf_model = train_random_forest(X, y)

    joblib.dump(rf_model, MODEL_PATH)
    print(f"✓ Saved RandomForest model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
