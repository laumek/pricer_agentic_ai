"""
train_ensemble.py

Train the linear regression ensemble model that combines:
- SpecialistAgent
- FrontierAgent
- RandomForestAgent

and save it as `ensemble_model.pkl`.
"""

import pickle
from typing import List

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import chromadb
from tqdm import tqdm

from price_intel.data.items import Item
from price_intel.agents.specialist_agent import SpecialistAgent
from price_intel.agents.frontier_agent import FrontierAgent
from price_intel.agents.random_forest_agent import RandomForestAgent


DB_PATH = "products_vectorstore"
COLLECTION_NAME = "products"
TEST_PKL_PATH = "amazon_items_test.pkl"
ENSEMBLE_MODEL_PATH = "ensemble_model.pkl"


def description_from_item(item: Item) -> str:
    """
    Extract description from the Item
    """

    text = item.prompt.split("to the nearest dollar?\n\n", 1)[-1]
    text = text.split("\n\nPrice is $", 1)[0]
    return text


def load_test_items(path: str) -> List[Item]:
    with open(path, "rb") as f:
        items = pickle.load(f)
    print(f"Loaded {len(items):,} test items from {path}")
    return items


def main():
    # 1. Load test items
    test_items = load_test_items(TEST_PKL_PATH)

    # 2. Connect to Chroma for FrontierAgent
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # 3. Initialize agents
    specialist = SpecialistAgent()
    frontier = FrontierAgent(collection)
    random_forest = RandomForestAgent()

    specialists = []
    frontiers = []
    random_forests = []
    prices = []

    eval_slice = test_items[1000:1250]

    for item in tqdm(eval_slice, desc="Collecting ensemble training data"):
        text = description_from_item(item)
        s = specialist.price(text)
        f = frontier.price(text)
        r = random_forest.price(text)
        specialists.append(s)
        frontiers.append(f)
        random_forests.append(r)
        prices.append(item.price)

    mins = [min(s, f, r) for s, f, r in zip(specialists, frontiers, random_forests)]
    maxes = [max(s, f, r) for s, f, r in zip(specialists, frontiers, random_forests)]

    X = pd.DataFrame(
        {
            "Specialist": specialists,
            "Frontier": frontiers,
            "RandomForest": random_forests,
            "Min": mins,
            "Max": maxes,
        }
    )
    y = pd.Series(prices)

    # 4. Train linear regression
    np.random.seed(42)
    lr = LinearRegression()
    lr.fit(X, y)

    feature_columns = X.columns.tolist()
    print("Ensemble feature coefficients:")
    for feature, coef in zip(feature_columns, lr.coef_):
        print(f"{feature}: {coef:.2f}")
    print(f"Intercept = {lr.intercept_:.2f}")

    joblib.dump(lr, ENSEMBLE_MODEL_PATH)
    print(f"✓ Saved ensemble model to {ENSEMBLE_MODEL_PATH}")


if __name__ == "__main__":
    main()
