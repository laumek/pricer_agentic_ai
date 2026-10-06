"""
train_ensemble.py

Train the linear regression ensemble model that combines:
- SpecialistAgent
- FrontierAgent
- RandomForestAgent

and save it as models/ensemble_model_<provider>.pkl, where <provider> is the
frontier LLM provider used (openai / deepseek / claude).

Usage:
    python src/price_intel/train/train_ensemble.py --provider claude
"""

import argparse

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import chromadb
from tqdm import tqdm

from price_intel.data.eval_data import ENSEMBLE_FIT_END, ENSEMBLE_FIT_START, load_eval_items
from price_intel.agents.specialist_agent import SpecialistAgent
from price_intel.agents.frontier_agent import FrontierAgent
from price_intel.agents.random_forest_agent import RandomForestAgent
from price_intel.agents.ensemble_agent import ensemble_model_path


DB_PATH = "products_vectorstore"
COLLECTION_NAME = "products"


def parse_args():
    parser = argparse.ArgumentParser(description="Fit the ensemble's linear regression.")
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek", "claude"],
        help="Frontier LLM provider (defaults to FRONTIER_PROVIDER / the agent's default)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load test items
    test_items = load_eval_items(ENSEMBLE_FIT_START, ENSEMBLE_FIT_END)

    # 2. Connect to Chroma for FrontierAgent
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # 3. Initialize agents
    specialist = SpecialistAgent()
    frontier = FrontierAgent(collection, provider=args.provider)
    random_forest = RandomForestAgent()

    specialists = []
    frontiers = []
    random_forests = []
    prices = []
    skipped = 0

    for item in tqdm(test_items, desc="Collecting ensemble training data"):
        text = item.description
        s = specialist.price(text)
        f = frontier.price(text)
        r = random_forest.price(text)
        if f is None:
            # Don't fit on imputed values - skip items the frontier model couldn't price
            skipped += 1
            continue
        specialists.append(s)
        frontiers.append(f)
        random_forests.append(r)
        prices.append(item.price)

    print(f"Skipped {skipped} of {len(test_items)} items where the Frontier Agent returned no price")

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

    model_path = ensemble_model_path(frontier.provider)
    joblib.dump(lr, model_path)
    print(f"✓ Saved ensemble model ({frontier.provider} / {frontier.MODEL}) to {model_path}")


if __name__ == "__main__":
    main()
