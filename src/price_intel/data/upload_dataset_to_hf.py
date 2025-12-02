# imports

import pickle
from datasets import Dataset, DatasetDict
from price_intel.data.env_setup import setup_environment, login_huggingface

setup_environment()
login_huggingface()


with open("amazon_items_train.pkl", "rb") as f:
    train = pickle.load(f)

with open("amazon_items_test.pkl", "rb") as f:
    test = pickle.load(f)

train_prompts = [item.prompt for item in train]
train_prices = [item.price for item in train]
test_prompts = [item.test_prompt() for item in test]
test_prices = [item.price for item in test]

train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

DATASET_NAME = "laureen-ai/pricer-data"
dataset.push_to_hub(DATASET_NAME, private = True)

