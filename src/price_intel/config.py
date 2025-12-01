"""
Specialist model configuration.

"""

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

PROJECT_NAME = "pricer"
HF_USER = "laureen-ai"

# Example run name from your notebook; you can change per training run
RUN_NAME = "2025-11-30_17.02.00"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"

# Revision of the fine-tuned model on HF
REVISION = "631505be96f028dc7f867303ec34742a9748c068"

# Name of the fine-tuned model on Hugging Face Hub
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Modal / infra related
CACHE_DIR = "/cache"
GPU_TYPE = "T4"
MIN_CONTAINERS = 0  # Change this to 1 if you want Modal to be always running (keep warm), otherwise it will go cold after 2 mins

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"
