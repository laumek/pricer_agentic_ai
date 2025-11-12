"""
env_setup.py

Environment and authentication utilities.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login

def setup_environment():
    """Load API keys and set environment variables."""
    load_dotenv(override=True)
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return os.environ["HF_TOKEN"]

def login_huggingface(token: str):
    """Authenticate to Hugging Face Hub."""
    if not token:
        raise ValueError("Missing Hugging Face token (HF_TOKEN).")
    login(token, add_to_git_credential=True)
    print("✓ Logged into Hugging Face.")