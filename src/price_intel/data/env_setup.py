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
    os.environ["PUSHOVER_USER"] = os.getenv("PUSHOVER_USER", "")
    os.environ["PUSHOVER_TOKEN"] = os.getenv("PUSHOVER_TOKEN", "")


def login_huggingface() -> None:
    """Authenticate to Hugging Face Hub."""
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise ValueError("Missing Hugging Face token (HF_TOKEN). Make sure it's set in your .env file.")
    
    login(token, add_to_git_credential=True)
    print("✓ Logged into Hugging Face.")

