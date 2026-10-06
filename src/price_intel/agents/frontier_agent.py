"""
FrontierAgent

RAG-style agent:
- Encodes query description with SentenceTransformer
- Retrieves similar products from Chroma
- Calls the configured LLM provider (OpenAI, DeepSeek or Claude) with those examples as context
- Returns the estimated price, or None if the provider's reply had no usable price

The provider is chosen with the FRONTIER_PROVIDER env var (openai / deepseek / claude);
see frontier_providers.resolve_provider for the defaults.
"""


from typing import List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

from price_intel.agents.agent import Agent
from price_intel.agents.frontier_providers import (
    PriceResult,
    estimate_price,
    make_client,
    resolve_provider,
)
from price_intel.data.env_setup import setup_environment

# Load API keys from .env and set environment variables
setup_environment()

class FrontierAgent(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, collection: Collection, provider: Optional[str] = None):
        """
        Set up this instance by connecting to the LLM provider, to the Chroma datastore,
        and initializing the embedding model.

        :param collection: a Chroma collection containing product documents & metadata
        :param provider: 'openai', 'deepseek' or 'claude'; defaults to FRONTIER_PROVIDER
        """
        self.log("Initializing Frontier Agent")

        self.config = resolve_provider(provider)
        self.provider = self.config.name
        self.MODEL = self.config.model
        self.client = make_client(self.config)
        self.failures = 0
        self.log(f"Frontier Agent is set up with {self.provider}")

        self.collection = collection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(self.EMBEDDING_MODEL, device=device)


        self.log(
            f"Frontier Agent is ready "
            f"(llm='{self.MODEL}', embedding_model='{self.EMBEDDING_MODEL}')"
        )

    def find_similars(self, description: str, k: int = 5) -> Tuple[List[str], List[float]]:
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log(f"Frontier Agent is performing a RAG search of the Chroma datastore to find {k} similar products")
        vector = self.encoder.encode([description])  # shape (1, d)
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(),
            n_results=k,
        )
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def estimate(self, description: str, documents: List[str], prices: List[float]) -> PriceResult:
        """
        Call the LLM only (no retrieval), given already-retrieved similar products.
        Used by price() and by the benchmark, which retrieves once and reuses the context.
        """
        return estimate_price(self.client, self.config, description, documents, prices)

    def price(self, description: str) -> Optional[float]:
        """
        Estimate the price of the described product with the configured LLM,
        by looking up k similar products and including them in the prompt to give context
        :param description: description of the product
        :return: predicted price, or None if the LLM's reply had no usable price
        """
        documents, prices = self.find_similars(description, k=5)
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        result = self.estimate(description, documents, prices)
        if result.price is None:
            self.failures += 1
            self.log(f"Frontier Agent could not get a price ({result.failure}) - returning None")
            return None
        self.log(f"Frontier Agent completed - predicting ${result.price:.2f}")
        return result.price
