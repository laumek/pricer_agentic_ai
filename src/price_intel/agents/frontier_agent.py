"""
FrontierAgent

RAG-style agent:
- Encodes query description with SentenceTransformer
- Retrieves similar products from Chroma
- Calls OpenAI or DeepSeek chat model with those examples as context
- Extracts a numeric price from the model's answer
"""


import os
import re
import torch
from typing import List, Dict, Tuple

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.models.Collection import Collection

from price_intel.agents.agent import Agent
from price_intel.data.env_setup import setup_environment

# Load API keys from .env and set environment variables
setup_environment()

class FrontierAgent(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE

    DEFAULT_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, collection: Collection):
        """
        Set up this instance by connecting to OpenAI or DeepSeek, to the Chroma datastore,
        and initializing the embedding model.

        :param collection: a Chroma collection containing product documents & metadata
        """
        self.log("Initializing Frontier Agent")

        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if deepseek_api_key:
            self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            self.MODEL = "deepseek-chat"
            self.log("Frontier Agent is set up with DeepSeek")

        elif openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
            self.MODEL = self.DEFAULT_MODEL
            self.log("Frontier Agent is set up with OpenAI")

        else:
            raise RuntimeError(
                "No LLM API key found. Provide either:\n"
                " - OPENAI_API_KEY\n"
                " - DEEPSEEK_API_KEY\n"
                "Make sure you called setup_environment() before using FrontierAgent."
            )


        self.collection = collection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(self.EMBEDDING_MODEL, device=device)


        self.log(
            f"Frontier Agent is ready "
            f"(llm='{self.MODEL}', embedding_model='{self.EMBEDDING_MODEL}')"
        )


    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt

        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = ("To provide some context, here are some other items that might be similar"
        "to the item you need to estimate.\n\n")

        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI/ Deepseek
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by the LLM
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def find_similars(self, description: str, k: int = 5) -> Tuple[List[str], List[float]]:
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find {k} similar products")
        vector = self.encoder.encode([description])  # shape (1, d)
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(),
            n_results=k,
        )
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s) -> float:
        """
        Extract a floating point number from a string.

        :param s: string (LLM reply)
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Make a call to OpenAI or DeepSeek to estimate the price of the described product,
        by looking up k similar products and including them in the prompt to give context
        :param description: description of the product
        :return: predicted price (float)
        """
        documents, prices = self.find_similars(description, k=5)
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        response = self.client.chat.completions.create(
            model=self.MODEL, 
            messages=self.messages_for(description, documents, prices),
            seed=42,
            max_tokens=5
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result