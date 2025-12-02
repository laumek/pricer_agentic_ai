"""
RandomForestAgent

Uses a pre-trained RandomForestRegressor on sentence-transformer embeddings
to estimate the price of a product from its description.
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import joblib
from price_intel.agents.agent import Agent

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


class RandomForestAgent(Agent):

    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(
            self,
            model_filename: str = "random_forest_model.pkl",
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"     
                 ):
        """
        Initialize the Random Forest agent by loading the saved model
        and the SentenceTransformer encoder.

        :param model_path: path to the trained RandomForest model (joblib file)
        :param embedding_model: name of the SentenceTransformer model to use
        """
        self.log("Random Forest Agent is initializing") 

        model_path = MODEL_DIR / model_filename

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RandomForest model not found at '{model_path}'. "
                f"Train it with the training script before using this agent."
            )

        self.vectorizer = SentenceTransformer(embedding_model)
        self.model = joblib.load(model_path)

        self.log(
            f"Random Forest Agent is ready "
            f"(model='{model_path}', embedding_model='{embedding_model}')"
        )


    def price(self, description: str) -> float:
        """
        Use a Random Forest model to estimate the price of the described item
        :param description: free-text description of the product
        :return: the price as a float
        """        
        self.log("Random Forest Agent is starting a prediction")
        vector = self.vectorizer.encode([description]) # shape (1, d)
        result = max(0, self.model.predict(vector)[0])
        self.log(f"Random Forest Agent completed - predicting ${result:.2f}")
        return result