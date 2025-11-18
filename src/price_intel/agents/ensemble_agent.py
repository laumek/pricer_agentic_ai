"""
EnsembleAgent

Combines:
- SpecialistAgent (LLM fine-tuned)
- FrontierAgent (RAG + LLM)
- RandomForestAgent (embedding + RF)

using a LinearRegression model trained offline and saved as `ensemble_model.pkl`.
"""
import os
import pandas as pd
import joblib

from price_intel.agents.agent import Agent
from price_intel.agents.specialist_agent import SpecialistAgent
from price_intel.agents.frontier_agent import FrontierAgent
from price_intel.agents.random_forest_agent import RandomForestAgent

class EnsembleAgent(Agent):

    name = "Ensemble Agent"
    color = Agent.YELLOW
    
    def __init__(
        self,
        collection,
        model_path: str = "ensemble_model.pkl",
    ):
        """
        Initialize the EnsembleAgent by constructing all sub-agents and
        loading the linear regression model used to combine them.

        :param collection: Chroma collection to use for FrontierAgent
        :param model_path: path to the trained ensemble model
        """
        self.log("Initializing Ensemble Agent")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Ensemble model not found at '{model_path}'. "
                f"Train it with `train_ensemble.py` before using this agent."
            )

        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.random_forest = RandomForestAgent()
        self.model = joblib.load('ensemble_model.pkl')

        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Run the ensemble model:
        - Ask each sub-agent to price the product
        - Feed those predictions to the linear model
        - Return the final weighted price

        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - collaborating with specialist, frontier, and random forest agents")

        specialist = self.specialist.price(description)
        frontier = self.frontier.price(description)
        random_forest = self.random_forest.price(description)
        
        X = pd.DataFrame({
            'Specialist': [specialist],
            'Frontier': [frontier],
            'RandomForest': [random_forest],
            'Min': [min(specialist, frontier, random_forest)],
            'Max': [max(specialist, frontier, random_forest)],
        })
        y = max(0, self.model.predict(X)[0])
        self.log(f"Ensemble Agent complete - returning ${y:.2f}")
        return y