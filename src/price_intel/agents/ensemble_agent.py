"""
EnsembleAgent

Combines:
- SpecialistAgent (LLM fine-tuned)
- FrontierAgent (RAG + LLM)
- RandomForestAgent (embedding + RF)

using a LinearRegression model trained offline with `train_ensemble.py`.

The regression weights depend on which LLM the FrontierAgent used when they were fitted,
so each provider gets its own file: models/ensemble_model_<provider>.pkl.
The original models/ensemble_model.pkl (fitted on gpt-4o-mini) is the fallback.
"""
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib

from price_intel.agents.agent import Agent
from price_intel.agents.specialist_agent import SpecialistAgent
from price_intel.agents.frontier_agent import FrontierAgent
from price_intel.agents.random_forest_agent import RandomForestAgent
from price_intel.config import MODEL_DIR

LEGACY_MODEL_PATH = MODEL_DIR / "ensemble_model.pkl"
LEGACY_MODEL_PROVIDER = "openai"


def ensemble_model_path(provider: str) -> Path:
    """
    Where the ensemble fitted on this frontier provider's predictions is stored.
    """
    return MODEL_DIR / f"ensemble_model_{provider}.pkl"


class EnsembleAgent(Agent):

    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(
        self,
        collection,
        model_path: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize the EnsembleAgent by constructing all sub-agents and
        loading the linear regression model used to combine them.

        :param collection: Chroma collection to use for FrontierAgent
        :param model_path: path to the trained ensemble model; defaults to the one for the provider
        :param provider: frontier LLM provider; defaults to FRONTIER_PROVIDER
        """
        self.log("Initializing Ensemble Agent")

        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection, provider=provider)
        self.random_forest = RandomForestAgent()

        path = Path(model_path) if model_path else self.default_model_path(self.frontier.provider)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Ensemble model not found at '{path}'. "
                f"Train it with `train_ensemble.py` before using this agent."
            )
        self.model = joblib.load(path)

        self.log(f"Ensemble Agent is ready (model='{path}')")

    def default_model_path(self, provider: str) -> Path:
        """
        Prefer the ensemble fitted on this provider's predictions. Otherwise fall back to the
        original ensemble_model.pkl, warning if it was fitted on a different provider.
        """
        path = ensemble_model_path(provider)
        if path.exists():
            return path
        if provider != LEGACY_MODEL_PROVIDER:
            message = (
                f"No ensemble model fitted on '{provider}' predictions ({path.name}). "
                f"Falling back to {LEGACY_MODEL_PATH.name}, whose weights were fitted on "
                f"{LEGACY_MODEL_PROVIDER} - re-fit with `train_ensemble.py --provider {provider}`."
            )
            self.log(message)
            logging.warning(message)
        return LEGACY_MODEL_PATH

    def price(self, description: str) -> float:
        """
        Run the ensemble model:
        - Ask each sub-agent to price the product
        - Feed those predictions to the linear model
        - Return the final weighted price

        If the Frontier Agent can't produce a price, the mean of the other two is used in its place.

        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - collaborating with specialist, frontier, and random forest agents")

        specialist = self.specialist.price(description)
        frontier = self.frontier.price(description)
        random_forest = self.random_forest.price(description)

        if frontier is None:
            frontier = (specialist + random_forest) / 2
            self.log(f"Frontier Agent failed - using the mean of Specialist and Random Forest (${frontier:.2f})")

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
