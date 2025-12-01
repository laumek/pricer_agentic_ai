"""
SpecialistAgent

Wraps a fine-tuned LLM served remotely on Modal. This agent is used
as the "specialist" in the ensemble (alongside Frontier + RandomForest).
"""

import os
import modal

from price_intel.agents.agent import Agent


class SpecialistAgent(Agent):
    """
    An Agent that calls a fine-tuned LLM running remotely on Modal.
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(
        self,
        app_name: str | None = None,
        class_name: str | None = None,
    ):
        """
        Set up this Agent by creating an instance of the Modal class.

        :param app_name: Modal app name (defaults to env or "pricer-service")
        :param class_name: Modal class name (defaults to env or "Pricer")
        """
        self.log("Specialist Agent is initializing - connecting to Modal")

        app_name = app_name or 'pricer-service'
        class_name = class_name or 'Pricer'

        # Modal expects the app & class to already be deployed
        Pricer = modal.Cls.from_name(app_name, class_name)
        self.pricer = Pricer()

        self.log(
            f"Specialist Agent is ready "
            f"(app='{app_name}', class='{class_name}')"
        )

    def price(self, description: str) -> float:
        """
        Make a remote call to return the estimate of the price of this item.

        :param description: description of the product
        :return: predicted price as float
        """
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = float(self.pricer.price.remote(description))
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result