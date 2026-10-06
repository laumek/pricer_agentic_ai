"""
frontier_providers.py

Everything the FrontierAgent needs to talk to an LLM provider:
- choosing a provider (OpenAI, DeepSeek or Claude) from config
- building the prompt
- calling the provider and turning its reply into a price

This module deliberately avoids torch / Chroma imports so it can be unit-tested cheaply.
"""

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import pydantic
from pydantic import BaseModel


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    api_key_env: str
    base_url: Optional[str] = None


PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
    "deepseek": ProviderConfig(
        "deepseek", "deepseek-chat", "DEEPSEEK_API_KEY", base_url="https://api.deepseek.com"
    ),
    "claude": ProviderConfig("claude", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
}

# OpenAI / DeepSeek continue the "Price is $" prefill, so a few tokens are enough.
OPENAI_MAX_TOKENS = 5
# Claude replies with JSON like {"price": 129.99} (~10 tokens); leave headroom so it isn't truncated.
CLAUDE_MAX_TOKENS = 256

SYSTEM_MESSAGE = "You estimate prices of items. Reply only with the price, no explanation"
PREFILL = "Price is $"


class PricePrediction(BaseModel):
    """Structured-output schema for Claude: {"price": number}"""

    price: float


@dataclass
class PriceResult:
    """
    Outcome of a single LLM call.
    price is None when no usable price could be extracted; failure says why.
    """

    price: Optional[float]
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    raw: str = ""
    failure: Optional[str] = None


def resolve_provider(
    explicit: Optional[str] = None, env: Optional[Mapping[str, str]] = None
) -> ProviderConfig:
    """
    Decide which provider to use:
    1. the explicit argument, if given
    2. otherwise the FRONTIER_PROVIDER environment variable
    3. otherwise the original behaviour: DeepSeek if DEEPSEEK_API_KEY is set, else OpenAI

    :raises ValueError: unknown provider name
    :raises RuntimeError: the chosen provider's API key is not set
    """
    env = os.environ if env is None else env
    choice = (explicit or env.get("FRONTIER_PROVIDER") or "").strip().lower()
    if not choice:
        choice = "deepseek" if env.get("DEEPSEEK_API_KEY") else "openai"

    if choice not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown frontier provider '{choice}'. Choose one of: {', '.join(PROVIDER_CONFIGS)}"
        )

    config = PROVIDER_CONFIGS[choice]
    if not env.get(config.api_key_env):
        raise RuntimeError(
            f"Frontier provider '{choice}' needs {config.api_key_env}.\n"
            "Add it to your .env file and make sure setup_environment() was called."
        )
    return config


def make_client(config: ProviderConfig):
    """
    Create the SDK client for this provider: the Anthropic SDK for Claude,
    the OpenAI SDK for OpenAI and DeepSeek (which exposes an OpenAI-compatible API).
    """
    api_key = os.environ[config.api_key_env]
    if config.name == "claude":
        import anthropic

        return anthropic.Anthropic(api_key=api_key)

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=config.base_url)


def make_context(similars: List[str], prices: List[float]) -> str:
    """
    Create context that can be inserted into the prompt

    :param similars: similar products to the one being estimated
    :param prices: prices of the similar products
    :return: text to insert in the prompt that provides context
    """
    message = (
        "To provide some context, here are some other items that might be similar"
        "to the item you need to estimate.\n\n"
    )
    for similar, price in zip(similars, prices):
        message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
    return message


def user_prompt_for(description: str, similars: List[str], prices: List[float]) -> str:
    """
    The user prompt shared by every provider, so they all see exactly the same question.
    """
    user_prompt = make_context(similars, prices)
    user_prompt += "And now the question for you:\n\n"
    user_prompt += "How much does this cost?\n\n" + description
    return user_prompt


def openai_messages_for(
    description: str, similars: List[str], prices: List[float]
) -> List[Dict[str, str]]:
    """
    Message list for OpenAI / DeepSeek: system + user prompt, plus the "Price is $" prefill.
    """
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt_for(description, similars, prices)},
        {"role": "assistant", "content": PREFILL},
    ]


def valid_price(value: Optional[float]) -> Optional[float]:
    """
    Return the price if it is a usable estimate (finite and > 0), otherwise None.
    """
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def parse_price(reply: Optional[str]) -> Optional[float]:
    """
    Extract a price from a free-text LLM reply (OpenAI / DeepSeek).
    Returns None, instead of 0.0, when there is no usable number.

    :param reply: string (LLM reply)
    """
    if not reply:
        return None
    s = reply.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if not match:
        return None
    return valid_price(float(match.group()))


def call_openai_compatible(
    client, model: str, description: str, similars: List[str], prices: List[float]
) -> PriceResult:
    """
    Price an item with OpenAI or DeepSeek (chat completions + regex parsing).
    """
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=openai_messages_for(description, similars, prices),
        seed=42,
        max_tokens=OPENAI_MAX_TOKENS,
    )
    latency = time.perf_counter() - start

    reply = response.choices[0].message.content or ""
    usage = response.usage
    price = parse_price(reply)
    return PriceResult(
        price=price,
        latency_s=latency,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        raw=reply,
        failure=None if price is not None else "unparseable reply",
    )


def call_claude(
    client, model: str, description: str, similars: List[str], prices: List[float]
) -> PriceResult:
    """
    Price an item with Claude, using structured outputs ({"price": number}) instead of regex.

    No assistant prefill (current Claude models reject it) and no temperature
    (the SDK no longer accepts it); the schema does the formatting work instead.
    """
    start = time.perf_counter()
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=SYSTEM_MESSAGE,
            messages=[{"role": "user", "content": user_prompt_for(description, similars, prices)}],
            output_format=PricePrediction,
        )
    except pydantic.ValidationError as e:
        # The SDK validates the JSON inside parse(); a truncated or malformed reply lands here
        return PriceResult(
            price=None,
            latency_s=time.perf_counter() - start,
            raw=str(e)[:200],
            failure="invalid structured output",
        )
    latency = time.perf_counter() - start
    return price_from_claude_response(response, latency)


def price_from_claude_response(response, latency_s: float = 0.0) -> PriceResult:
    """
    Turn a parsed Claude response into a PriceResult, treating refusals,
    truncation and non-positive prices as failures.
    """
    usage = response.usage
    result = PriceResult(
        price=None,
        latency_s=latency_s,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
    )

    if response.stop_reason == "refusal":
        result.failure = "refusal"
        return result
    if response.stop_reason == "max_tokens":
        result.failure = "max_tokens"
        return result

    parsed = response.parsed_output
    if parsed is None:
        result.failure = "no structured output"
        return result

    result.raw = str(parsed.price)
    result.price = valid_price(parsed.price)
    if result.price is None:
        result.failure = "non-positive price"
    return result


def estimate_price(
    client, config: ProviderConfig, description: str, similars: List[str], prices: List[float]
) -> PriceResult:
    """
    Price an item with whichever provider the config points at.
    API errors (network, auth, rate limits after the SDK's retries) are not caught here.
    """
    if config.name == "claude":
        return call_claude(client, config.model, description, similars, prices)
    return call_openai_compatible(client, config.model, description, similars, prices)
