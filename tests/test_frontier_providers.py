"""
Tests for provider selection and price parsing in frontier_providers.
No network or API keys needed: the SDK clients are replaced by small fakes.
"""

from types import SimpleNamespace

import pytest

from price_intel.agents.frontier_providers import (
    PREFILL,
    PricePrediction,
    call_claude,
    call_openai_compatible,
    openai_messages_for,
    parse_price,
    price_from_claude_response,
    resolve_provider,
)


# --- Provider selection -------------------------------------------------------------------


def test_defaults_to_openai_when_only_openai_key_is_set():
    config = resolve_provider(env={"OPENAI_API_KEY": "sk-test"})
    assert config.name == "openai"
    assert config.model == "gpt-4o-mini"


def test_defaults_to_deepseek_when_its_key_is_set():
    # Original behaviour: DeepSeek wins whenever its key is present
    config = resolve_provider(env={"OPENAI_API_KEY": "sk-test", "DEEPSEEK_API_KEY": "ds-test"})
    assert config.name == "deepseek"


def test_never_picks_claude_implicitly():
    env = {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "ant-test"}
    assert resolve_provider(env=env).name == "openai"


def test_env_var_selects_claude_with_haiku():
    env = {"FRONTIER_PROVIDER": "claude", "ANTHROPIC_API_KEY": "ant-test"}
    config = resolve_provider(env=env)
    assert config.name == "claude"
    assert config.model == "claude-haiku-4-5"


def test_env_var_is_case_and_whitespace_insensitive():
    env = {"FRONTIER_PROVIDER": "  Claude ", "ANTHROPIC_API_KEY": "ant-test"}
    assert resolve_provider(env=env).name == "claude"


def test_explicit_argument_beats_env_var():
    env = {"FRONTIER_PROVIDER": "claude", "ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"}
    assert resolve_provider("openai", env=env).name == "openai"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown frontier provider"):
        resolve_provider("gemini", env={"OPENAI_API_KEY": "sk-test"})


def test_missing_key_for_chosen_provider_raises():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        resolve_provider("claude", env={"OPENAI_API_KEY": "sk-test"})


def test_empty_key_counts_as_missing():
    # setup_environment() sets missing keys to "", so "" must not count as configured
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        resolve_provider(env={"OPENAI_API_KEY": ""})


# --- Free-text price parsing (OpenAI / DeepSeek) -------------------------------------------


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("129.99", 129.99),
        ("$1,299.99", 1299.99),
        ("45", 45.0),
        ("about 45 dollars", 45.0),
        (" 12.5\n", 12.5),
        (".5", 0.5),
    ],
)
def test_parse_price_extracts_number(reply, expected):
    assert parse_price(reply) == pytest.approx(expected)


@pytest.mark.parametrize("reply", ["", None, "N/A", "I don't know", "0", "0.00", "-5", "-12.50"])
def test_parse_price_returns_none_for_unusable_replies(reply):
    assert parse_price(reply) is None


# --- OpenAI-compatible calls ----------------------------------------------------------------


def fake_openai_client(reply):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=3),
    )
    create = lambda **kwargs: response
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_openai_messages_keep_the_prefill():
    messages = openai_messages_for("A kettle", ["A toaster"], [25.0])
    assert messages[-1] == {"role": "assistant", "content": PREFILL}


def test_openai_call_returns_price_and_usage():
    result = call_openai_compatible(fake_openai_client("49.99"), "gpt-4o-mini", "A kettle", [], [])
    assert result.price == pytest.approx(49.99)
    assert (result.input_tokens, result.output_tokens) == (1000, 3)
    assert result.failure is None


def test_openai_unparseable_reply_is_an_explicit_failure():
    result = call_openai_compatible(fake_openai_client("sorry"), "gpt-4o-mini", "A kettle", [], [])
    assert result.price is None
    assert result.failure == "unparseable reply"


# --- Claude structured outputs --------------------------------------------------------------


def claude_response(price=None, stop_reason="end_turn"):
    parsed = PricePrediction(price=price) if price is not None else None
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=parsed,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=9),
    )


def test_claude_response_with_price():
    result = price_from_claude_response(claude_response(price=89.5))
    assert result.price == pytest.approx(89.5)
    assert result.failure is None
    assert (result.input_tokens, result.output_tokens) == (1200, 9)


@pytest.mark.parametrize(
    "response, failure",
    [
        (claude_response(stop_reason="refusal"), "refusal"),
        (claude_response(stop_reason="max_tokens"), "max_tokens"),
        (claude_response(price=None), "no structured output"),
        (claude_response(price=0.0), "non-positive price"),
        (claude_response(price=-10.0), "non-positive price"),
    ],
)
def test_claude_failures_return_none_with_reason(response, failure):
    result = price_from_claude_response(response)
    assert result.price is None
    assert result.failure == failure


def test_claude_call_sends_schema_and_no_prefill():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return claude_response(price=20.0)

    client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    result = call_claude(client, "claude-haiku-4-5", "A kettle", ["A toaster"], [25.0])

    assert result.price == pytest.approx(20.0)
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["output_format"] is PricePrediction
    assert [m["role"] for m in captured["messages"]] == ["user"]
    assert "temperature" not in captured


def test_claude_malformed_json_is_an_explicit_failure():
    def parse(**kwargs):
        # What the SDK raises when the reply doesn't validate against the schema
        PricePrediction.model_validate_json('{"price": ')

    client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    result = call_claude(client, "claude-haiku-4-5", "A kettle", [], [])
    assert result.price is None
    assert result.failure == "invalid structured output"
