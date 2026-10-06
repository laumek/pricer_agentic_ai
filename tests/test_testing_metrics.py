"""
Tests for the evaluation metrics in price_intel.train.testing.
"""

import pytest

from price_intel.data.eval_data import description_from_prompt
from price_intel.train.testing import summarize


def test_summarize_counts_failures_separately_from_errors():
    summary = summarize([110.0, None, 50.0, 300.0], [100.0, 80.0, 100.0, 100.0])

    assert summary["items"] == 4
    assert summary["failures"] == 1
    assert summary["failure_rate"] == pytest.approx(0.25)
    # errors over the 3 scored items: 10, 50, 200
    assert summary["average_error"] == pytest.approx(260 / 3)
    # within 20%: only 110 vs 100
    assert summary["within_20_rate"] == pytest.approx(1 / 3)
    # green = error < $40 or < 20%: only 110 vs 100
    assert summary["hit_rate"] == pytest.approx(1 / 3)


def test_summarize_hit_rate_uses_the_40_dollar_rule():
    # $30 off on a $50 item is 60% off, but still "green" by the < $40 rule
    summary = summarize([80.0], [50.0])
    assert summary["hit_rate"] == 1.0
    assert summary["within_20_rate"] == 0.0


def test_summarize_all_failed():
    summary = summarize([None, None], [10.0, 20.0])
    assert summary["failure_rate"] == 1.0
    assert summary["average_error"] is None


def test_description_from_prompt_strips_question_and_price_prefix():
    prompt = "How much does this cost to the nearest dollar?\n\nKettle 1.7L\nStainless steel\n\nPrice is $"
    assert description_from_prompt(prompt) == "Kettle 1.7L\nStainless steel"
