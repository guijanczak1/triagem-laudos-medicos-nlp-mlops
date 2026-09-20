"""Tests for triagem.data.urgency.

Covers the exact lexicons/prior/quantiles required by T4's acceptance
criteria, the scoring function's determinism, threshold fitting/mapping,
and JSON persistence round-trip.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from triagem.data.urgency import (
    CONDITION_PRIOR,
    HIGH_QUANTILE,
    LOW_QUANTILE,
    NORMAL_TERMS,
    THRESHOLDS_FILENAME,
    URGENT_TERMS,
    UrgencyThresholds,
    compute_urgency_score,
    fit_thresholds,
    load_thresholds,
    map_urgency,
    save_thresholds,
)


def test_urgent_terms_match_backlog_lexicon_exactly() -> None:
    """URGENT_TERMS must be exactly the lexicon specified in backlog.json T4."""
    assert URGENT_TERMS == (
        "acute",
        "severe",
        "malignant",
        "metastatic",
        "emergency",
        "hemorrhage",
        "sepsis",
        "shock",
        "infarction",
        "rupture",
        "obstruction",
        "failure",
        "fatal",
        "critical",
        "carcinoma",
    )


def test_normal_terms_match_backlog_lexicon_exactly() -> None:
    """NORMAL_TERMS must be exactly the lexicon specified in backlog.json T4."""
    assert NORMAL_TERMS == (
        "chronic",
        "benign",
        "mild",
        "routine",
        "stable",
        "follow-up",
        "screening",
        "elective",
        "remission",
        "asymptomatic",
        "well-controlled",
    )


def test_condition_prior_matches_backlog_exactly() -> None:
    """CONDITION_PRIOR values must match backlog.json T4 exactly."""
    assert CONDITION_PRIOR == {
        1: 0.15,
        2: 0.00,
        3: 0.05,
        4: 0.15,
        5: -0.05,
    }


def test_quantiles_match_backlog_exactly() -> None:
    """The fixed quantiles must be q=0.35 (low) and q=0.75 (high)."""
    assert LOW_QUANTILE == 0.35
    assert HIGH_QUANTILE == 0.75


class TestComputeUrgencyScore:
    """Behavior of compute_urgency_score."""

    def test_pure_urgent_text_scores_higher_than_pure_normal_text(self) -> None:
        urgent_text = "Acute severe emergency with hemorrhage and shock."
        normal_text = "Chronic mild routine stable follow-up screening."

        urgent_score = compute_urgency_score(urgent_text, condition_label=2)
        normal_score = compute_urgency_score(normal_text, condition_label=2)

        assert urgent_score > normal_score

    def test_neutral_text_score_equals_condition_prior(self) -> None:
        """No keyword hits => score is exactly the category prior."""
        text = "The patient was examined by the physician in the clinic today."
        for label, prior in CONDITION_PRIOR.items():
            assert compute_urgency_score(text, label) == pytest.approx(prior)

    def test_unknown_condition_label_defaults_prior_to_zero(self) -> None:
        text = "The patient was examined by the physician in the clinic today."
        assert compute_urgency_score(text, condition_label=999) == pytest.approx(0.0)

    def test_matching_is_case_insensitive(self) -> None:
        lower = compute_urgency_score("acute severe emergency", condition_label=2)
        upper = compute_urgency_score("ACUTE SEVERE EMERGENCY", condition_label=2)
        mixed = compute_urgency_score("Acute Severe Emergency", condition_label=2)
        assert lower == upper == mixed

    def test_matching_uses_word_boundaries_not_substrings(self) -> None:
        """'chronically' must not spuriously match the term 'chronic'."""
        substring_text = "chronically evolving presentation"
        exact_text = "chronic evolving presentation"

        substring_score = compute_urgency_score(substring_text, condition_label=2)
        exact_score = compute_urgency_score(exact_text, condition_label=2)

        # The whole-word hit must lower the score relative to no hit at all.
        assert exact_score < substring_score

    def test_hyphenated_terms_are_matched(self) -> None:
        follow_up_score = compute_urgency_score("routine follow-up visit", condition_label=2)
        no_hit_score = compute_urgency_score("routine appointment visit", condition_label=2)
        # "routine" and "follow-up" both hit NORMAL_TERMS -> more negative than
        # just "routine" alone.
        assert follow_up_score < no_hit_score

    def test_is_deterministic_across_repeated_calls(self) -> None:
        text = "Acute malignant carcinoma with metastatic spread and organ failure."
        first = compute_urgency_score(text, condition_label=1)
        second = compute_urgency_score(text, condition_label=1)
        assert first == second

    def test_empty_text_returns_prior_only(self) -> None:
        score = compute_urgency_score("", condition_label=2)
        assert score == pytest.approx(0.0)

    def test_score_is_length_invariant_given_the_same_keyword_hits(self) -> None:
        """T4 re-tuning: the score is raw hit count, NOT normalized by word count.

        Diluting a fixed set of keyword hits with a lot of neutral filler
        text must not change the score -- that length-sensitivity (a
        mismatch with how TF-IDF's L2 normalization represents documents)
        was the diagnosed root cause of the original formula's low ceiling.
        """
        short_text = "Acute severe emergency."
        padded_text = short_text + " " + "the patient was seen in clinic today " * 20

        assert compute_urgency_score(short_text, condition_label=2) == pytest.approx(
            compute_urgency_score(padded_text, condition_label=2)
        )

    def test_raw_hit_count_is_not_divided_by_word_count(self) -> None:
        """Two hits in a two-word text score the same as two hits in a longer one."""
        two_word_text = "Acute severe"
        longer_text = "Acute condition remains severe despite treatment and monitoring"

        assert compute_urgency_score(two_word_text, condition_label=2) == pytest.approx(
            compute_urgency_score(longer_text, condition_label=2)
        )


class TestFitThresholds:
    """Behavior of fit_thresholds."""

    def test_uses_fixed_quantiles_0_35_and_0_75(self) -> None:
        scores = [float(i) for i in range(100)]  # 0..99, evenly spaced

        thresholds = fit_thresholds(scores)

        assert thresholds.low == pytest.approx(34.65)
        assert thresholds.high == pytest.approx(74.25)
        assert thresholds.q_low == 0.35
        assert thresholds.q_high == 0.75

    def test_raises_on_empty_scores(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            fit_thresholds([])

    def test_deterministic_regardless_of_input_order(self) -> None:
        scores = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2]
        first = fit_thresholds(scores)
        second = fit_thresholds(list(reversed(scores)))
        assert first.low == pytest.approx(second.low)
        assert first.high == pytest.approx(second.high)


class TestMapUrgency:
    """Behavior of map_urgency."""

    def test_boundaries_are_normal_le_low_lt_atencao_lt_high_le_urgente(self) -> None:
        thresholds = UrgencyThresholds(low=0.0, high=0.2)

        assert map_urgency(-0.5, thresholds) == "normal"
        assert map_urgency(-0.001, thresholds) == "normal"
        assert map_urgency(0.0, thresholds) == "atencao"
        assert map_urgency(0.1, thresholds) == "atencao"
        assert map_urgency(0.199, thresholds) == "atencao"
        assert map_urgency(0.2, thresholds) == "urgente"
        assert map_urgency(1.0, thresholds) == "urgente"

    def test_return_value_always_in_allowed_label_set(self) -> None:
        thresholds = UrgencyThresholds(low=-0.1, high=0.1)
        for score in (-10.0, -0.1, 0.0, 0.1, 10.0):
            assert map_urgency(score, thresholds) in {"normal", "atencao", "urgente"}

    def test_no_class_drops_below_15_percent_on_realistic_continuous_scores(self) -> None:
        """Property check: fixed q=0.35/0.75 thresholds keep every class >= 15%."""
        rng = random.Random(42)
        scores = [rng.gauss(0.0, 0.1) for _ in range(5000)]
        thresholds = fit_thresholds(scores)

        labels = [map_urgency(s, thresholds) for s in scores]
        total = len(labels)
        for label in ("normal", "atencao", "urgente"):
            share = labels.count(label) / total
            assert share >= 0.15, f"{label} share {share:.3f} is below 15%"


class TestThresholdPersistence:
    """save_thresholds / load_thresholds round-trip."""

    def test_round_trip_preserves_values(self, tmp_path: Path) -> None:
        thresholds = UrgencyThresholds(low=-0.05, high=0.2)
        out_path = tmp_path / THRESHOLDS_FILENAME

        written_path = save_thresholds(thresholds, out_path, seed=42)
        loaded = load_thresholds(written_path)

        assert written_path == out_path
        assert loaded.low == pytest.approx(thresholds.low)
        assert loaded.high == pytest.approx(thresholds.high)
        assert loaded.q_low == LOW_QUANTILE
        assert loaded.q_high == HIGH_QUANTILE
        assert loaded.seed == 42

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        thresholds = UrgencyThresholds(low=0.0, high=0.1)
        out_path = tmp_path / "nested" / "dir" / THRESHOLDS_FILENAME

        save_thresholds(thresholds, out_path)

        assert out_path.exists()

    def test_written_json_is_human_readable(self, tmp_path: Path) -> None:
        thresholds = UrgencyThresholds(low=0.0, high=0.1)
        out_path = tmp_path / THRESHOLDS_FILENAME

        save_thresholds(thresholds, out_path, seed=7)

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["low"] == 0.0
        assert payload["high"] == 0.1
        assert payload["seed"] == 7
