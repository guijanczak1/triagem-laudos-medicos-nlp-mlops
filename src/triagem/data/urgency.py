"""Deterministic urgency heuristic: lexicon match + clinical category as tie-breaker.

Decided WITH the project user (see
``state/mlet-tech-challenge-fase-3/answers.md``, A1): the Medical Abstracts
TC Corpus has no native urgency label, only a clinical category
(``condition_label``). Mapping straight from category to urgency would make
the classification problem trivial -- the model would just be
re-identifying the disguised original category. Instead, urgency is derived
primarily from clinical keywords present in the abstract text, with the
clinical category acting only as a secondary, low-weight tie-breaker.

This is a DIDACTIC heuristic, not a validated clinical classification -- see
``docs/model_card.md`` (task T8).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

UrgencyLabel = Literal["normal", "atencao", "urgente"]

#: Keywords that push a text toward higher urgency. Versioned in code per T4.
URGENT_TERMS: tuple[str, ...] = (
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

#: Keywords that push a text toward lower urgency. Versioned in code per T4.
NORMAL_TERMS: tuple[str, ...] = (
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

#: Secondary, low-weight tie-breaker signal per clinical category
#: (``condition_label`` from ``medical_tc_labels.csv``). Per A1, keywords are
#: the primary signal; this only nudges borderline scores.
CONDITION_PRIOR: dict[int, float] = {
    1: 0.15,  # neoplasms
    2: 0.00,  # digestive system diseases
    3: 0.05,  # nervous system diseases
    4: 0.15,  # cardiovascular diseases
    5: -0.05,  # general pathological conditions
}

#: Fixed quantiles (of the TRAIN split's scores) that split the population
#: into normal / atencao / urgente. Fixed per T4 acceptance criteria.
LOW_QUANTILE = 0.35
HIGH_QUANTILE = 0.75

THRESHOLDS_FILENAME = "urgency_thresholds.json"


def _compile_term_pattern(terms: Sequence[str]) -> re.Pattern[str]:
    """Compile a case-insensitive, word-boundary alternation over ``terms``."""
    alternation = "|".join(re.escape(term) for term in terms)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_URGENT_PATTERN = _compile_term_pattern(URGENT_TERMS)
_NORMAL_PATTERN = _compile_term_pattern(NORMAL_TERMS)


def compute_urgency_score(text: str, condition_label: int) -> float:
    """Compute the deterministic urgency score for one abstract.

    ``(urgent keyword hits - normal keyword hits)``, normalized by the
    text's word count, plus ``CONDITION_PRIOR[condition_label]`` as a
    secondary, low-weight tie-breaker. A pure function of its inputs: the
    same text and category always yield the same score, which is what makes
    ``build_dataset`` reproducible byte-for-byte across runs.

    Args:
        text: Abstract text to score.
        condition_label: Clinical category id (1-5) from
            ``medical_tc_labels.csv``.

    Returns:
        The urgency score. Higher means more urgent.
    """
    urgent_hits = len(_URGENT_PATTERN.findall(text))
    normal_hits = len(_NORMAL_PATTERN.findall(text))
    word_count = max(len(text.split()), 1)
    prior = CONDITION_PRIOR.get(condition_label, 0.0)
    return (urgent_hits - normal_hits) / word_count + prior


@dataclass(frozen=True)
class UrgencyThresholds:
    """The two score cutoffs that separate normal / atencao / urgente.

    Attributes:
        low: Scores strictly below this are ``normal`` (the
            ``LOW_QUANTILE`` cutoff).
        high: Scores in ``[low, high)`` are ``atencao``; scores ``>= high``
            are ``urgente`` (the ``HIGH_QUANTILE`` cutoff).
        q_low: Quantile used to compute ``low`` (recorded for provenance).
        q_high: Quantile used to compute ``high`` (recorded for provenance).
        seed: Seed the owning ``build_dataset`` run was invoked with
            (recorded for provenance only -- the thresholds themselves are a
            deterministic function of the scores, not of the seed).
    """

    low: float
    high: float
    q_low: float = LOW_QUANTILE
    q_high: float = HIGH_QUANTILE
    seed: int | None = None


def fit_thresholds(scores: Sequence[float]) -> UrgencyThresholds:
    """Fit the fixed-quantile urgency thresholds on a set of scores.

    Uses the fixed quantiles ``LOW_QUANTILE`` (0.35) and ``HIGH_QUANTILE``
    (0.75). Must be called on the TRAIN split's scores only; the TEST split
    reuses the resulting thresholds (see ``triagem.data.build``) so no
    information leaks from test into the label definition.

    Args:
        scores: Urgency scores, typically the training split's.

    Returns:
        The fitted low/high cutoffs.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if len(scores) == 0:
        raise ValueError("cannot fit urgency thresholds on an empty score list")
    series = pd.Series(scores, dtype="float64")
    low = float(series.quantile(LOW_QUANTILE))
    high = float(series.quantile(HIGH_QUANTILE))
    return UrgencyThresholds(low=low, high=high)


def map_urgency(score: float, thresholds: UrgencyThresholds) -> UrgencyLabel:
    """Map a single score to ``{"normal", "atencao", "urgente"}`` via ``thresholds``."""
    if score < thresholds.low:
        return "normal"
    if score < thresholds.high:
        return "atencao"
    return "urgente"


def save_thresholds(thresholds: UrgencyThresholds, out_path: Path, seed: int | None = None) -> Path:
    """Persist ``thresholds`` as JSON to ``out_path``, creating parent dirs.

    Args:
        thresholds: Fitted thresholds to persist.
        out_path: Destination JSON file (e.g.
            ``data/processed/urgency_thresholds.json``).
        seed: Optional seed to record for provenance (see
            ``UrgencyThresholds.seed``).

    Returns:
        ``out_path``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(thresholds)
    if seed is not None:
        payload["seed"] = seed
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def load_thresholds(path: Path) -> UrgencyThresholds:
    """Load previously persisted thresholds from ``path``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return UrgencyThresholds(
        low=payload["low"],
        high=payload["high"],
        q_low=payload.get("q_low", LOW_QUANTILE),
        q_high=payload.get("q_high", HIGH_QUANTILE),
        seed=payload.get("seed"),
    )
