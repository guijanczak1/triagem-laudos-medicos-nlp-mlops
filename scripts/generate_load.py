"""Load generator to populate the provisioned Grafana dashboard (T19).

Fires ``n`` ``POST /predict`` requests against a running instance of the
triage API, using a deterministic **mix of texts across the three urgency
classes** (``normal``/``atencao``/``urgente``) drawn from the processed
dataset (same seam as ``scripts/measure_latency.py``'s ``sample_texts``),
so the Prometheus counters/histograms this project instruments
(``triagem.serving.metrics``, T17) have non-zero, varied data to plot on
``docker/grafana/dashboards/triagem_api.json`` (T19) once Grafana is open.

This is a load *generator* for observability, not a latency benchmark --
see ``scripts/measure_latency.py``/``docs/latency_baseline.md`` for
percentile measurement. No percentiles are computed here on purpose; this
script only needs to produce realistic, class-diverse traffic.

Usage::

    # against the local docker compose stack (T18): api on :8000
    docker compose up -d
    python scripts/generate_load.py --n 300

    # against a manually-started API (e.g. `poetry run uvicorn ...`)
    python scripts/generate_load.py --n 100 --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from triagem.data.loaders import VALID_LABELS, load_dataset

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class LoadResult:
    """Summary of one :func:`generate_load` run."""

    requested: int
    ok: int
    failed: int
    mean_latency_ms: float
    wall_elapsed_s: float

    @property
    def throughput_rps(self) -> float:
        """Requests completed per wall-clock second (0.0 if nothing ran)."""
        return self.ok / self.wall_elapsed_s if self.wall_elapsed_s > 0 else 0.0


def build_text_mix(n: int, seed: int = 42, data_path: Path | None = None) -> list[str]:
    """Return ``n`` texts sampled deterministically across the 3 urgency classes.

    Draws roughly ``n / 3`` texts per class (``normal``, ``atencao``,
    ``urgente``) from the processed dataset, then shuffles the combined pool
    with the same seed -- so ``triagem_predictions_total`` (T17) sees a
    realistic mix of predicted labels instead of one class dominating the
    whole run, and re-running with the same ``seed``/``data_path`` always
    produces the same request sequence.

    Args:
        n: Total number of texts to return.
        seed: Seed for both per-class sampling and the final shuffle.
        data_path: Optional dataset override (tests use the small fixture).
            Defaults to ``Settings.data_path``.

    Returns:
        A list of exactly ``n`` text strings (classes as evenly split as
        ``n`` allows; the remainder is topped up from the first class with
        rows).

    Raises:
        ValueError: If ``n <= 0`` or the dataset has zero rows for every
            class.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    df = load_dataset(data_path)
    rng = np.random.default_rng(seed)
    labels = sorted(VALID_LABELS)
    per_class = n // len(labels)

    texts: list[str] = []
    for label in labels:
        pool = df.loc[df["label"] == label, "text"].astype(str).tolist()
        if not pool:
            continue
        take = per_class if per_class > 0 else 1
        indices = rng.integers(0, len(pool), size=take)
        texts.extend(pool[int(i)] for i in indices)

    if not texts:
        raise ValueError("dataset has no rows for any of the known labels")

    # Top up to exactly n (integer division above can leave n mod 3 short).
    while len(texts) < n:
        pool_index = rng.integers(0, len(texts))
        texts.append(texts[int(pool_index)])
    texts = texts[:n]

    rng.shuffle(texts)
    return texts


def wait_for_health(base_url: str, timeout_s: float = 30.0, interval_s: float = 1.0) -> bool:
    """Poll ``GET {base_url}/health`` until it answers 200, or ``timeout_s`` elapses.

    Returns:
        ``True`` once healthy, ``False`` if ``timeout_s`` elapsed first.
    """
    deadline = time.monotonic() + timeout_s
    url = base_url.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=5.0)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval_s)
    return False


def generate_load(
    base_url: str,
    texts: Sequence[str],
    timeout_s: float = 10.0,
    delay_ms: float = 0.0,
) -> LoadResult:
    """Fire one ``POST /predict`` per text in ``texts`` against ``base_url``.

    A failed request (network error or non-2xx status) is logged and
    counted, never raised -- one bad request must not abort the whole load
    run, since the point is to generate traffic, not to assert correctness
    (that is ``tests/serving/test_api.py``'s job).

    Args:
        base_url: Base URL of the running API (e.g. ``http://127.0.0.1:8000``).
        texts: Request bodies to send, one per request, in order.
        timeout_s: Per-request HTTP timeout.
        delay_ms: Optional pause between requests, to spread load over a
            longer wall-clock window (e.g. for a nicer-looking time range
            on the dashboard). ``0`` (default) sends as fast as possible.

    Returns:
        A :class:`LoadResult` with the request/success/failure counts.
    """
    url = base_url.rstrip("/") + "/predict"
    session = requests.Session()

    ok = 0
    failed = 0
    latencies_ms: list[float] = []
    wall_start = time.perf_counter()

    for text in texts:
        start = time.perf_counter()
        try:
            response = session.post(url, json={"text": text}, timeout=timeout_s)
            response.raise_for_status()
            ok += 1
        except requests.RequestException as exc:
            failed += 1
            logger.warning("request failed: %s", exc)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    wall_elapsed_s = time.perf_counter() - wall_start
    mean_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    return LoadResult(
        requested=len(texts),
        ok=ok,
        failed=failed,
        mean_latency_ms=mean_latency_ms,
        wall_elapsed_s=wall_elapsed_s,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="Total requests to send.")
    parser.add_argument("--url", type=str, default=DEFAULT_BASE_URL, help="Base URL of the API.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the text mix.")
    parser.add_argument("--data-path", type=Path, default=None, help="Override Settings.data_path.")
    parser.add_argument(
        "--delay-ms", type=float, default=0.0, help="Pause between requests, in milliseconds."
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Per-request timeout, in seconds."
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for GET /health before giving up.",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Do not poll GET /health first; fire requests immediately.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint: wait for the API, then fire the load, then print a summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    if not args.skip_wait:
        logger.info("waiting for %s/health (timeout=%.0fs)...", args.url, args.wait_timeout)
        if not wait_for_health(args.url, timeout_s=args.wait_timeout):
            logger.error(
                "%s/health did not answer 200 within %.0fs; is the stack up? "
                "(`docker compose up -d`)",
                args.url,
                args.wait_timeout,
            )
            return 1

    texts = build_text_mix(args.n, seed=args.seed, data_path=args.data_path)
    logger.info(
        "sending %d requests to %s/predict (mix of the 3 urgency classes)...", len(texts), args.url
    )

    result = generate_load(args.url, texts, timeout_s=args.timeout, delay_ms=args.delay_ms)

    logger.info(
        "done: %d/%d ok, %d failed, mean=%.1fms, %.1f req/s over %.1fs -- "
        "open Grafana (http://localhost:3000) to see the dashboard update",
        result.ok,
        result.requested,
        result.failed,
        result.mean_latency_ms,
        result.throughput_rps,
        result.wall_elapsed_s,
    )

    if result.ok == 0:
        logger.error("every request failed; nothing to see on the dashboard")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
