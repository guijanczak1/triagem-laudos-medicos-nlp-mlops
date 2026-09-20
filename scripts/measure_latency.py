"""Reproducible latency baseline for the triage inference service (T12).

Measures wall-clock inference latency (p50/p90/p95/p99 + throughput) of two
targets, on the same hardware, the same deterministic input sample and the
same number of executions:

(a) **in-process** -- calls the ``sklearn`` backend of
    :class:`triagem.serving.predictor.Predictor` directly, no network hop.
(b) **http** -- end-to-end ``POST /predict`` against a running container
    (``docker run -p 8000:8000 triagem-api:local``), including HTTP/ASGI
    overhead.

Every run does a declared **warm-up** (untimed calls -- the first inference
allocates memory / builds internal caches and would otherwise skew the
numbers) and draws its input sample with a **fixed seed**, so re-running the
script reproduces the same measured workload.

Usage::

    # in-process only (no Docker required -- this is what the unit test uses)
    python scripts/measure_latency.py --skip-http

    # in-process + HTTP against an already-running container
    docker run -d --rm -p 8000:8000 --name triagem-bench triagem-api:local
    python scripts/measure_latency.py
    docker stop triagem-bench

Writes ``docs/benchmarks/baseline.json`` (raw numbers) and
``docs/latency_baseline.md`` (human-readable report) by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from triagem.data.loaders import load_dataset
from triagem.serving.predictor import Predictor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HTTP_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_JSON_OUT = PROJECT_ROOT / "docs" / "benchmarks" / "baseline.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "docs" / "latency_baseline.md"


@dataclass(frozen=True)
class Target:
    """A single latency-measurement target: a name plus a one-text call.

    Attributes:
        name: Short identifier used in reports/JSON (e.g. ``"in_process_sklearn"``).
        call: Invokes the target for one input text. Raises on failure;
            :func:`measure_latency` does not swallow errors -- a broken
            target must fail the benchmark, not silently skew the numbers.
    """

    name: str
    call: Callable[[str], None]


@dataclass(frozen=True)
class LatencyResult:
    """Percentile/throughput summary of one :func:`measure_latency` run."""

    target: str
    n: int
    warmup: int
    seed: int
    mean_ms: float
    stdev_ms: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    throughput_rps: float
    measured_at: str

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable representation of this result."""
        return asdict(self)


def sample_texts(n: int, seed: int = 42, data_path: Path | None = None) -> list[str]:
    """Deterministically sample ``n`` input texts for the benchmark.

    Args:
        n: Number of texts to draw.
        seed: Seed for the sampling RNG -- the same ``seed`` (with the same
            dataset) always yields the same texts, in the same order, so
            benchmark runs are comparable across backends/targets.
        data_path: Optional override for the processed dataset CSV.
            Defaults to ``Settings.data_path`` (see ``load_dataset``).
            Tests pass the small fixture dataset instead of the full corpus.

    Returns:
        A list of ``n`` text strings, sampled with replacement from the
        dataset (cycling the corpus if ``n`` exceeds its row count).

    Raises:
        ValueError: If the resolved dataset has zero rows.
    """
    df = load_dataset(data_path)
    pool = df["text"].astype(str).tolist()
    if not pool:
        raise ValueError("dataset has no rows to sample texts from")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pool), size=n)
    return [pool[int(i)] for i in indices]


def in_process_sklearn_target(models_dir: Path | None = None) -> Target:
    """Target (a): in-process inference via the ``sklearn`` Predictor (T9)."""
    predictor = Predictor.load(backend="sklearn", models_dir=models_dir)

    def _call(text: str) -> None:
        predictor.predict(text)

    return Target(name="in_process_sklearn", call=_call)


def http_target(base_url: str = DEFAULT_HTTP_BASE_URL, timeout: float = 10.0) -> Target:
    """Target (b): end-to-end ``POST /predict`` against a running container."""
    import requests

    url = base_url.rstrip("/") + "/predict"
    session = requests.Session()

    def _call(text: str) -> None:
        response = session.post(url, json={"text": text}, timeout=timeout)
        response.raise_for_status()

    return Target(name="http_container", call=_call)


def http_target_reachable(base_url: str = DEFAULT_HTTP_BASE_URL, timeout: float = 3.0) -> bool:
    """Return ``True`` if ``GET {base_url}/health`` answers 200 within ``timeout``s."""
    import requests

    try:
        response = requests.get(base_url.rstrip("/") + "/health", timeout=timeout)
    except requests.RequestException:
        return False
    return response.status_code == 200


def measure_latency(
    target: Target,
    n: int = 200,
    warmup: int = 20,
    seed: int = 42,
    data_path: Path | None = None,
) -> LatencyResult:
    """Measure p50/p90/p95/p99 latency and throughput of ``target``.

    Warms up ``target`` with ``warmup`` untimed calls, then times ``n``
    further individual calls. Both phases draw from the same deterministic,
    seeded input sample (see :func:`sample_texts`).

    Args:
        target: The target to measure (see :func:`in_process_sklearn_target`
            / :func:`http_target`).
        n: Number of measured calls (default 200).
        warmup: Number of untimed warm-up calls before measurement starts
            (default 20) -- discarded so first-call overhead (graph/model
            load, connection setup) never pollutes the reported numbers.
        seed: Seed for the deterministic input sample.
        data_path: Optional dataset override, forwarded to
            :func:`sample_texts` (tests use a small fixture).

    Returns:
        A :class:`LatencyResult` with the full percentile/throughput table.

    Raises:
        ValueError: If ``n <= 0`` or ``warmup < 0``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    texts = sample_texts(n + warmup, seed=seed, data_path=data_path)
    warmup_texts, measured_texts = texts[:warmup], texts[warmup:]

    for text in warmup_texts:
        target.call(text)

    durations_ms: list[float] = []
    wall_start = time.perf_counter()
    for text in measured_texts:
        start = time.perf_counter()
        target.call(text)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    wall_elapsed_s = time.perf_counter() - wall_start

    p50, p90, p95, p99 = (float(p) for p in np.percentile(durations_ms, [50, 90, 95, 99]))
    return LatencyResult(
        target=target.name,
        n=n,
        warmup=warmup,
        seed=seed,
        mean_ms=statistics.fmean(durations_ms),
        stdev_ms=statistics.pstdev(durations_ms),
        min_ms=min(durations_ms),
        p50_ms=p50,
        p90_ms=p90,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=max(durations_ms),
        throughput_rps=(len(durations_ms) / wall_elapsed_s) if wall_elapsed_s > 0 else float("inf"),
        measured_at=datetime.now(UTC).isoformat(),
    )


def _package_version(name: str) -> str:
    """Return the installed version of ``name``, or ``"not installed"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _machine_info() -> dict[str, str]:
    """Describe the machine/software the benchmark ran on (required by the report)."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": str(os.cpu_count()),
        "scikit_learn_version": _package_version("scikit-learn"),
        "fastapi_version": _package_version("fastapi"),
        "numpy_version": _package_version("numpy"),
    }


def _render_markdown(payload: dict[str, object]) -> str:
    """Render ``payload`` (see :func:`main`) as the ``docs/latency_baseline.md`` report."""
    machine = payload["machine"]
    assert isinstance(machine, dict)
    results = payload["results"]
    assert isinstance(results, list)

    lines = [
        "# Baseline de latencia local em container",
        "",
        "Baseline de latencia (PDF Etapa 1, tarefa T12): mede a inferencia in-process do",
        "backend `sklearn` (T9) e a chamada HTTP ponta a ponta contra o container",
        "`triagem-api:local` (T11), no mesmo hardware, com a mesma amostra de entrada e a",
        "mesma quantidade de execucoes, com warm-up declarado.",
        "",
        "## Metodologia",
        "",
        f"- Maquina: {machine['platform']} ({machine['processor']}, "
        f"{machine['cpu_count']} CPUs)",
        f"- Python {machine['python_version']} | scikit-learn {machine['scikit_learn_version']} "
        f"| fastapi {machine['fastapi_version']} | numpy {machine['numpy_version']}",
        f"- Gerado em: {payload['generated_at']}",
    ]
    if results:
        r0 = results[0]
        assert isinstance(r0, dict)
        lines += [
            f"- n = {r0['n']} execucoes medidas por alvo, warm-up = {r0['warmup']} chamadas "
            "descartadas antes da medicao",
            f"- Amostra de entrada com seed fixa ({r0['seed']}), textos do dataset processado "
            "(`data/processed/laudos.csv`)",
        ]
    lines.append("")

    skip_reason = payload.get("http_target_skipped_reason")
    if skip_reason:
        lines += [
            f"> Alvo HTTP end-to-end **nao medido** nesta execucao: {skip_reason}. Suba o "
            "container (`docker run -d --rm -p 8000:8000 triagem-api:local`) e rode "
            "`python scripts/measure_latency.py` novamente para completar a tabela.",
            "",
        ]

    lines += [
        "## Resultados (percentis em milissegundos)",
        "",
        "| Alvo | n | warm-up | media | p50 | p90 | p95 | p99 | throughput (req/s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        assert isinstance(r, dict)
        lines.append(
            f"| {r['target']} | {r['n']} | {r['warmup']} | {r['mean_ms']:.2f} | "
            f"{r['p50_ms']:.2f} | {r['p90_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | "
            f"{r['throughput_rps']:.1f} |"
        )

    lines += [
        "",
        "## Leitura honesta",
        "",
        "- O alvo p95 end-to-end < 100 ms (PDF) e **informativo, nao bloqueante** -- o numero "
        "acima e reportado como medido, mesmo se estiver acima do alvo.",
        "- Reportamos percentis (p50/p90/p95/p99), nao so a media: a media esconde a cauda, "
        "que e o que doi em producao.",
        "- Numeros brutos desta execucao (JSON): `docs/benchmarks/baseline.json`.",
        "- Este e o baseline (sklearn) da Etapa 1; a comparacao contra o backend ONNX "
        "otimizado (T20/T21) fica em `docs/latency_report.md`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Measured calls per target.")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed warm-up calls per target.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the input text sample.")
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Override Settings.models_dir."
    )
    parser.add_argument(
        "--http-url",
        type=str,
        default=DEFAULT_HTTP_BASE_URL,
        help="Base URL of a running container to also benchmark end-to-end.",
    )
    parser.add_argument(
        "--skip-http",
        action="store_true",
        help="Only measure the in-process target; never attempt the HTTP target.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint: measure both targets (HTTP best-effort) and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    results: list[LatencyResult] = []

    logger.info(
        "measuring target (a): in-process sklearn predictor (n=%d, warmup=%d)", args.n, args.warmup
    )
    in_process = in_process_sklearn_target(models_dir=args.models_dir)
    results.append(measure_latency(in_process, n=args.n, warmup=args.warmup, seed=args.seed))

    http_skipped_reason: str | None = None
    if args.skip_http:
        http_skipped_reason = "--skip-http passed"
    elif not http_target_reachable(args.http_url):
        http_skipped_reason = f"{args.http_url}/health not reachable"
    else:
        logger.info("measuring target (b): end-to-end HTTP against %s", args.http_url)
        results.append(
            measure_latency(
                http_target(args.http_url), n=args.n, warmup=args.warmup, seed=args.seed
            )
        )

    if http_skipped_reason:
        logger.warning("skipping HTTP end-to-end target: %s", http_skipped_reason)

    payload: dict[str, object] = {
        "machine": _machine_info(),
        "generated_at": datetime.now(UTC).isoformat(),
        "http_target_skipped_reason": http_skipped_reason,
        "results": [r.as_dict() for r in results],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.json_out)

    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(_render_markdown(payload), encoding="utf-8")
    logger.info("wrote %s", args.md_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
