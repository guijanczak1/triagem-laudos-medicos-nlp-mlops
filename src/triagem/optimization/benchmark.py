"""Formal baseline (sklearn) x ONNX latency benchmark + equivalence report (T21).

This is the project's **official** ``sklearn`` vs ``onnx`` comparison,
distinct in scope from two earlier deliverables it builds on without
repeating their work:

- ``scripts/measure_latency.py`` (T12) measures the ``sklearn`` backend
  alone against two *targets* (in-process vs. end-to-end HTTP through the
  container) -- it never loads the ``onnx`` backend.
- ``triagem.optimization.onnx_export`` (T20) exports ``models/model.onnx``
  and, as a side check gating its own ``--quantize`` flag, already measured
  sklearn/onnx prediction equivalence on the full real test split (2,888
  rows) at **99.5152%** and found dynamic int8 quantization does not help
  on this model (see ``models/model_onnx_meta.json``:
  ``quantization_check``). Those numbers are not re-derived here; the
  equivalence check in this module reuses the exact same holdout split
  (``split_dataset(df, test_size=0.2, seed=42)`` on
  ``data/processed/laudos.csv`` -- the same split ``triagem.training.train``
  computes its reported metrics on) purely to *validate* T20's finding
  against this task's own artifact, formally, as part of this report.

:func:`benchmark` measures both backends through the real, public
:class:`~triagem.serving.predictor.Predictor` API (never a raw
``onnxruntime``/``sklearn`` call bypassing it), on the same hardware, the
same deterministic input sample and the same number of executions, with a
declared warm-up -- see ``harness/latency.md``: "nunca reportar ganho sem
dizer hardware, execucoes, batch, warm-up".

**Honesty rule (hard, non-negotiable):** if the ONNX p50 does not beat the
sklearn p50 by at least ``thresholds.p50_reduction_min_pct`` (20%, see
``state/backlog.json``), this module reports the real percentage anyway --
it never manufactures a passing number. A well-measured negative result is
a valid deliverable; a manufactured "win" is not.

Usage::

    python -m triagem.optimization.benchmark

Writes ``docs/benchmarks/comparison.json`` (raw numbers) and
``docs/latency_report.md`` (human-readable report).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from triagem.config import get_settings
from triagem.data.loaders import load_dataset, split_dataset
from triagem.optimization.onnx_export import EQUIVALENCE_MIN_PCT
from triagem.serving.predictor import Predictor

logger = logging.getLogger(__name__)

#: src/triagem/optimization/benchmark.py -> project root (four levels up).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_JSON_OUT = PROJECT_ROOT / "docs" / "benchmarks" / "comparison.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "docs" / "latency_report.md"

#: ``state/backlog.json``: ``thresholds.p50_reduction_min_pct`` -- the target
#: fraction the ONNX backend's p50 must be *below* the sklearn baseline's.
#: Never relaxed to force a pass; see the honesty rule in the module docstring.
P50_REDUCTION_MIN_PCT = 20.0

#: Fraction of the dataset the equivalence check (and the latency sample
#: pool) is drawn from -- must match ``triagem.training.train``'s holdout
#: split so this task validates against the exact set T7/T20 already used.
HOLDOUT_TEST_SIZE = 0.2


@dataclass(frozen=True)
class BackendLatency:
    """Percentile/throughput summary of one backend's measured run.

    Attributes:
        backend: ``"sklearn"`` or ``"onnx"``.
        n: Number of measured (post-warm-up) single-text ``predict()`` calls.
        warmup: Number of untimed calls made before measurement started.
        seed: Seed used to draw the input sample.
        mean_ms / stdev_ms / min_ms / p50_ms / p90_ms / p95_ms / p99_ms /
            max_ms: Per-call latency statistics, in milliseconds.
        throughput_rps: ``n / wall_clock_seconds`` for the measured phase.
    """

    backend: str
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

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable representation of this result."""
        return asdict(self)


@dataclass(frozen=True)
class EquivalenceResult:
    """sklearn vs onnx prediction agreement on the project's holdout test split.

    Attributes:
        n_compared: Number of holdout rows compared (via ``predict_batch``,
            one label per row, from each backend).
        n_matches: How many of those rows got the identical predicted label
            from both backends.
        equivalence_pct: ``n_matches / n_compared * 100``.
        equivalence_min_pct: The required floor
            (``state/backlog.json``: ``thresholds.prediction_equivalence_min_pct``).
        passed: Whether ``equivalence_pct >= equivalence_min_pct``.
    """

    n_compared: int
    n_matches: int
    equivalence_pct: float
    equivalence_min_pct: float
    passed: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        """Return a JSON-serializable representation of this result."""
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkResult:
    """Full output of :func:`benchmark`: per-backend latency + equivalence + verdict.

    Attributes:
        backends: ``backend name -> BackendLatency``, in measurement order.
        equivalence: The sklearn/onnx prediction-agreement check.
        p50_reduction_pct: ``(sklearn.p50 - onnx.p50) / sklearn.p50 * 100``.
            ``None`` when either backend was not measured (e.g. ``backends``
            did not include both) or the sklearn p50 is zero.
        p50_reduction_min_pct: The required floor (see
            :data:`P50_REDUCTION_MIN_PCT`).
        p50_reduction_met: Whether the real ``p50_reduction_pct`` meets the
            floor -- ``False`` (with the real number still reported) is a
            valid, honestly-measured outcome, not an error.
        seed: Seed shared by the latency sample and the equivalence check.
        measured_at: ISO timestamp of when the benchmark ran.
    """

    backends: dict[str, BackendLatency]
    equivalence: EquivalenceResult
    p50_reduction_pct: float | None
    p50_reduction_min_pct: float
    p50_reduction_met: bool
    seed: int
    measured_at: str


def _sample_with_replacement(pool: Sequence[str], n: int, seed: int) -> list[str]:
    """Deterministically draw ``n`` items from ``pool`` (with replacement, seeded)."""
    if not pool:
        raise ValueError("cannot sample from an empty pool")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pool), size=n)
    return [pool[int(i)] for i in indices]


def _measure_backend_latency(
    predictor: Predictor,
    warmup_texts: Sequence[str],
    measured_texts: Sequence[str],
    seed: int,
) -> BackendLatency:
    """Time ``len(measured_texts)`` single-item ``predict()`` calls, after warm-up.

    Mirrors ``scripts/measure_latency.py``'s own warm-up-then-measure shape
    (untimed warm-up calls discarded, individual calls timed with
    ``time.perf_counter``), applied here to the ``Predictor`` directly so
    both backends are compared through the identical call path.
    """
    for text in warmup_texts:
        predictor.predict(text)

    durations_ms: list[float] = []
    wall_start = time.perf_counter()
    for text in measured_texts:
        start = time.perf_counter()
        predictor.predict(text)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    wall_elapsed_s = time.perf_counter() - wall_start

    p50, p90, p95, p99 = (float(p) for p in np.percentile(durations_ms, [50, 90, 95, 99]))
    return BackendLatency(
        backend=predictor.metadata.backend,
        n=len(measured_texts),
        warmup=len(warmup_texts),
        seed=seed,
        mean_ms=statistics.fmean(durations_ms),
        stdev_ms=statistics.pstdev(durations_ms),
        min_ms=min(durations_ms),
        p50_ms=p50,
        p90_ms=p90,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=max(durations_ms),
        throughput_rps=(
            (len(durations_ms) / wall_elapsed_s) if wall_elapsed_s > 0 else float("inf")
        ),
    )


def measure_equivalence(
    predictors: dict[str, Predictor],
    texts: Sequence[str],
    equivalence_min_pct: float = EQUIVALENCE_MIN_PCT,
) -> EquivalenceResult:
    """Compare sklearn vs onnx predicted labels on ``texts`` through the public API.

    Args:
        predictors: Must contain ``"sklearn"`` and ``"onnx"`` keys to
            actually compare anything; with either (or both) missing this
            returns a vacuous, always-``passed`` result (``n_compared=0``)
            -- callers that request a partial ``backends`` tuple in
            :func:`benchmark` get a benchmark that still runs, just without
            an equivalence verdict.
        texts: Holdout texts to compare on (see :data:`HOLDOUT_TEST_SIZE`).
        equivalence_min_pct: Required floor. Defaults to the project's
            official threshold (:data:`triagem.optimization.onnx_export.EQUIVALENCE_MIN_PCT`,
            99.0 -- ``state/backlog.json``: ``thresholds.prediction_equivalence_min_pct``).

    Returns:
        The :class:`EquivalenceResult`.
    """
    if "sklearn" not in predictors or "onnx" not in predictors or not texts:
        return EquivalenceResult(
            n_compared=0,
            n_matches=0,
            equivalence_pct=100.0,
            equivalence_min_pct=equivalence_min_pct,
            passed=True,
        )

    sklearn_labels = [p.label for p in predictors["sklearn"].predict_batch(texts)]
    onnx_labels = [p.label for p in predictors["onnx"].predict_batch(texts)]
    matches = sum(1 for a, b in zip(sklearn_labels, onnx_labels, strict=True) if a == b)
    equivalence_pct = matches / len(texts) * 100.0

    return EquivalenceResult(
        n_compared=len(texts),
        n_matches=matches,
        equivalence_pct=equivalence_pct,
        equivalence_min_pct=equivalence_min_pct,
        passed=equivalence_pct >= equivalence_min_pct,
    )


def benchmark(
    backends: Sequence[str] = ("sklearn", "onnx"),
    n: int = 200,
    warmup: int = 20,
    seed: int = 42,
    models_dir: Path | None = None,
    data_path: Path | None = None,
) -> BenchmarkResult:
    """Run the formal sklearn x onnx latency + equivalence benchmark.

    Args:
        backends: Backends to load and measure, in order. Defaults to both
            -- the equivalence check and ``p50_reduction_pct`` only compute
            when both ``"sklearn"`` and ``"onnx"`` are present.
        n: Measured ``predict()`` calls per backend (default 200).
        warmup: Untimed warm-up calls per backend before measurement starts
            (default 20) -- discarded so first-call overhead (vectorizer
            vocabulary lookup / onnxruntime graph warm state) never skews
            the reported numbers.
        seed: Seed for the deterministic input sample and the holdout
            split -- the same seed always reproduces the same measured
            workload and the same equivalence comparison set.
        models_dir: Directory holding ``model.joblib`` / ``model.onnx``.
            Defaults to ``Settings.models_dir``.
        data_path: Processed dataset CSV. Defaults to ``Settings.data_path``.

    Returns:
        The full :class:`BenchmarkResult`.

    Raises:
        ValueError: ``n <= 0`` or ``warmup < 0``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    settings = get_settings()
    resolved_models_dir = models_dir if models_dir is not None else settings.models_dir
    resolved_data_path = data_path if data_path is not None else settings.data_path

    df = load_dataset(resolved_data_path)
    _, test_df = split_dataset(df, test_size=HOLDOUT_TEST_SIZE, seed=seed)
    test_texts = test_df["text"].astype(str).tolist()

    sample = _sample_with_replacement(test_texts, n + warmup, seed=seed)
    warmup_texts, measured_texts = sample[:warmup], sample[warmup:]

    predictors: dict[str, Predictor] = {}
    backend_results: dict[str, BackendLatency] = {}
    for backend in backends:
        predictor = Predictor.load(backend=backend, models_dir=resolved_models_dir)  # type: ignore[arg-type]
        predictors[backend] = predictor
        logger.info("measuring backend=%s (n=%d, warmup=%d)", backend, n, warmup)
        backend_results[backend] = _measure_backend_latency(
            predictor, warmup_texts, measured_texts, seed
        )

    equivalence = measure_equivalence(predictors, test_texts)

    p50_reduction_pct: float | None = None
    p50_reduction_met = False
    sklearn_result = backend_results.get("sklearn")
    onnx_result = backend_results.get("onnx")
    if sklearn_result is not None and onnx_result is not None and sklearn_result.p50_ms > 0:
        p50_reduction_pct = (
            (sklearn_result.p50_ms - onnx_result.p50_ms) / sklearn_result.p50_ms * 100.0
        )
        p50_reduction_met = p50_reduction_pct >= P50_REDUCTION_MIN_PCT

    return BenchmarkResult(
        backends=backend_results,
        equivalence=equivalence,
        p50_reduction_pct=p50_reduction_pct,
        p50_reduction_min_pct=P50_REDUCTION_MIN_PCT,
        p50_reduction_met=p50_reduction_met,
        seed=seed,
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
        "onnxruntime_version": _package_version("onnxruntime"),
        "skl2onnx_version": _package_version("skl2onnx"),
        "numpy_version": _package_version("numpy"),
    }


def _result_to_payload(result: BenchmarkResult, machine: dict[str, str]) -> dict[str, object]:
    """Build the JSON-serializable payload written to ``docs/benchmarks/comparison.json``."""
    return {
        "generated_at": result.measured_at,
        "machine": machine,
        "seed": result.seed,
        "holdout_test_size": HOLDOUT_TEST_SIZE,
        "backends": {name: r.as_dict() for name, r in result.backends.items()},
        "equivalence": result.equivalence.as_dict(),
        "p50_reduction_pct": result.p50_reduction_pct,
        "p50_reduction_min_pct": result.p50_reduction_min_pct,
        "p50_reduction_met": result.p50_reduction_met,
    }


def _render_markdown(payload: dict[str, object]) -> str:
    """Render ``payload`` (see :func:`_result_to_payload`) as ``docs/latency_report.md``."""
    machine = payload["machine"]
    assert isinstance(machine, dict)
    backends = payload["backends"]
    assert isinstance(backends, dict)
    equivalence = payload["equivalence"]
    assert isinstance(equivalence, dict)

    p50_reduction_pct = payload["p50_reduction_pct"]
    p50_reduction_min_pct = payload["p50_reduction_min_pct"]
    assert isinstance(p50_reduction_min_pct, float)
    p50_reduction_met = payload["p50_reduction_met"]

    lines = [
        "# Relatorio de latencia: baseline (sklearn) x ONNX",
        "",
        "Comparacao formal de latencia entre o backend `sklearn` (baseline, T7) e o "
        "backend `onnx` (T20) do `Predictor` (T9), no mesmo hardware, com a mesma amostra "
        "de entrada e a mesma quantidade de execucoes, mais o teste de equivalencia de "
        "predicoes entre os dois backends no split de teste do projeto (T21).",
        "",
        "## Metodologia",
        "",
        f"- Maquina: {machine['platform']} ({machine['processor']}, "
        f"{machine['cpu_count']} CPUs)",
        f"- Python {machine['python_version']} | scikit-learn {machine['scikit_learn_version']} "
        f"| onnxruntime {machine['onnxruntime_version']} | skl2onnx {machine['skl2onnx_version']} "
        f"| numpy {machine['numpy_version']}",
        f"- Gerado em: {payload['generated_at']}",
        f"- Seed: {payload['seed']} (amostra de latencia e split de holdout deterministicos)",
        f"- Split de holdout: `split_dataset(test_size={payload['holdout_test_size']}, "
        f"seed={payload['seed']})` sobre `data/processed/laudos.csv` -- o mesmo split que "
        "`triagem.training.train` usa para reportar `metrics.json` (T7) e que T20 usou para "
        "medir equivalencia (99.5152% em 2.888 linhas -- ver `models/model_onnx_meta.json`)",
        "- Warm-up declarado por backend (descartado da medicao) + chamadas medidas "
        "individualmente via `Predictor.predict()` (mesmo caminho publico usado pela API)",
        "",
        "## Latencia por backend (percentis em milissegundos)",
        "",
        "| Backend | n | warm-up | media | p50 | p90 | p95 | p99 | throughput (req/s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in backends.items():
        assert isinstance(r, dict)
        lines.append(
            f"| {name} | {r['n']} | {r['warmup']} | {r['mean_ms']:.4f} | "
            f"{r['p50_ms']:.4f} | {r['p90_ms']:.4f} | {r['p95_ms']:.4f} | {r['p99_ms']:.4f} | "
            f"{r['throughput_rps']:.1f} |"
        )
    lines.append("")

    lines += [
        "## Reducao de p50 (onnx sobre sklearn)",
        "",
    ]
    if p50_reduction_pct is None:
        lines.append(
            "- Nao calculado nesta execucao (requer medir os dois backends `sklearn` e `onnx`)."
        )
    else:
        verdict = "ATINGIDA" if p50_reduction_met else "NAO ATINGIDA"
        lines += [
            f"- Reducao medida: **{p50_reduction_pct:.2f}%** "
            f"(meta: >= {p50_reduction_min_pct:.0f}%) -- **{verdict}**.",
        ]
        if not p50_reduction_met:
            lines.append(
                "- Numero real reportado sem maquiagem (regra dura de "
                "`harness/latency.md`): para este modelo (TF-IDF + LogisticRegression, "
                "pequeno), o overhead fixo do backend pode dominar sobre o ganho do grafo "
                "ONNX -- ver a leitura honesta abaixo."
            )
    lines.append("")

    eq_verdict = "ATINGIDA" if equivalence["passed"] else "NAO ATINGIDA"
    lines += [
        "## Equivalencia de predicoes sklearn x onnx",
        "",
        f"- Comparadas {equivalence['n_compared']} linhas do split de teste (holdout do "
        f"seed acima): {equivalence['n_matches']} predicoes identicas.",
        f"- Equivalencia: **{equivalence['equivalence_pct']:.4f}%** "
        f"(meta: >= {equivalence['equivalence_min_pct']:.0f}%) -- **{eq_verdict}**.",
        "- Esta e a mesma verificacao que T20 ja havia rodado sobre o split de teste "
        "completo (99.5152% em 2.888 linhas); o numero acima e medido de novo aqui, contra "
        "o artefato oficial deste projeto, como parte do entregavel formal desta tarefa.",
        "",
        "## Leitura honesta",
        "",
        "- Reportamos percentis (p50/p90/p95/p99), nao so a media: a media esconde a "
        "cauda, que e o que doi em producao.",
        "- Quantizacao dinamica int8 foi avaliada em T20 e **descartada** -- nao trouxe "
        "ganho real de p50 neste modelo (ver `models/model_onnx_meta.json`: "
        "`quantization_check`). Este relatorio nao reaplica essa tentativa.",
        "- Nenhum numero acima foi ajustado para bater a meta; se a reducao de p50 nao "
        "atingiu o alvo, isso esta reportado explicitamente acima, nao omitido.",
        "- Numeros brutos desta execucao (JSON): `docs/benchmarks/comparison.json`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Measured calls per backend.")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed warm-up calls per backend.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the input sample/split.")
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Override Settings.models_dir."
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Override Settings.data_path.")
    parser.add_argument(
        "--backends",
        type=str,
        default="sklearn,onnx",
        help="Comma-separated backends to measure (default: sklearn,onnx).",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint: run the benchmark and write the report.

    Returns:
        ``0`` normally. ``1`` if the sklearn/onnx equivalence check fails
        the project's floor (``state/backlog.json``:
        ``thresholds.prediction_equivalence_min_pct``) -- a missed p50
        reduction target never fails the CLI, it is only reported honestly.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())

    result = benchmark(
        backends=backends,
        n=args.n,
        warmup=args.warmup,
        seed=args.seed,
        models_dir=args.models_dir,
        data_path=args.data_path,
    )

    machine = _machine_info()
    payload = _result_to_payload(result, machine)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.json_out)

    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(_render_markdown(payload), encoding="utf-8")
    logger.info("wrote %s", args.md_out)

    for name, r in result.backends.items():
        print(f"{name}: p50={r.p50_ms:.4f}ms p95={r.p95_ms:.4f}ms n={r.n} warmup={r.warmup}")
    if result.p50_reduction_pct is not None:
        verdict = "MET" if result.p50_reduction_met else "NOT MET"
        print(
            f"p50 reduction (onnx over sklearn): {result.p50_reduction_pct:.2f}% "
            f"(target >= {result.p50_reduction_min_pct:.0f}%) -- {verdict}"
        )
    print(
        f"equivalence: {result.equivalence.equivalence_pct:.4f}% "
        f"(target >= {result.equivalence.equivalence_min_pct:.0f}%) "
        f"n_compared={result.equivalence.n_compared}"
    )

    if not result.equivalence.passed:
        print("EQUIVALENCE GATE FAILED -- see docs/latency_report.md")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
