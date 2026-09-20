"""Training/retraining DAG: download -> build dataset -> train -> evaluate -> publish.

Thin orchestration only: every task below calls real ``triagem`` package
functions (``triagem.data.download.download_raw`` / T3,
``triagem.data.build.build_dataset`` / T4,
``triagem.training.train.train`` / T7) -- no training or evaluation logic is
duplicated in this module. ``train()`` already computes the holdout metrics
and checks them against the project's ``QualityGate``
(``triagem.training.evaluate``); ``evaluate_model`` below only reads that
already-computed result from XCom and fails the DAG run if the gate did not
pass, so a model that misses the target is never silently promoted to
``publish_artifacts``.

apache-airflow is intentionally NOT a native Windows dependency (pwd/fcntl
do not exist there) -- it lives in the optional Poetry group ``airflow``
(see ``pyproject.toml``) and only actually runs inside the Airflow Docker
image (``docker/Dockerfile.airflow``, T15) or in CI on an Ubuntu runner.
This module therefore can only be *imported* where apache-airflow is
installed; ``tests/dags/test_dag_integrity.py`` guards against that with
``pytest.importorskip("airflow")`` so the suite still passes (skipped, not
failed) on a native Windows dev machine.

Paths and the training seed are resolved from ``triagem.config.Settings``
(itself overridable via ``TRIAGEM_*`` environment variables -- see
``docker-compose.yml``'s ``airflow`` profile, T15) plus two optional Airflow
Variables (``triagem_seed``, ``triagem_force_download``); nothing is
hardcoded to a local machine path.
"""

from __future__ import annotations

import logging
from typing import Any

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DAG_ID = "train_triagem"

#: Fixed start date -- never datetime.now(), which would shift the
#: schedule's anchor (and Airflow's catchup/backfill window) on every parse.
START_DATE = pendulum.datetime(2024, 1, 1, tz="UTC")

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "triagem",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}

#: Artifact keys pushed to XCom by ``train_model`` and consumed by
#: ``evaluate_model`` / ``publish_artifacts``.
_ARTIFACT_KEYS = ("model_path", "metrics_path", "label_encoder_path", "confusion_matrix_path")


def _seed() -> int:
    """Training/build seed, overridable via the ``triagem_seed`` Airflow Variable."""
    return int(Variable.get("triagem_seed", default_var="42"))


def _download_dataset(**_: Any) -> None:
    """Task: download the raw Medical Abstracts TC Corpus (T3).

    Heavy imports and all I/O happen inside the task body, never at module
    import time, so the scheduler's repeated DAG-file parsing stays cheap.
    """
    from triagem.data.download import download_raw

    force = Variable.get("triagem_force_download", default_var="false").strip().lower() == "true"
    paths = download_raw(force=force)
    logger.info("download_dataset: wrote %s", {name: str(path) for name, path in paths.items()})


def _build_dataset(**_: Any) -> None:
    """Task: build the processed, urgency-labeled dataset (T4)."""
    from triagem.config import get_settings
    from triagem.data.build import build_dataset

    settings = get_settings()
    out_path = build_dataset(raw_dir=settings.raw_dir, out_path=settings.data_path, seed=_seed())
    logger.info("build_dataset: wrote %s", out_path)


def _train_model(ti: Any, **_: Any) -> None:
    """Task: train, evaluate on the holdout, and persist artifacts (T7).

    Pushes the JSON-serializable subset of ``TrainResult`` to XCom --
    including the gate outcome already computed inside ``train()`` -- so
    downstream tasks never need to re-run or duplicate the evaluation.
    """
    from triagem.training.train import TrainConfig, train

    result = train(TrainConfig(seed=_seed()))
    ti.xcom_push(
        key="train_result",
        value={
            "macro_f1": result.macro_f1,
            "recall_urgente": result.recall_urgente,
            "gate_passed": result.gate_passed,
            "gate_failures": list(result.gate_failures),
            "model_path": str(result.model_path),
            "metrics_path": str(result.metrics_path),
            "label_encoder_path": str(result.label_encoder_path),
            "confusion_matrix_path": str(result.confusion_matrix_path),
        },
    )


def _evaluate_model(ti: Any, **_: Any) -> None:
    """Task: enforce the project's quality gate.

    The gate itself (macro_f1 >= ``QualityGate.macro_f1_min``,
    recall_urgente >= ``QualityGate.recall_urgente_min`` -- 0.55/0.70 by
    default, see ``triagem.training.evaluate.QualityGate``) was already
    checked inside ``train_model`` via
    ``triagem.training.evaluate.check_quality_gate``. This task only reads
    that result back from XCom and fails the DAG run when it did not pass.
    """
    train_result = ti.xcom_pull(task_ids="train_model", key="train_result")
    if train_result is None:
        raise AirflowException("train_model produced no result to evaluate.")

    if not train_result["gate_passed"]:
        raise AirflowException("quality gate FAILED: " + "; ".join(train_result["gate_failures"]))

    logger.info(
        "quality gate PASSED: macro_f1=%.4f recall_urgente=%.4f",
        train_result["macro_f1"],
        train_result["recall_urgente"],
    )


def _publish_artifacts(ti: Any, **_: Any) -> None:
    """Task: confirm the trained artifacts landed on the shared volume.

    Real work, not a bare ``print`` -- it checks each artifact path
    ``train_model`` reported actually exists on disk before the run is
    considered complete.
    """
    from pathlib import Path

    train_result = ti.xcom_pull(task_ids="train_model", key="train_result")
    if train_result is None:
        raise AirflowException("train_model produced no artifacts to publish.")

    missing = [key for key in _ARTIFACT_KEYS if not Path(train_result[key]).exists()]
    if missing:
        raise AirflowException(f"missing artifact(s) on disk: {missing}")

    logger.info("publish_artifacts: all artifacts present at %s", train_result["model_path"])


with DAG(
    dag_id=DAG_ID,
    description="Download -> build dataset -> train -> evaluate -> publish the triage model.",
    schedule="@weekly",
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["triagem", "training"],
) as dag:
    download_dataset = PythonOperator(
        task_id="download_dataset",
        python_callable=_download_dataset,
    )
    build_dataset = PythonOperator(
        task_id="build_dataset",
        python_callable=_build_dataset,
    )
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )
    publish_artifacts = PythonOperator(
        task_id="publish_artifacts",
        python_callable=_publish_artifacts,
    )

    download_dataset >> build_dataset >> train_model >> evaluate_model >> publish_artifacts
