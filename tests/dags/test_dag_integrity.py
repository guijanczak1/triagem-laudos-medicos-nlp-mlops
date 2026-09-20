"""Integrity test for ``dags/train_triagem_dag.py``.

Requires apache-airflow (the optional Poetry group ``airflow`` --
``poetry install --with airflow``). apache-airflow does not install
natively on Windows (it needs ``pwd``/``fcntl``, which do not exist there),
so this entire module is SKIPPED (not failed) on a native Windows dev
machine via ``pytest.importorskip("airflow")``. It runs for real inside the
Airflow Docker image (T15) and in CI on an ubuntu-latest runner (T16),
where apache-airflow is actually installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402

DAG_ID = "train_triagem"

DAGS_DIR = Path(__file__).resolve().parent.parent.parent / "dags"

#: Expected tasks, in their required upstream -> downstream order (T14
#: acceptance: download_dataset >> build_dataset >> train_model >>
#: evaluate_model >> publish_artifacts).
EXPECTED_TASK_ORDER: tuple[str, ...] = (
    "download_dataset",
    "build_dataset",
    "train_model",
    "evaluate_model",
    "publish_artifacts",
)


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    """Load only ``dags/`` (not Airflow's bundled example DAGs)."""
    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def test_dag_imports_without_errors(dagbag: DagBag) -> None:
    """The DAG file must parse and import cleanly."""
    assert dagbag.import_errors == {}


def test_dag_id_present(dagbag: DagBag) -> None:
    """``train_triagem`` must be discovered by the DagBag."""
    assert DAG_ID in dagbag.dags


def test_dag_has_expected_tasks_and_no_cycles(dagbag: DagBag) -> None:
    """Exactly the five expected tasks are present, and the graph is acyclic.

    ``DAG.topological_sort()`` raises ``AirflowDagCycleException`` if the
    task graph has a cycle, so a successful sort of the expected count is
    itself the cycle check.
    """
    dag = dagbag.dags[DAG_ID]
    assert set(dag.task_ids) == set(EXPECTED_TASK_ORDER)

    ordered = dag.topological_sort()
    assert [task.task_id for task in ordered] == list(EXPECTED_TASK_ORDER)


def test_dag_dependencies_in_order(dagbag: DagBag) -> None:
    """Each task depends on exactly the previous one, in the documented order."""
    dag = dagbag.dags[DAG_ID]

    for upstream_id, downstream_id in zip(
        EXPECTED_TASK_ORDER, EXPECTED_TASK_ORDER[1:], strict=True
    ):
        upstream = dag.get_task(upstream_id)
        downstream = dag.get_task(downstream_id)
        assert downstream_id in upstream.downstream_task_ids
        assert upstream_id in downstream.upstream_task_ids

    first_task = dag.get_task(EXPECTED_TASK_ORDER[0])
    last_task = dag.get_task(EXPECTED_TASK_ORDER[-1])
    assert first_task.upstream_task_ids == set()
    assert last_task.downstream_task_ids == set()


def test_dag_schedule_and_safety_settings(dagbag: DagBag) -> None:
    """catchup/max_active_runs/retries match the T14 acceptance criteria."""
    dag = dagbag.dags[DAG_ID]

    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert dag.default_args.get("owner") == "triagem"
    assert dag.default_args.get("retries") == 1


def test_no_task_carries_a_literal_credential() -> None:
    """No credential-shaped literal is embedded in the DAG source (airflow.md's hard rule)."""
    source = (DAGS_DIR / "train_triagem_dag.py").read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("password=", "secret=", "api_key=", "aws_access_key_id="):
        assert forbidden not in lowered
