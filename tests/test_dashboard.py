"""Tests for the provisioned Grafana dashboard and its YAML providers (T19).

No Docker/Grafana required: this only parses the versioned files on disk
(JSON/YAML) and checks their shape -- the actual "does Grafana render it"
check is done manually/via the Grafana API per ``docs/monitoring.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_JSON = PROJECT_ROOT / "docker" / "grafana" / "dashboards" / "triagem_api.json"
DASHBOARD_PROVIDER_YML = (
    PROJECT_ROOT / "docker" / "grafana" / "provisioning" / "dashboards" / "dashboard.yml"
)
DATASOURCE_YML = (
    PROJECT_ROOT / "docker" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
)

#: The PromQL queries the T19 backlog entry requires (see
#: state/mlet-tech-challenge-fase-3/backlog.json, task T19's acceptance
#: criteria) -- guards against the dashboard drifting from the expressions
#: that were agreed on. Panels 1/2/4 use the backlog's literal expressions
#: verbatim. Panel 3 (taxa de erro) is the one deliberate exception: the
#: backlog's literal `rate(triagem_errors_total[5m]) /
#: rate(triagem_requests_total[5m])` never matches any series in real
#: Prometheus (triagem_errors_total{endpoint,type} and
#: triagem_requests_total{endpoint,method,status} have different label
#: sets, and Prometheus vector division only matches identical label sets)
#: -- confirmed live against the running stack: a real 422 incremented
#: triagem_errors_total but the literal query still returned 0 series. The
#: panel instead aggregates both sides with sum() first (same two metrics,
#: same 5m window, correct match semantics) -- see the dashboard JSON's own
#: panel description and docs/monitoring.md for the full rationale.
REQUIRED_EXPRESSIONS = (
    "rate(triagem_requests_total[1m])",
    "triagem_request_duration_seconds_bucket",
    "triagem_predictions_total",
)

#: Panel 3 (taxa de erro): the two source metrics/windows the backlog names
#: must still both be present, even though they're wrapped in sum() to fix
#: label matching -- see REQUIRED_EXPRESSIONS' comment above.
REQUIRED_ERROR_RATE_FRAGMENTS = (
    "rate(triagem_errors_total[5m])",
    "rate(triagem_requests_total[5m])",
)


def _load_dashboard() -> dict[str, object]:
    return dict(json.loads(DASHBOARD_JSON.read_text(encoding="utf-8")))


def _all_target_expressions(dashboard: dict[str, object]) -> list[str]:
    panels = dashboard["panels"]
    assert isinstance(panels, list)
    exprs: list[str] = []
    for panel in panels:
        assert isinstance(panel, dict)
        for target in panel["targets"]:
            assert isinstance(target, dict)
            exprs.append(str(target["expr"]))
    return exprs


class TestDashboardJson:
    """docker/grafana/dashboards/triagem_api.json -- parse + shape."""

    def test_file_exists(self) -> None:
        assert DASHBOARD_JSON.is_file()

    def test_parses_as_valid_json(self) -> None:
        dashboard = _load_dashboard()
        assert isinstance(dashboard, dict)

    def test_has_at_least_four_panels(self) -> None:
        dashboard = _load_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        assert len(panels) >= 4

    def test_every_panel_has_at_least_one_target_with_a_promql_expr(self) -> None:
        dashboard = _load_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        for panel in panels:
            assert isinstance(panel, dict)
            targets = panel["targets"]
            assert isinstance(targets, list) and len(targets) >= 1
            for target in targets:
                assert isinstance(target, dict)
                expr = target["expr"]
                assert isinstance(expr, str) and expr.strip()

    def test_every_panel_has_a_title_and_a_type(self) -> None:
        dashboard = _load_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        for panel in panels:
            assert isinstance(panel, dict)
            assert isinstance(panel.get("title"), str) and panel["title"]
            assert isinstance(panel.get("type"), str) and panel["type"]

    def test_required_promql_expressions_are_all_present(self) -> None:
        """3 of the 4 T19 backlog queries (panels 1/2/4) must appear verbatim."""
        exprs = _all_target_expressions(_load_dashboard())
        joined = " | ".join(exprs)
        for required in REQUIRED_EXPRESSIONS:
            assert required in joined, f"missing required PromQL expression: {required!r}"

    def test_error_rate_panel_uses_sum_to_fix_label_matching(self) -> None:
        """Panel 3 must divide sum()s (not raw rate()s -- see module docstring)."""
        exprs = _all_target_expressions(_load_dashboard())
        error_rate_exprs = [e for e in exprs if "triagem_errors_total" in e]
        assert len(error_rate_exprs) == 1
        expr = error_rate_exprs[0]
        for fragment in REQUIRED_ERROR_RATE_FRAGMENTS:
            assert fragment in expr, f"missing fragment in error-rate expr: {fragment!r}"
        assert (
            expr.startswith("sum(") and " / sum(" in expr
        ), f"error-rate expr must sum() both sides before dividing, got: {expr!r}"

    def test_panels_use_no_metric_outside_the_six_declared_in_metrics_py(self) -> None:
        """Every referenced metric must be one of triagem.serving.metrics's 6 (T17 contract)."""
        declared_metrics = {
            "triagem_requests_total",
            "triagem_request_duration_seconds",
            "triagem_request_duration_seconds_bucket",
            "triagem_predictions_total",
            "triagem_errors_total",
            "triagem_inference_duration_seconds",
            "triagem_model_info",
        }
        exprs = _all_target_expressions(_load_dashboard())
        for expr in exprs:
            for token in expr.replace("(", " ").replace(")", " ").split():
                if token.startswith("triagem_"):
                    metric_name = token.split("[")[0].split("{")[0]
                    assert (
                        metric_name in declared_metrics
                    ), f"expr {expr!r} references undeclared metric {metric_name!r}"

    def test_panels_reference_the_provisioned_prometheus_datasource_uid(self) -> None:
        dashboard = _load_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        for panel in panels:
            assert isinstance(panel, dict)
            datasource = panel.get("datasource")
            assert isinstance(datasource, dict)
            assert datasource.get("uid") == "prometheus"


class TestProvisioningYaml:
    """The two apiVersion-1 provisioning YAML files -- parse + wiring."""

    def test_dashboard_provider_yml_parses_and_points_at_the_mounted_dashboards_dir(self) -> None:
        config = yaml.safe_load(DASHBOARD_PROVIDER_YML.read_text(encoding="utf-8"))
        assert config["apiVersion"] == 1
        providers = config["providers"]
        assert isinstance(providers, list) and len(providers) == 1
        assert providers[0]["type"] == "file"
        assert providers[0]["options"]["path"] == "/etc/grafana/dashboards"

    def test_datasource_yml_declares_the_fixed_uid_dashboards_rely_on(self) -> None:
        config = yaml.safe_load(DATASOURCE_YML.read_text(encoding="utf-8"))
        datasources = config["datasources"]
        assert isinstance(datasources, list) and len(datasources) == 1
        assert datasources[0]["uid"] == "prometheus"
        assert datasources[0]["type"] == "prometheus"
