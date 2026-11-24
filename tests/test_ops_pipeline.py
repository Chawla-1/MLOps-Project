"""
Tests for CI, K8s specs, Logging, and Monitoring/Drift components.
These tests run without needing raw CSVs (self-contained fixtures).
"""

import io
import json
import logging
import os
import yaml
import numpy as np
import pandas as pd
import pytest

from src.logger import JSONFormatter, get_logger
from src.monitor import calculate_psi, evaluate_promotion


def test_k8s_deployment_spec():
    """Verify k8s/deployment.yaml matches problem requirements."""
    k8s_path = os.path.join(os.path.dirname(__file__), "..", "k8s", "deployment.yaml")
    assert os.path.exists(k8s_path), "k8s/deployment.yaml must exist"

    with open(k8s_path, "r", encoding="utf-8") as f:
        deploy = yaml.safe_load(f)

    assert deploy["metadata"]["name"] == "mp141-api"
    assert deploy["spec"]["replicas"] == 2

    container = deploy["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "mp141-api:1.0"
    assert container["ports"][0]["containerPort"] == 8000

    # Verify probes
    readiness = container["readinessProbe"]
    liveness = container["livenessProbe"]
    assert readiness["httpGet"]["path"] == "/health"
    assert liveness["httpGet"]["path"] == "/health"
    # Liveness probe must wait longer than readiness probe
    assert liveness["initialDelaySeconds"] > readiness["initialDelaySeconds"]


def test_k8s_service_spec():
    """Verify k8s/service.yaml matches problem requirements."""
    svc_path = os.path.join(os.path.dirname(__file__), "..", "k8s", "service.yaml")
    assert os.path.exists(svc_path), "k8s/service.yaml must exist"

    with open(svc_path, "r", encoding="utf-8") as f:
        svc = yaml.safe_load(f)

    assert svc["metadata"]["name"] == "mp141-svc"
    assert svc["spec"]["selector"]["app"] == "mp141-api"
    port_spec = svc["spec"]["ports"][0]
    assert port_spec["port"] == 80
    assert port_spec["targetPort"] == 8000


def test_json_logging_format():
    """Verify logger emits valid single-line JSON and redacts personal info."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    # Log record with extra fields
    record = logger.makeRecord(
        name="test_logger",
        level=logging.INFO,
        fn="",
        lno=0,
        msg="prediction generated",
        args=(),
        exc_info=None,
    )
    record.extra_info = {
        "prediction": 8500.0,
        "latency_ms": 12.4,
        "name": "Secret Person",  # Should be redacted
        "student_id": "12345"    # Should be redacted
    }
    logger.handle(record)

    output = stream.getvalue().strip()
    data = json.loads(output)
    assert data["level"] == "INFO"
    assert data["message"] == "prediction generated"
    assert data["prediction"] == 8500.0
    assert "name" not in data
    assert "student_id" not in data


def test_psi_calculation():
    """Test Population Stability Index with identical vs drifted distributions."""
    np.random.seed(42)
    ref = pd.Series(np.random.normal(loc=10, scale=2, size=1000))
    same = pd.Series(np.random.normal(loc=10, scale=2, size=1000))
    shifted = pd.Series(np.random.normal(loc=20, scale=2, size=1000))

    psi_same = calculate_psi(ref, same, num_bins=10)
    psi_shifted = calculate_psi(ref, shifted, num_bins=10)

    # Identical distributions should have very low PSI (< 0.1)
    assert psi_same < 0.10
    # Significantly drifted distribution should have high PSI (> 0.25)
    assert psi_shifted > 0.25


def test_promotion_rule():
    """Test model promotion decision logic (must improve by at least 10%)."""
    # 15% improvement -> promote
    decision, imp = evaluate_promotion(mae_v1=1000.0, mae_v2=850.0, threshold=0.10)
    assert decision == "promote"
    assert imp == 0.15

    # 5% improvement -> keep
    decision, imp = evaluate_promotion(mae_v1=1000.0, mae_v2=950.0, threshold=0.10)
    assert decision == "keep"
    assert imp == 0.05

    # Regression (worse) -> keep
    decision, imp = evaluate_promotion(mae_v1=1000.0, mae_v2=1050.0, threshold=0.10)
    assert decision == "keep"
