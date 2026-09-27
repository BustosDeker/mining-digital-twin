"""Tests de la Fase 9: rutas REST de simulación, ML, sesiones y WebSocket.

Nota metodológica: la reproducción automática en tiempo real (background
asyncio task del `SessionManager`) se valida de forma fiable con un
servidor Uvicorn real + cliente `websockets` (ver verificación manual en el
desarrollo de esta fase: streaming continuo confirmado paso a paso). El
`TestClient` de Starlette, al usar un event loop distinto por cada llamada
síncrona, no garantiza que una tarea `asyncio.create_task` lanzada en una
llamada seguiría corriendo en otra — es una limitación del arnés de
pruebas, no del servidor. Por eso estos tests cubren el ciclo de vida vía
REST (determinista) y la conexión inicial del WebSocket, sin depender de
recibir múltiples mensajes de streaming dentro del mismo test.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)

_DEFAULT_LAYOUT = {
    "n_levels": 3,
    "galleries_per_level": 8,
    "n_refuge_chambers": 3,
    "n_exits": 2,
    "n_risk_zones": 4,
    "random_seed": 42,
}


def _create_session(scenario_name: str, **overrides) -> dict:
    payload = {
        "scenario_name": scenario_name,
        "n_agents": 8,
        "router_name": "adaptive_astar",
        "hazard_type": "fire",
        "hazard_intensity": 0.9,
        "layout_config": _DEFAULT_LAYOUT,
        **overrides,
    }
    r = client.post("/api/simulations", json=payload)
    assert r.status_code == 200
    return r.json()


def test_health_check():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_session_and_get_state_has_agents_immediately():
    session = _create_session("test_api_create")
    assert session["status"] == "ready"

    r = client.get(f"/api/simulations/{session['session_id']}")
    assert r.status_code == 200
    snapshot = r.json()
    assert snapshot["step"] == 0
    assert len(snapshot["agents"]) == 8  # bug de la Fase 9 corregido: agentes desde el inicio
    assert len(snapshot["nodes"]) > 0
    assert len(snapshot["edges"]) > 0


def test_get_nonexistent_session_returns_404():
    r = client.get("/api/simulations/does-not-exist")
    assert r.status_code == 404


def test_manual_step_advances_state_deterministically():
    session = _create_session("test_api_step")
    sid = session["session_id"]

    r = client.post(f"/api/simulations/{sid}/step")
    assert r.status_code == 200
    assert r.json()["step"] == 1

    r = client.post(f"/api/simulations/{sid}/step")
    assert r.json()["step"] == 2


def test_pause_stop_reset_transitions():
    session = _create_session("test_api_lifecycle")
    sid = session["session_id"]

    r = client.post(f"/api/simulations/{sid}/start")
    assert r.json()["status"] == "running"

    r = client.post(f"/api/simulations/{sid}/pause")
    assert r.json()["status"] == "paused"

    r = client.post(f"/api/simulations/{sid}/stop")
    assert r.json()["status"] == "stopped"

    r = client.post(f"/api/simulations/{sid}/reset")
    assert r.json()["status"] == "ready"
    assert r.json()["step"] == 0


def test_list_sessions_includes_created_session():
    session = _create_session("test_api_list")
    r = client.get("/api/simulations")
    assert r.status_code == 200
    ids = [s["session_id"] for s in r.json()]
    assert session["session_id"] in ids


def test_websocket_receives_initial_snapshot_immediately():
    session = _create_session("test_api_ws")
    sid = session["session_id"]
    with client.websocket_connect(f"/ws/simulations/{sid}") as ws:
        initial = ws.receive_json()
    assert initial["session_id"] == sid
    assert initial["step"] == 0
    assert len(initial["agents"]) == 8


def test_websocket_unknown_session_closes_with_error_code():
    with pytest.raises(Exception):  # noqa: B017 - starlette lanza WebSocketDisconnect al rechazar
        with client.websocket_connect("/ws/simulations/unknown-session"):
            pass


def test_stopping_session_persists_to_history():
    session = _create_session("test_api_persist")
    sid = session["session_id"]
    client.post(f"/api/simulations/{sid}/step")
    client.post(f"/api/simulations/{sid}/stop")

    r = client.get("/api/sessions")
    assert r.status_code == 200
    ids = [rec["session_id"] for rec in r.json()]
    assert sid in ids

    r = client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["session_id"] == sid


def test_get_nonexistent_session_history_returns_404():
    r = client.get("/api/sessions/does-not-exist")
    assert r.status_code == 404


def test_list_datasets_includes_synthetic_and_real():
    r = client.get("/api/ml/datasets")
    assert r.status_code == 200
    names = {d["dataset_name"] for d in r.json()}
    assert "wesad_synthetic_demo" in names
    assert "wesad" in names


def test_run_eda_via_api_generates_artifacts():
    r = client.post("/api/ml/datasets/wesad_synthetic_demo/eda", json={"subject_limit": 2})
    assert r.status_code == 200
    assert r.json()["quality_report"]["n_subjects"] == 2

    r2 = client.get("/api/ml/datasets/wesad_synthetic_demo/eda/artifacts")
    assert r2.status_code == 200
    assert "class_distribution.png" in r2.json()


def test_eda_for_unavailable_dataset_returns_409():
    r = client.post("/api/ml/datasets/wesad/eda", json={})
    assert r.status_code == 409  # WESAD real no está descargado en este entorno


def test_eda_for_unknown_dataset_returns_404():
    r = client.post("/api/ml/datasets/does_not_exist/eda", json={})
    assert r.status_code == 404


def test_list_and_activate_models_roundtrip():
    r = client.get("/api/ml/models?architecture_name=features_mlp")
    assert r.status_code == 200
    models = r.json()
    assert len(models) > 0

    version_id = models[0]["version_id"]
    r2 = client.post(f"/api/ml/models/features_mlp/{version_id}/activate")
    assert r2.status_code == 200
    assert r2.json()["version_id"] == version_id

    r3 = client.get("/api/ml/models/active")
    assert r3.status_code == 200
    assert r3.json()["pointer"]["version_id"] == version_id


def test_predict_without_active_model_arch_mismatch_or_success():
    r = client.get("/api/ml/models/active")
    metadata = r.json()["metadata"]
    feature_names = (metadata.get("extra") or {}).get("feature_names")

    if not feature_names:
        pytest.skip("El modelo activo actual no registró feature_names en 'extra'.")

    features = {name: 0.5 for name in feature_names}
    r2 = client.post("/api/ml/predict", json={"features": features})
    assert r2.status_code == 200
    body = r2.json()
    assert body["predicted_class"] in metadata["class_names"]
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_list_reports_returns_pdf_filenames():
    r = client.get("/api/ml/reports")
    assert r.status_code == 200
    assert all(name.endswith(".pdf") for name in r.json())


def test_routing_comparison_endpoint_returns_statistical_decision():
    payload = {
        "scenario_name": "test_api_routing_mc",
        "router_names": ["adaptive_astar", "q_learning"],
        "n_runs": 3,
        "n_agents": 6,
        "hazard_type": "fire",
        "layout_config": _DEFAULT_LAYOUT,
        "q_learning_training_episodes": 10,
    }
    r = client.post("/api/evaluation/routing-comparison", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "adaptive_astar" in body["per_router_summary"]
    assert "q_learning" in body["per_router_summary"]
    assert "statistically_justified_best" in body["statistical_decision"]
