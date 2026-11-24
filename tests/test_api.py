import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_api_roundtrip(monkeypatch, tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setenv("ASKDOCS_CONFIG", "configs/offline.yaml")
    monkeypatch.setenv("ASKDOCS_SET", f'["index_dir={tmp_path}/idx", "observability.trace_dir={tmp_path}/traces"]')
    from fastapi.testclient import TestClient

    from askdocs.api import server

    server._state.clear()
    with TestClient(server.app) as client:
        r = client.post("/ask", json={"question": "What does error LQ-1033 mean?", "include_contexts": True})
        assert r.status_code == 200
        body = r.json()
        assert body["citations"] and body["contexts"] and body["trace_id"]
        assert client.post("/feedback", json={"trace_id": body["trace_id"], "helpful": True}).json() == {"ok": True}
        assert "askdocs_request_latency_ms_bucket" in client.get("/metrics").text
        assert "<svg" in client.get("/dashboard").text
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.post("/ask", json={"question": ""}).status_code == 422
