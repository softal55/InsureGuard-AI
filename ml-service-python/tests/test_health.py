from __future__ import annotations


def test_health_ok(client):  # noqa: ANN001
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert body.get("modelLoaded") is True
