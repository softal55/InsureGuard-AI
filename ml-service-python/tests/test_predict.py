from __future__ import annotations


def test_predict_returns_core_fields(client):  # noqa: ANN001
    r = client.post(
        "/predict?explain=compact",
        json={
            "total_claim_amount": 25000,
            "vehicle_claim": 8000,
            "property_claim": 7000,
            "incident_hour_of_day": 14,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "fraudProbability" in body
    assert "requestId" in body
    assert "modelMeta" in body
    assert "modelConfidence" in body
    assert "driftDetected" in body
