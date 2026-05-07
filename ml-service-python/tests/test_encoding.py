from __future__ import annotations


def test_detect_drift_sigma_rule(client):  # noqa: ANN001
    from main import detect_drift

    assert detect_drift(100.0, 0.0, 10.0) is True
    assert detect_drift(5.0, 0.0, 10.0) is False
    assert detect_drift(5.0, 0.0, 0.0) is False


def test_feature_matrix_aligned(client):  # noqa: ANN001
    from main import _encode_row

    raw = {
        "total_claim_amount": 25000,
        "vehicle_claim": 8000,
        "property_claim": 7000,
        "incident_hour_of_day": 14,
    }
    X = _encode_row(raw)
    assert X.shape[0] == 1
    assert X.shape[1] > 0
