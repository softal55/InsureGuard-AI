"""Pytest wiring: skip API tests unless trained artifacts exist under ml-service-python/."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    root = Path(__file__).resolve().parent.parent
    if (root / "artifacts" / "fraud_model.pkl").exists():
        return
    skip = pytest.mark.skip(
        reason="Requires artifacts/fraud_model.pkl (run data-pipeline/train_model.py)",
    )
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def client():  # noqa: ANN001
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)
