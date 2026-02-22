"""Integration tests for FastAPI endpoints."""
from __future__ import annotations

import pytest


@pytest.mark.skipif(True, reason="Requires running API server")
class TestAPIEndpoints:

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient

        from haftung_ai.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_analyze_without_files(self):
        from fastapi.testclient import TestClient

        from haftung_ai.api.main import app

        client = TestClient(app)
        response = client.post("/analyze")
        assert response.status_code == 422  # Missing required files
