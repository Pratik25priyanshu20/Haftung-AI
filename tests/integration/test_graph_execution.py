"""Integration tests for the full LangGraph orchestrator."""
from __future__ import annotations

import pytest


@pytest.mark.skipif(True, reason="Requires all agents + API keys configured")
class TestGraphExecution:

    def test_s1_graph_compiles(self):
        from haftung_ai.agents.orchestrator import build_graph

        graph = build_graph("S1")
        assert graph is not None

    def test_s2_graph_compiles(self):
        from haftung_ai.agents.orchestrator import build_graph

        graph = build_graph("S2")
        assert graph is not None

    def test_s3_graph_compiles(self):
        from haftung_ai.agents.orchestrator import build_graph

        graph = build_graph("S3")
        assert graph is not None

    def test_invalid_variant_raises(self):
        from haftung_ai.agents.orchestrator import build_graph

        with pytest.raises(ValueError, match="Unknown variant"):
            build_graph("S99")
