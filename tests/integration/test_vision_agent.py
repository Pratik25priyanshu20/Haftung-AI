"""Integration tests for VisionAgent (requires YOLO model download)."""
from __future__ import annotations

import pytest

from haftung_ai.types.state import HaftungState


@pytest.mark.skipif(True, reason="Requires YOLO model and sample video")
class TestVisionAgentIntegration:
    """Integration tests that require actual model weights and video."""

    def test_vision_agent_processes_video(self, tmp_path):
        from haftung_ai.agents.vision_agent import VisionAgent

        agent = VisionAgent()
        state: HaftungState = {
            "video_path": "data/sample_videos/test.mp4",
            "errors": [],
            "warnings": [],
        }
        result = agent(state)
        assert result.get("frames_processed", 0) > 0

    def test_vision_agent_missing_video(self):
        from haftung_ai.agents.vision_agent import VisionAgent

        agent = VisionAgent()
        state: HaftungState = {
            "video_path": "/nonexistent/video.mp4",
            "errors": [],
            "warnings": [],
        }
        result = agent(state)
        assert len(result.get("errors", [])) > 0
