"""Vision Agent — wraps perception pipeline into a LangGraph node."""
from __future__ import annotations

import logging
from typing import Any

from haftung_ai.types.state import HaftungState, validate_state

logger = logging.getLogger(__name__)


class VisionAgent:
    """Processes dashcam video through detection, tracking, Kalman, scene graph, and impact detection.

    This is a clean ~150-line agent (not the 1100-line Autobahn orchestrator).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._detector = None
        self._tracker = None
        self._kalman = None
        self._temporal = None
        self._scene_builder = None
        self._impact_detector = None

    def _lazy_init(self) -> None:
        if self._detector is not None:
            return
        from haftung_ai.perception.detector import YOLODetector
        from haftung_ai.perception.impact_detector import ImpactDetector
        from haftung_ai.perception.kalman import KalmanTrackManager
        from haftung_ai.perception.scene_graph import SceneGraphBuilder
        from haftung_ai.perception.temporal import TemporalPredictor
        from haftung_ai.perception.tracker import DeepSORTTracker

        self._detector = YOLODetector(
            model_name=self._config.get("detector_model", "yolov8n"),
        )
        self._tracker = DeepSORTTracker()
        self._kalman = KalmanTrackManager(
            process_noise=self._config.get("process_noise", 0.5),
            measurement_noise=self._config.get("measurement_noise", 1.0),
        )
        self._temporal = TemporalPredictor()
        self._scene_builder = SceneGraphBuilder()
        self._impact_detector = ImpactDetector()

    def __call__(self, state: HaftungState) -> HaftungState:
        missing = validate_state(state, ["video_path"])
        if missing:
            state.setdefault("errors", []).append(f"VisionAgent: missing {missing}")
            return state

        self._lazy_init()

        from haftung_ai.perception.video_input import VideoInput
        from haftung_ai.safety.ttc import compute_ttc

        video = VideoInput(state["video_path"])
        all_tracks: list[dict] = []
        world_models: list[dict] = []
        impact_frame = None
        impact_timestamp = None
        frames_processed = 0

        for frame_id, packet in video.frames():
            frames_processed += 1

            detections = self._detector.infer(packet.frame)
            tracks, trajectories = self._tracker.update(packet.frame, detections)

            alive_ids = set()
            for trk in tracks:
                x, y, vx, vy = self._kalman.update_track(trk.track_id, trk.bbox_xyxy[0], trk.bbox_xyxy[1])
                trk.x, trk.y, trk.vx, trk.vy = x, y, vx, vy
                alive_ids.add(trk.track_id)

                dist = (x**2 + y**2) ** 0.5
                closing = -((x * vx + y * vy) / dist) if dist > 1e-6 else 0.0
                trk.ttc = compute_ttc(dist, closing)

            self._kalman.prune(alive_ids)
            predictions = self._temporal.predict(self._kalman, alive_ids)
            scene = self._scene_builder.build(tracks, frame_id, packet.timestamp)

            if impact_frame is None:
                impact_event = self._impact_detector.check_frame(tracks, frame_id, packet.timestamp)
                if impact_event:
                    impact_frame = impact_event.frame_id
                    impact_timestamp = impact_event.timestamp

            for trk in tracks:
                all_tracks.append({
                    "frame_id": frame_id,
                    "track_id": trk.track_id,
                    "class_name": trk.class_name,
                    "bbox": trk.bbox_xyxy,
                    "x": trk.x,
                    "y": trk.y,
                    "vx": trk.vx,
                    "vy": trk.vy,
                    "ttc": trk.ttc,
                })

            world_models.append({
                "frame_id": frame_id,
                "timestamp": packet.timestamp,
                "num_detections": len(detections),
                "num_tracks": len(tracks),
                "scene_edges": len(scene.edges),
            })

        video.release()

        # Run RTS smoother for better post-hoc trajectories
        smoothed = self._kalman.smooth_all()

        state["frames_processed"] = frames_processed
        state["tracks"] = all_tracks
        state["impact_frame"] = impact_frame
        state["impact_timestamp"] = impact_timestamp
        state["world_models"] = world_models

        logger.info("VisionAgent: processed %d frames, impact_frame=%s", frames_processed, impact_frame)
        return state
