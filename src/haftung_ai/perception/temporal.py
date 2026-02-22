"""Temporal object prediction (from Autobahn)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from haftung_ai.perception.kalman import KalmanTrackManager


@dataclass
class PredictionPoint:
    t_s: float
    x_m: float
    y_m: float


@dataclass
class TrajectoryHypothesis:
    mode: str  # "straight", "left_deviate", "right_deviate"
    probability: float
    points: list[PredictionPoint] = field(default_factory=list)


class TemporalPredictor:
    """Predict future positions from Kalman velocity estimates."""

    def __init__(self, horizons_s: tuple[float, ...] = (0.5, 1.0, 2.0)):
        self.horizons_s = horizons_s

    def predict(self, kalman_manager: KalmanTrackManager, alive_ids: set) -> dict[int, list[PredictionPoint]]:
        predictions: dict[int, list[PredictionPoint]] = {}
        for tid in alive_ids:
            kf = kalman_manager.get_filter(tid)
            if kf is None:
                continue
            x, y = kf.position
            vx, vy = kf.velocity
            pts: list[PredictionPoint] = []
            for t in self.horizons_s:
                pts.append(PredictionPoint(t_s=t, x_m=x + vx * t, y_m=y + vy * t))
            predictions[tid] = pts
        return predictions

    def predict_topk(
        self,
        kalman_manager: KalmanTrackManager,
        alive_ids: set,
        k: int = 3,
        lateral_sigma_scale: float = 1.5,
    ) -> dict[int, list[TrajectoryHypothesis]]:
        topk: dict[int, list[TrajectoryHypothesis]] = {}
        for tid in alive_ids:
            kf = kalman_manager.get_filter(tid)
            if kf is None:
                continue
            x, y = kf.position
            vx, vy = kf.velocity
            vx_var = float(kf.P[2, 2])
            vy_var = float(kf.P[3, 3])
            lateral_std = np.sqrt(vx_var) * lateral_sigma_scale
            total_vel_var = vx_var + vy_var + 1e-6
            straight_conf = float(np.clip(1.0 / (1.0 + total_vel_var), 0.3, 0.8))
            deviate_conf = (1.0 - straight_conf) / 2.0

            hypotheses: list[TrajectoryHypothesis] = []
            straight_pts = [PredictionPoint(t_s=t, x_m=x + vx * t, y_m=y + vy * t) for t in self.horizons_s]
            hypotheses.append(TrajectoryHypothesis(mode="straight", probability=straight_conf, points=straight_pts))

            left_pts = [PredictionPoint(t_s=t, x_m=x + (vx - lateral_std) * t, y_m=y + vy * t) for t in self.horizons_s]
            hypotheses.append(TrajectoryHypothesis(mode="left_deviate", probability=deviate_conf, points=left_pts))

            right_pts = [PredictionPoint(t_s=t, x_m=x + (vx + lateral_std) * t, y_m=y + vy * t) for t in self.horizons_s]
            hypotheses.append(TrajectoryHypothesis(mode="right_deviate", probability=deviate_conf, points=right_pts))

            hypotheses.sort(key=lambda h: h.probability, reverse=True)
            topk[tid] = hypotheses[:k]
        return topk
