"""Per-object Kalman filter with RTS smoother (from Autobahn + backward pass)."""
from __future__ import annotations

import numpy as np


class ObjectKalmanFilter:
    """Linear Kalman filter for a single tracked object.

    State vector: [x, y, vx, vy] (position and velocity in meters).
    """

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 1.0,
        dt: float = 0.033,
    ):
        self.dt = dt
        self.x = np.zeros(4, dtype=np.float64)
        self.F = np.eye(4, dtype=np.float64)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.P = np.eye(4, dtype=np.float64) * 10.0
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # History for RTS smoother
        self._x_history: list[np.ndarray] = []
        self._P_history: list[np.ndarray] = []
        self._x_pred_history: list[np.ndarray] = []
        self._P_pred_history: list[np.ndarray] = []

    def predict(self) -> np.ndarray:
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        self._x_pred_history.append(x_pred.copy())
        self._P_pred_history.append(P_pred.copy())
        self.x = x_pred
        self.P = P_pred
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        eye = np.eye(4)
        self.P = (eye - K @ self.H) @ self.P
        self._x_history.append(self.x.copy())
        self._P_history.append(self.P.copy())
        return self.x.copy()

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def rts_smooth(self) -> list[np.ndarray]:
        """Rauch-Tung-Striebel backward smoothing pass.

        Returns smoothed state estimates for all time steps.
        """
        n = len(self._x_history)
        if n == 0:
            return []
        if n == 1:
            return [self._x_history[0].copy()]

        x_smooth = [np.zeros(4)] * n
        P_smooth = [np.zeros((4, 4))] * n
        x_smooth[-1] = self._x_history[-1].copy()
        P_smooth[-1] = self._P_history[-1].copy()

        for k in range(n - 2, -1, -1):
            P_pred = self._P_pred_history[k + 1] if k + 1 < len(self._P_pred_history) else self.F @ self._P_history[k] @ self.F.T + self.Q
            P_pred_inv = np.linalg.inv(P_pred + np.eye(4) * 1e-8)
            G = self._P_history[k] @ self.F.T @ P_pred_inv

            x_pred = self._x_pred_history[k + 1] if k + 1 < len(self._x_pred_history) else self.F @ self._x_history[k]
            x_smooth[k] = self._x_history[k] + G @ (x_smooth[k + 1] - x_pred)
            P_smooth[k] = self._P_history[k] + G @ (P_smooth[k + 1] - P_pred) @ G.T

        return x_smooth


class KalmanTrackManager:
    """Manages per-ID Kalman filters."""

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 1.0,
        dt: float = 0.033,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.dt = dt
        self.filters: dict[int, ObjectKalmanFilter] = {}

    def update_track(self, track_id: int, x_m: float, y_m: float) -> tuple[float, float, float, float]:
        if track_id not in self.filters:
            kf = ObjectKalmanFilter(self.process_noise, self.measurement_noise, self.dt)
            kf.x[:2] = [x_m, y_m]
            self.filters[track_id] = kf

        kf = self.filters[track_id]
        kf.predict()
        state = kf.update(np.array([x_m, y_m]))
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])

    def prune(self, alive_ids: set) -> None:
        dead = [tid for tid in self.filters if tid not in alive_ids]
        for tid in dead:
            del self.filters[tid]

    def get_filter(self, track_id: int) -> ObjectKalmanFilter | None:
        return self.filters.get(track_id)

    def smooth_all(self) -> dict[int, list[np.ndarray]]:
        """Run RTS smoother on all tracked objects. Returns smoothed states per track."""
        smoothed: dict[int, list[np.ndarray]] = {}
        for tid, kf in self.filters.items():
            smoothed[tid] = kf.rts_smooth()
        return smoothed
