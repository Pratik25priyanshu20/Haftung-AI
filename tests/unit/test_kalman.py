"""Tests for Kalman filter and RTS smoother."""
import numpy as np

from haftung_ai.perception.kalman import KalmanTrackManager, ObjectKalmanFilter


def test_kalman_init():
    kf = ObjectKalmanFilter(process_noise=0.5, measurement_noise=1.0, dt=0.033)
    assert kf.x.shape == (4,)
    assert kf.P.shape == (4, 4)
    assert kf.dt == 0.033


def test_kalman_predict_update():
    kf = ObjectKalmanFilter(dt=0.1)
    kf.x[:2] = [5.0, 10.0]
    kf.predict()
    state = kf.update(np.array([5.1, 10.0]))
    assert abs(state[0] - 5.1) < 1.0
    assert abs(state[1] - 10.0) < 1.0


def test_kalman_velocity_estimation():
    kf = ObjectKalmanFilter(dt=0.1, process_noise=0.1, measurement_noise=0.1)
    kf.x[:2] = [0.0, 0.0]
    for i in range(20):
        kf.predict()
        kf.update(np.array([i * 1.0, 0.0]))
    vx, vy = kf.velocity
    assert vx > 5.0  # Should converge toward ~10 m/s


def test_kalman_position_property():
    kf = ObjectKalmanFilter()
    kf.x[:2] = [3.0, 7.0]
    assert kf.position == (3.0, 7.0)


def test_rts_smooth_single():
    kf = ObjectKalmanFilter(dt=0.1)
    kf.x[:2] = [0.0, 0.0]
    kf.predict()
    kf.update(np.array([1.0, 0.0]))
    smoothed = kf.rts_smooth()
    assert len(smoothed) == 1


def test_rts_smooth_reduces_noise():
    kf = ObjectKalmanFilter(dt=0.1, process_noise=0.01, measurement_noise=5.0)
    kf.x[:2] = [0.0, 0.0]
    np.random.seed(42)
    # Object moving at 1 m/step with noise
    for i in range(30):
        kf.predict()
        kf.update(np.array([i * 1.0 + np.random.randn() * 2.0, 0.0]))

    forward_states = kf._x_history
    smoothed = kf.rts_smooth()
    assert len(smoothed) == len(forward_states)

    # Smoothed should be closer to ground truth
    forward_errors = [abs(forward_states[i][0] - i * 1.0) for i in range(len(forward_states))]
    smooth_errors = [abs(smoothed[i][0] - i * 1.0) for i in range(len(smoothed))]
    assert sum(smooth_errors) <= sum(forward_errors)


def test_rts_smooth_empty():
    kf = ObjectKalmanFilter()
    assert kf.rts_smooth() == []


def test_track_manager_update():
    mgr = KalmanTrackManager(dt=0.1)
    x, y, vx, vy = mgr.update_track(1, 5.0, 10.0)
    assert abs(x - 5.0) < 5.0
    assert 1 in mgr.filters


def test_track_manager_prune():
    mgr = KalmanTrackManager()
    mgr.update_track(1, 0.0, 0.0)
    mgr.update_track(2, 1.0, 1.0)
    mgr.prune({1})
    assert 1 in mgr.filters
    assert 2 not in mgr.filters


def test_track_manager_smooth_all():
    mgr = KalmanTrackManager(dt=0.1)
    for i in range(10):
        mgr.update_track(1, float(i), 0.0)
        mgr.update_track(2, 0.0, float(i))
    smoothed = mgr.smooth_all()
    assert 1 in smoothed
    assert 2 in smoothed
    assert len(smoothed[1]) == 10
