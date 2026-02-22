"""Tests for scene graph builder."""
from dataclasses import dataclass

from haftung_ai.perception.scene_graph import SceneGraphBuilder


@dataclass
class MockTrack:
    track_id: int
    class_name: str = "car"
    x: float | None = None
    y: float | None = None
    vx: float | None = None
    vy: float | None = None
    ttc: float | None = None


def test_empty_tracks():
    builder = SceneGraphBuilder()
    graph = builder.build([], frame_id=0)
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_single_track():
    tracks = [MockTrack(track_id=1, x=10.0, y=20.0, vx=5.0, vy=0.0)]
    builder = SceneGraphBuilder()
    graph = builder.build(tracks)
    assert 1 in graph.nodes
    assert len(graph.edges) == 0


def test_two_close_tracks():
    tracks = [
        MockTrack(track_id=1, x=0.0, y=0.0, vx=5.0, vy=0.0),
        MockTrack(track_id=2, x=10.0, y=0.0, vx=-5.0, vy=0.0),
    ]
    builder = SceneGraphBuilder(proximity_threshold_m=50.0)
    graph = builder.build(tracks)
    assert len(graph.edges) == 1
    assert graph.edges[0].relation_type == "approaching"


def test_two_far_tracks():
    tracks = [
        MockTrack(track_id=1, x=0.0, y=0.0, vx=0.0, vy=0.0),
        MockTrack(track_id=2, x=100.0, y=0.0, vx=0.0, vy=0.0),
    ]
    builder = SceneGraphBuilder(proximity_threshold_m=50.0)
    graph = builder.build(tracks)
    assert len(graph.edges) == 0


def test_diverging_tracks():
    tracks = [
        MockTrack(track_id=1, x=0.0, y=0.0, vx=-5.0, vy=0.0),
        MockTrack(track_id=2, x=10.0, y=0.0, vx=5.0, vy=0.0),
    ]
    builder = SceneGraphBuilder()
    graph = builder.build(tracks)
    assert len(graph.edges) == 1
    assert graph.edges[0].relation_type == "diverging"


def test_following_tracks():
    tracks = [
        MockTrack(track_id=1, x=0.0, y=0.0, vx=5.0, vy=0.0),
        MockTrack(track_id=2, x=5.0, y=0.0, vx=5.0, vy=0.0),  # same velocity, close y
    ]
    builder = SceneGraphBuilder()
    graph = builder.build(tracks)
    assert len(graph.edges) == 1
    assert graph.edges[0].relation_type == "following"


def test_graph_metadata():
    tracks = [MockTrack(track_id=1, x=0.0, y=0.0)]
    builder = SceneGraphBuilder()
    graph = builder.build(tracks, frame_id=42, timestamp=1.5)
    assert graph.frame_id == 42
    assert graph.timestamp == 1.5
