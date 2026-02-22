"""Tests for experiment runners."""
from __future__ import annotations

import json

import pytest

from evaluation.runners.run_experiment import load_ground_truth


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal evaluation dataset."""
    accidents_dir = tmp_path / "accidents"
    for category in ["rear_end", "side_collision"]:
        accident_dir = accidents_dir / f"{category}_001"
        accident_dir.mkdir(parents=True)

        gt = {
            "accident_id": f"{category}_001",
            "category": category,
            "primary_cause": f"{category} cause",
            "accident_type": category,
            "contributing_factors": [{"category": "speed", "factor": "fast", "severity": "primary"}],
            "responsibility": [{"party": "ego", "percentage": 60.0, "rationale": "test"}],
        }
        (accident_dir / "ground_truth.json").write_text(json.dumps(gt))
        (accident_dir / "video.mp4").touch()
        (accident_dir / "can_log.csv").write_text("timestamp,arbitration_id,data,channel\n")

    return accidents_dir


def test_load_ground_truth(sample_dataset):
    accident_dir = sample_dataset / "rear_end_001"
    gt = load_ground_truth(accident_dir)
    assert gt["accident_id"] == "rear_end_001"
    assert gt["primary_cause"] == "rear_end cause"


def test_ground_truth_has_required_fields(sample_dataset):
    accident_dir = sample_dataset / "side_collision_001"
    gt = load_ground_truth(accident_dir)
    assert "primary_cause" in gt
    assert "responsibility" in gt
    assert "contributing_factors" in gt


def test_dataset_structure(sample_dataset):
    dirs = list(sample_dataset.iterdir())
    assert len(dirs) == 2
    for d in dirs:
        assert (d / "ground_truth.json").exists()
        assert (d / "video.mp4").exists()
        assert (d / "can_log.csv").exists()
