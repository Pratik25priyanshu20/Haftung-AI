"""Create evaluation dataset directory structure with template ground truths."""
from __future__ import annotations

import json
from pathlib import Path

# 6 categories x 5 each = 30 accidents
CATEGORIES = [
    "rear_end",
    "side_collision",
    "head_on",
    "intersection",
    "pedestrian",
    "single_vehicle",
]
SAMPLES_PER_CATEGORY = 5

GROUND_TRUTH_TEMPLATE = {
    "accident_id": "",
    "category": "",
    "primary_cause": "",
    "accident_type": "",
    "contributing_factors": [
        {
            "category": "",
            "factor": "",
            "severity": "primary",
        }
    ],
    "responsibility": [
        {
            "party": "ego",
            "percentage": 0.0,
            "rationale": "",
        },
        {
            "party": "other_1",
            "percentage": 0.0,
            "rationale": "",
        },
    ],
    "legal_references": [],
    "description": "",
    "weather": "clear",
    "time_of_day": "daytime",
    "road_type": "urban",
    "speed_limit_kmh": 50,
}


def create_dataset(output_dir: Path) -> None:
    """Create the evaluation dataset directory structure."""
    accidents_dir = output_dir / "accidents"
    accidents_dir.mkdir(parents=True, exist_ok=True)

    for category in CATEGORIES:
        for i in range(1, SAMPLES_PER_CATEGORY + 1):
            accident_id = f"{category}_{i:03d}"
            accident_dir = accidents_dir / accident_id
            accident_dir.mkdir(exist_ok=True)

            # Create ground truth template
            gt = {**GROUND_TRUTH_TEMPLATE}
            gt["accident_id"] = accident_id
            gt["category"] = category
            gt["accident_type"] = category

            gt_path = accident_dir / "ground_truth.json"
            if not gt_path.exists():
                with open(gt_path, "w") as f:
                    json.dump(gt, f, indent=2)

            # Create placeholder files
            video_placeholder = accident_dir / "video.mp4"
            if not video_placeholder.exists():
                video_placeholder.touch()

            can_placeholder = accident_dir / "can_log.csv"
            if not can_placeholder.exists():
                can_placeholder.write_text("timestamp,arbitration_id,data,channel\n")

            print(f"Created: {accident_id}")

    print(f"\nDataset structure created at {accidents_dir}")
    print(f"Total accidents: {len(CATEGORIES) * SAMPLES_PER_CATEGORY}")
    print("\nNext steps:")
    print("  1. Add video.mp4 files for each accident")
    print("  2. Add can_log.csv data for each accident")
    print("  3. Fill in ground_truth.json for each accident")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create evaluation dataset structure")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation" / "dataset",
    )
    args = parser.parse_args()

    create_dataset(args.output)
