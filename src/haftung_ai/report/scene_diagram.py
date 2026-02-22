"""Bird's-eye view (BEV) scene diagram using matplotlib."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SceneDiagramGenerator:
    """Generate BEV scene diagrams for accident reports."""

    def __init__(self, figsize: tuple[int, int] = (10, 10)):
        self.figsize = figsize

    def generate(
        self,
        ego_states: list[dict],
        tracks: list[dict],
        impact_frame: int | None = None,
        output_path: str | Path = "scene_diagram.png",
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Plot ego trajectory
        if ego_states:
            ex = [s.get("x", 0) for s in ego_states]
            ey = [s.get("y", 0) for s in ego_states]
            ax.plot(ex, ey, "b-", linewidth=2, label="Ego-Fahrzeug")
            ax.plot(ex[0], ey[0], "bs", markersize=10)
            if len(ex) > 1:
                ax.plot(ex[-1], ey[-1], "b^", markersize=10)

        # Plot other vehicle tracks
        track_ids = set(t.get("track_id") for t in tracks if t.get("x") is not None)
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(track_ids), 1)))

        for idx, tid in enumerate(sorted(track_ids)):
            trk_points = sorted(
                [t for t in tracks if t.get("track_id") == tid and t.get("x") is not None],
                key=lambda t: t.get("frame_id", 0),
            )
            if not trk_points:
                continue
            tx = [t["x"] for t in trk_points]
            ty = [t["y"] for t in trk_points]
            color = colors[idx % len(colors)]
            label = f"{trk_points[0].get('class_name', 'Objekt')} #{tid}"
            ax.plot(tx, ty, "-", color=color, linewidth=1.5, label=label)
            ax.plot(tx[0], ty[0], "s", color=color, markersize=8)

        # Mark impact point
        if impact_frame is not None:
            impact_tracks = [t for t in tracks if t.get("frame_id") == impact_frame]
            for t in impact_tracks:
                if t.get("x") is not None:
                    ax.plot(t["x"], t["y"], "rx", markersize=15, markeredgewidth=3)

            ax.annotate(
                f"Aufprall (Frame {impact_frame})",
                xy=(impact_tracks[0]["x"], impact_tracks[0]["y"]) if impact_tracks and impact_tracks[0].get("x") else (0, 0),
                fontsize=12,
                color="red",
                fontweight="bold",
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Unfallszene — Draufsicht (BEV)")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Scene diagram saved: %s", output_path)
        return output_path
