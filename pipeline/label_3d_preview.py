"""Render a frame-sampled Anipose 3D preview without resampling source video."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import pandas as pd


def source_video_fps(path: Path) -> float:
    """Read only the source video's metadata and return its frame rate."""
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise RuntimeError(f"Could not read a positive frame rate from {path}")
    return fps


def sampled_pose_csv(source: Path, target_fps: float, source_fps: float,
                     destination: Path) -> tuple[int, float]:
    """Write sampled coordinates and return the stride and actual output FPS."""
    stride = max(1, round(source_fps / target_fps))
    output_fps = source_fps / stride
    frame = pd.read_csv(source)
    sampled = frame.iloc[::stride].copy().reset_index(drop=True)
    if "fnum" in sampled.columns:
        # Anipose's renderer expects frame numbers to index the sampled rows.
        sampled["fnum"] = sampled.index
    sampled.to_csv(destination, index=False)
    return stride, output_fps


def render_preview(config_path: Path, pose_csv: Path, source_video: Path,
                   output: Path, target_fps: float) -> None:
    """Render one coordinate CSV at a reduced preview frame rate."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib
    from anipose.label_videos_3d import visualize_labels

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    original_fps = source_video_fps(source_video)
    pose_csv.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=".label3d_",
            dir=pose_csv.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        stride, output_fps = sampled_pose_csv(
            pose_csv, target_fps, original_fps, temporary_path
        )
        print(
            f"Rendering {pose_csv.name}: source {original_fps:g} FPS, "
            f"every {stride} frame(s), output {output_fps:g} FPS",
            flush=True,
        )
        visualize_labels(config, str(temporary_path), str(output), output_fps)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pose-csv", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    render_preview(
        args.config, args.pose_csv, args.source_video, args.output, args.fps
    )


if __name__ == "__main__":
    main()
