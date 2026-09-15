"""Prepare and downsample calibration videos without changing their selection rules."""

from __future__ import annotations

import re
from pathlib import Path

from .calibration_prep import VIDEO_EXTS, downsample_dir, restructure_and_downsample


class VideoProcessor:
    def __init__(self, experiment_dir: str | Path, setup: dict):
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.setup = setup

    def prepare(self, session_dir: str | Path) -> None:
        session = Path(session_dir)
        cages = [session / name for name in self.setup["cages"] if (session / name).is_dir()]
        if not cages:
            raise ValueError(f"No configured cage/box directories found in {session}")
        if all((cage / "videos-raw").is_dir() and (cage / "calibration").is_dir() for cage in cages):
            return
        cal = self.setup["calibration"]
        restructure_and_downsample(
            session_folder=session, out_fps=cal["fps"],
            experiment=self.setup["experiment"],
            board_size=tuple(cal["board_size"]),
            detection_samples=cal["detection_samples"],
        )

    def downsample(self, session_dir: str | Path) -> list[Path]:
        """Resume downsampling for already organized calibration folders."""
        session = Path(session_dir)
        fps = self.setup["calibration"]["fps"]
        processed = []
        for name in self.setup["cages"]:
            calibration = session / name / "calibration"
            if not calibration.is_dir():
                continue
            if (calibration / "calibration.toml").is_file():
                continue  # Never change videos behind a completed camera calibration.
            videos = [p for p in calibration.iterdir()
                      if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
            if videos and all(p.stem.endswith(f"_{fps}fps") for p in videos):
                continue
            if any(re.search(r"_\d+(?:\.\d+)?fps$", p.stem) for p in videos):
                raise RuntimeError(
                    f"Calibration videos at a different FPS already exist in {calibration}. "
                    "Move those copies aside before downsampling from originals."
                )
            originals = calibration / "originals"
            if videos or (originals.is_dir() and any(p.suffix.lower() in VIDEO_EXTS for p in originals.iterdir())):
                downsample_dir(
                    calibration_dir=calibration, out_fps=fps,
                    require_annotation=self.setup["experiment"] == "mechanical_lockbox",
                )
                processed.append(calibration)
        return processed
