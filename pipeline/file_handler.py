"""Session import and video/output discovery."""

from __future__ import annotations

import shutil
from pathlib import Path

from .calibration_prep import VIDEO_EXTS, extract_or_prepare_folder
from .pose_inference import has_existing_track


class FileHandler:
    def __init__(self, experiment_dir: str | Path, setup: dict):
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.setup = setup

    def is_prepared(self, session_dir: Path) -> bool:
        cages = [session_dir / name for name in self.setup["cages"]]
        present = [cage for cage in cages if cage.is_dir()]
        return bool(present) and all(
            (cage / "videos-raw").is_dir() and (cage / "calibration").is_dir()
            for cage in present
        )

    def import_raw_data(self, source_path: str | Path) -> Path:
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Session input does not exist: {source}")
        if source.is_dir() and self.is_prepared(source) and source.parent == self.experiment_dir:
            return source
        destination = self.experiment_dir / source.stem
        if destination.exists():
            if self.is_prepared(destination):
                return destination
            if destination == source:
                return destination  # Raw folder already lives in the experiment root.
            raise FileExistsError(f"Session destination exists but is not prepared: {destination}")
        if source.is_file() and source.suffix.lower() != ".zip":
            raise ValueError(f"Expected a zip archive or session directory: {source}")
        if source.is_dir():
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            return destination
        return extract_or_prepare_folder(
            zip_path=str(source), target_dir=str(self.experiment_dir), unzip=True
        )

    def cage_dirs(self, session_dir: str | Path) -> list[Path]:
        root = Path(session_dir)
        return [root / name for name in self.setup["cages"] if (root / name).is_dir()]

    def raw_videos(self, session_dir: str | Path) -> list[Path]:
        suffix = "." + self.setup["pose"]["video_extension"].lstrip(".").lower()
        return sorted(
            video for cage in self.cage_dirs(session_dir)
            for video in (cage / "videos-raw").glob("*")
            if video.is_file() and video.suffix.lower() == suffix
        )

    def pending_tracks(self, session_dir: str | Path) -> list[Path]:
        return [video for video in self.raw_videos(session_dir)
                if not has_existing_track(video, video.parent.parent / "tracks")]

    def calibration_tomls(self, session_dir: str | Path) -> list[Path]:
        return sorted(
            cage / "calibration" / "calibration.toml"
            for cage in self.cage_dirs(session_dir)
            if (cage / "calibration" / "calibration.toml").is_file()
        )
