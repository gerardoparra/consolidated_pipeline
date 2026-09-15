"""Recalibrate one copied MLB2 five-camera set in a temporary project.

Run: python tests/smoke_calibration.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import Setup


def main() -> None:
    repository = Path(__file__).resolve().parent.parent
    setup = Setup.load(repository / "config" / "setup.json")
    source_root = setup.experiment_dir("local")
    source = next(
        (folder for folder in source_root.rglob("calibration")
         if folder.parent.name == "CAGE2" and len(list(folder.glob("*.avi"))) >= 5),
        None,
    )
    if source is None:
        raise RuntimeError("No complete CAGE2 calibration-video set found")
    with tempfile.TemporaryDirectory(prefix="calibration_smoke_", dir=repository) as temporary:
        root = Path(temporary)
        destination = root / "smoke_session" / "CAGE2" / "calibration"
        destination.mkdir(parents=True)
        (destination.parent / "videos-raw").mkdir()
        for video in sorted(source.glob("*.avi")):
            shutil.copy2(video, destination / video.name)
        setup.materialize(root, "local")
        command = setup.host("local")["anipose_command"] + ["calibrate"]
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
        output = destination / "calibration.toml"
        if completed.returncode or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(
                f"Anipose calibration smoke failed (exit {completed.returncode}):\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        print(f"Anipose calibrated {len(list(source.glob('*.avi')))} copied videos from {source}")


if __name__ == "__main__":
    main()
