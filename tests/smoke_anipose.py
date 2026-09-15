"""Real Anipose smoke check using an existing MLB2 calibration and synthetic 2D poses.

Run: python tests/smoke_anipose.py
The command creates and removes a temporary project inside this repository.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import Setup


def main() -> None:
    repository = Path(__file__).resolve().parent.parent
    setup = Setup.load(repository / "config" / "setup.json")
    source_root = setup.experiment_dir("local")
    calibration_source = next(
        (path for path in source_root.rglob("calibration.toml")
         if path.parent.parent.name == "CAGE2"), None
    )
    if calibration_source is None:
        raise RuntimeError("No CAGE2 calibration.toml found in the configured MLB2 experiment")
    with calibration_source.open("rb") as handle:
        cameras = tomllib.load(handle)

    with tempfile.TemporaryDirectory(prefix="anipose_smoke_", dir=repository) as temporary:
        root = Path(temporary)
        cage = root / "smoke_session" / "CAGE2"
        (cage / "calibration").mkdir(parents=True)
        (cage / "videos-raw").mkdir()
        (cage / "pose-2d").mkdir()
        shutil.copy2(calibration_source, cage / "calibration" / "calibration.toml")
        setup.materialize(root, "local")

        points_3d = np.array([[float(i * 3), float(i * 2), 0.0] for i in range(12)], dtype=np.float64)
        columns = pd.MultiIndex.from_product(
            [["mlb2_superanimal"], ["nose"], ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        for camera in (value for key, value in cameras.items() if key.startswith("cam_")):
            serial = camera["name"]
            projected, _ = cv2.projectPoints(
                points_3d,
                np.asarray(camera["rotation"], dtype=np.float64),
                np.asarray(camera["translation"], dtype=np.float64),
                np.asarray(camera["matrix"], dtype=np.float64),
                np.asarray(camera["distortions"], dtype=np.float64),
            )
            xy = projected.reshape(-1, 2)
            frame = pd.DataFrame(
                np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), 0.99)]),
                columns=columns,
            )
            stem = f"20260803_100000_camera{serial}"
            (cage / "videos-raw" / f"{stem}.mkv").write_bytes(b"fixture")
            frame.to_hdf(cage / "pose-2d" / f"{stem}.h5", key="df")

        command = setup.host("local")["anipose_command"] + ["triangulate"]
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
        output = cage / "pose-3d" / "20260803_100000_.csv"
        if completed.returncode or not output.is_file():
            raise RuntimeError(
                f"Anipose smoke failed (exit {completed.returncode}):\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        result = pd.read_csv(output)
        if len(result) != len(points_3d) or "nose_x" not in result.columns:
            raise RuntimeError(f"Unexpected 3D result: {output}")
        print(f"Anipose triangulated {len(result)} frames using {calibration_source}")


if __name__ == "__main__":
    main()
