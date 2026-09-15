"""Camera calibration, quality evaluation, pose inference, and triangulation."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .calibration_prep import VIDEO_EXTS
from .conversion import ConversionResult, convert_session
from .pose_inference import discover_videos, run_inference


@dataclass
class CalibrationResult:
    tomls: list[Path] = field(default_factory=list)
    skipped_cages: dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    status: str
    summary_csv: Path | None = None
    mean_error_px: float | None = None


@dataclass
class TriangulationResult:
    outputs: list[Path] = field(default_factory=list)
    skipped_trials: dict[str, str] = field(default_factory=dict)


class PoseProcessor:
    def __init__(self, experiment_dir: str | Path, setup: dict, *, environment: str = "auto"):
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.setup = setup
        self.environment = "local" if environment == "auto" and os.name == "nt" else (
            "hpc" if environment == "auto" else environment
        )

    def _anipose_command(self) -> list[str]:
        command = self.setup["hosts"][self.environment]["anipose_command"]
        if not isinstance(command, list) or not command:
            raise ValueError("Host anipose_command must be a non-empty list")
        executable = str(command[0])
        if Path(executable).is_absolute():
            if not Path(executable).is_file():
                raise FileNotFoundError(f"Anipose executable is missing: {executable}")
        elif shutil.which(executable) is None:
            raise RuntimeError(f"Anipose command {executable!r} is unavailable on this host")
        return [str(part) for part in command]

    def _run_anipose(self, action: str) -> None:
        command = self._anipose_command() + [action]
        try:
            completed = subprocess.run(
                command, cwd=self.experiment_dir, text=True, capture_output=True
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start Anipose {action}: {exc}") from exc
        if completed.returncode:
            raise RuntimeError(
                f"Anipose {action} failed (exit {completed.returncode}).\n"
                f"{completed.stderr or completed.stdout or 'No output from Anipose.'}"
            )

    def calibrate_cameras(self, session_dir: str | Path) -> CalibrationResult:
        session = Path(session_dir)
        result = CalibrationResult()
        calibratable = False
        for name in self.setup["cages"]:
            calibration = session / name / "calibration"
            if not calibration.is_dir():
                result.skipped_cages[name] = "Calibration directory is missing"
                continue
            toml = calibration / "calibration.toml"
            if toml.is_file() and toml.stat().st_size > 0:
                result.tomls.append(toml)
                continue
            videos = [p for p in calibration.iterdir()
                      if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
            if len(videos) < self.setup["anipose"]["num_cams"]:
                result.skipped_cages[name] = "No complete downsampled camera set"
                continue
            calibratable = True
        if calibratable:
            self._run_anipose("calibrate")
            result.tomls = sorted(
                session / name / "calibration" / "calibration.toml"
                for name in self.setup["cages"]
                if (session / name / "calibration" / "calibration.toml").is_file()
            )
            for name in self.setup["cages"]:
                if name not in result.skipped_cages and not (session / name / "calibration" / "calibration.toml").is_file():
                    result.skipped_cages[name] = "Anipose did not produce calibration.toml"
        return result

    def evaluate(self, session_dir: str | Path, calibration_tomls: list[Path],
                 label_csv_folder: str | Path | None = None) -> EvaluationResult:
        settings = self.setup["evaluation"]
        labels = label_csv_folder
        if labels is None and settings["label_csv_root"]:
            labels = Path(settings["label_csv_root"]) / Path(session_dir).name
        if labels is None:
            return EvaluationResult("unconfigured")
        folder = Path(labels)
        if not folder.is_dir() or not any(folder.glob("*.csv")):
            return EvaluationResult("pending_labels")
        if not calibration_tomls:
            return EvaluationResult("pending_calibration")
        import numpy as np
        try:
            from .reprojection import run_reprojection_batch
        except ImportError as exc:
            raise RuntimeError(
                "Reprojection evaluation needs the validation dependencies "
                "(install the project's validation extra)"
            ) from exc

        output_root = Path(settings["output_root"] or self.experiment_dir / "reprojection_outputs")
        summary, _ = run_reprojection_batch(
            calibration_inputs=calibration_tomls,
            label_csv_folder=folder,
            output_root=output_root,
            image_size=tuple(settings["image_size"]),
        )
        mean = float(np.nanmean(summary["mean_reprojection_error_px"]))
        status = "passed" if mean <= settings["threshold_px"] else "warning"
        return EvaluationResult(status, output_root / folder.name / "reprojection_summary.csv", mean)

    def detect_keypoints(self, video_root: str | Path | None = None,
                         result_root: str | Path | None = None,
                         extension: str | None = None,
                         inference_batch_size: int | None = None,
                         detector_batch_size: int | None = None,
                         adapt_batch_size: int | None = None) -> None:
        """Run the worker directly inside the DLC GPU container."""
        root = Path(video_root or self.experiment_dir)
        pose = self.setup["pose"]
        ext = extension or pose["video_extension"]
        videos = discover_videos(root, ext, "videos-raw")
        if not videos:
            print(f"No {ext} files found under {root}/.../videos-raw")
            return
        run_inference(
            videos, root, result_root or root, ext,
            skip_existing=pose["skip_existing"], cage_maps=self.setup["cages"],
            output_layout="sibling-tracks",
            inference_batch_size=inference_batch_size or pose["inference_batch_size"],
            detector_batch_size=detector_batch_size or pose["detector_batch_size"],
            adapt_batch_size=adapt_batch_size or pose["adapt_batch_size"],
            model_configs=pose["perspective_models"], model_name=pose["model_name"],
            detector_name=pose["detector_name"], max_individuals=pose["max_individuals"],
        )

    def convert_dlc_output_to_anipose(self, session_dir: str | Path) -> ConversionResult:
        return convert_session(
            session_dir, self.setup["cages"], self.setup["pose"]["video_extension"],
            self.setup["anipose"]["cam_regex"],
        )

    @staticmethod
    def _valid_pose3d(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            frame = pd.read_csv(path, nrows=1)
        except (OSError, ValueError, pd.errors.ParserError):
            return False
        columns = set(frame.columns)
        return (len(frame) == 1 and "fnum" in columns
                and any(name.endswith("_x") and name[:-2] + "_y" in columns
                        and name[:-2] + "_z" in columns for name in columns))

    def triangulate(self, session_dir: str | Path, conversion: ConversionResult) -> TriangulationResult:
        session = Path(session_dir)
        result = TriangulationResult(skipped_trials=dict(conversion.skipped_trials))
        pending: list[Path] = []
        for label in conversion.ready_trials:
            cage, trial = label.split("/", 1)
            output = session / cage / "pose-3d" / f"{trial}.csv"
            if self._valid_pose3d(output):
                result.outputs.append(output)
            else:
                if output.exists():
                    archived = output.with_name(output.name + ".invalid")
                    suffix = 1
                    while archived.exists():
                        archived = output.with_name(output.name + f".invalid.{suffix}")
                        suffix += 1
                    output.rename(archived)
                pending.append(output)
        if pending:
            self._run_anipose("triangulate")
            for output in pending:
                label = f"{output.parent.parent.name}/{output.stem}"
                if self._valid_pose3d(output):
                    result.outputs.append(output)
                else:
                    result.skipped_trials[label] = "Anipose did not create a 3D CSV"
        return result
