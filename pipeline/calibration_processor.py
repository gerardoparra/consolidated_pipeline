"""Camera calibration and optional manual-label reprojection checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .anipose_runner import AniposeRunner
from .calibration_prep import VIDEO_EXTS


@dataclass
class CalibrationResult:
    tomls: list[Path] = field(default_factory=list)
    skipped_cages: dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    status: str
    summary_csv: Path | None = None
    mean_error_px: float | None = None


class CalibrationProcessor(AniposeRunner):
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
        """Check calibration by triangulating manual points and reprojecting them."""
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
