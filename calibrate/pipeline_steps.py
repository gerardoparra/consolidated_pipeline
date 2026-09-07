"""
pipeline_steps.py
=================
Input/output dataclasses for each step of the video calibration pipeline.

Each step has a pair of dataclasses:
  - <Step>Input  : what the step receives
  - <Step>Output : what the step produces

The output of one step is intended to feed directly into the input of the next:

    PrepareInput -> [prepare] -> PrepareOutput
                                      |
                                 CalibrateInput -> [calibrate] -> CalibrateOutput
                                                                        |
                                                               EvaluateInput -> [evaluate] -> EvaluateOutput
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1: Prepare
# ---------------------------------------------------------------------------

@dataclass
class PrepareInput:
    """Inputs for the prepare step.

    Parameters
    ----------
    zip_path : Path
        Path to the downloaded .zip session archive.
    experiment_dir : Path
        Root directory for the experiment. The session folder will be
        extracted here.
    calibration_fps : float
        Target frame rate for downsampled calibration videos.
    box_subfolders : list[str]
        Names of box subfolders expected inside the session (e.g. BOX2, BOX3, BOX4).
    session_name : str, optional
        Override for the session folder name inside target_root.
        Defaults to the zip file stem (e.g. ``2026-02-26_session1``).
    """

    zip_path: Path
    experiment_dir: Path
    calibration_fps: float
    box_subfolders: list[str]
    session_name: str | None = None

    def __post_init__(self) -> None:
        self.zip_path = Path(self.zip_path)
        self.experiment_dir = Path(self.experiment_dir)


@dataclass
class PrepareOutput:
    """Outputs produced by the prepare step.

    Parameters
    ----------
    session_dir : Path
        Root directory of the extracted and restructured session.
    calibration_dirs : list[Path]
        One calibration directory per box (contains downsampled videos).
    raw_video_dirs : list[Path]
        One videos-raw directory per box (contains experiment videos).
    """

    session_dir: Path
    calibration_dirs: list[Path]
    raw_video_dirs: list[Path]


# ---------------------------------------------------------------------------
# Step 2: Calibrate
# ---------------------------------------------------------------------------

@dataclass
class CalibrateInput:
    """Inputs for the calibrate step.

    Parameters
    ----------
    session_dir : Path
        Root directory of the session (must contain box subfolders with
        a ``calibration/`` directory each holding downsampled videos).
    anipose_config_path : Path, optional
        Path to a custom Anipose ``config.toml``. When ``None`` the bundled
        example config is used.
    """

    session_dir: Path
    anipose_config_path: Path | None = None

    def __post_init__(self) -> None:
        self.session_dir = Path(self.session_dir)
        if self.anipose_config_path is not None:
            self.anipose_config_path = Path(self.anipose_config_path)


@dataclass
class CalibrateOutput:
    """Outputs produced by the calibrate step.

    Parameters
    ----------
    calibration_tomls : list[Path]
        Paths to the ``calibration.toml`` files produced by Anipose,
        one per box subfolder.
    """

    calibration_tomls: list[Path]


# ---------------------------------------------------------------------------
# Step 3: Evaluate
# ---------------------------------------------------------------------------

@dataclass
class EvaluateInput:
    """Inputs for the evaluate step.

    Parameters
    ----------
    calibration_tomls : list[Path]
        Paths to the ``calibration.toml`` files to evaluate.
    label_csv_folder : Path
        Directory containing DLC-style manual label CSVs.
    output_root : Path
        Root directory where figures and summary CSV will be written.
    reprojection_error_threshold_px : float
        Maximum acceptable mean reprojection error in pixels.
    image_size : tuple[int, int]
        (width, height) of camera images, used for reprojection plots.
    """

    calibration_tomls: list[Path]
    label_csv_folder: Path
    output_root: Path
    reprojection_error_threshold_px: float
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        self.label_csv_folder = Path(self.label_csv_folder)
        self.output_root = Path(self.output_root)


@dataclass
class EvaluateOutput:
    """Outputs produced by the evaluate step.

    Parameters
    ----------
    summary_csv : Path
        Path to the written ``reprojection_summary.csv``.
    mean_error_px : float
        Mean reprojection error across all calibrations and label CSVs.
    passed : bool
        True if mean_error_px is below the configured threshold.
    """

    summary_csv: Path
    mean_error_px: float
    passed: bool
