"""
config.py
=========
Pipeline configuration for the video calibration pipeline.

Load from a TOML file with:
    from video_calibration.config import load_config
    cfg = load_config("pipeline.toml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomli as tomllib


@dataclass
class PipelineConfig:
    """Top-level configuration for the calibration pipeline.

    Parameters
    ----------
    experiment_dir : Path
        Root directory for the experiment. Downloaded zip archives and
        extracted session folders both live here.
    label_csv_root : Path
        Root folder containing per-session DLC label CSV subfolders.
    calibration_fps : float
        Target frame rate for downsampled calibration videos (default 2.0).
    reprojection_error_threshold_px : float
        Maximum acceptable mean reprojection error in pixels.
        The evaluate stage will warn (and optionally gate) if exceeded.
    image_size : tuple[int, int]
        (width, height) of camera images, used for reprojection plots.
    box_subfolders : list[str]
        Names of box subfolders expected inside each session (e.g. BOX2, BOX3, BOX4).
    """

    experiment_dir: Path
    label_csv_root: Path
    calibration_fps: float = 2.0
    reprojection_error_threshold_px: float = 5.0
    image_size: tuple[int, int] = (1920, 1080)
    box_subfolders: list[str] = field(default_factory=lambda: ["BOX2", "BOX3", "BOX4"])

    def __post_init__(self) -> None:
        self.experiment_dir = Path(self.experiment_dir)
        self.label_csv_root = Path(self.label_csv_root)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a TOML file.

    Parameters
    ----------
    path : str or Path
        Path to the TOML configuration file.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    FileNotFoundError
        If the config file does not exist, with a hint to copy pipeline.toml.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Copy pipeline.toml, fill in your paths, and pass it with --config."
        )
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return PipelineConfig(**data)
