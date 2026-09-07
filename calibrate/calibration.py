"""
calibration.py
==============
Orchestrator for the video calibration pipeline.

Runs three steps in sequence:
  1. prepare   -- unzip, restructure, downsample calibration videos
  2. calibrate -- run Anipose calibration
  3. evaluate  -- compute reprojection error and check against threshold

Usage
-----
    python -m video_calibration.calibration session1.zip

Each step can also be called individually as a Python function.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from video_calibration.config import PipelineConfig, load_config
from video_calibration.pipeline_steps import (
    CalibrateInput,
    CalibrateOutput,
    EvaluateInput,
    EvaluateOutput,
    PrepareInput,
    PrepareOutput,
)


def prepare(inp: PrepareInput) -> PrepareOutput:
    """Unzip, restructure, and downsample calibration videos."""
    from video_calibration.prepare_calibration_dataset import (
        extract_zip,
        restructure_and_downsample,
    )

    session_dir = extract_zip(zip_path=inp.zip_path, target_dir=inp.experiment_dir)
    restructure_and_downsample(session_folder=session_dir, out_fps=inp.calibration_fps)

    calibration_dirs = sorted(
        session_dir / box / "calibration"
        for box in inp.box_subfolders
        if (session_dir / box / "calibration").is_dir()
    )
    raw_video_dirs = sorted(
        session_dir / box / "videos-raw"
        for box in inp.box_subfolders
        if (session_dir / box / "videos-raw").is_dir()
    )

    return PrepareOutput(
        session_dir=session_dir,
        calibration_dirs=calibration_dirs,
        raw_video_dirs=raw_video_dirs,
    )


_BUNDLED_ANIPOSE_CONFIG = Path(__file__).parent / "examples" / "anipose_config.toml"


def calibrate(inp: CalibrateInput) -> CalibrateOutput:
    """Run Anipose calibration on the prepared calibration directories.

    Copies the Anipose config to the session directory (if not already there),
    then runs ``anipose calibrate`` as a subprocess. Raises ``RuntimeError``
    if the process exits with a non-zero return code.

    If ``inp.anipose_config_path`` is ``None``, the bundled example config
    (``video_calibration/examples/anipose_config.toml``) is used.
    """
    import shutil
    import subprocess

    anipose_config_src = inp.anipose_config_path or _BUNDLED_ANIPOSE_CONFIG
    anipose_config_dest = inp.session_dir / "config.toml"
    if not anipose_config_dest.exists():
        shutil.copy2(anipose_config_src, anipose_config_dest)

    result = subprocess.run(
        ["anipose", "calibrate"],
        cwd=inp.session_dir,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"anipose calibrate failed with return code {result.returncode}.\n"
            "Check the output above for details."
        )

    calibration_tomls = sorted(
        p for p in inp.session_dir.rglob("calibration.toml")
        if p.parent.name.lower() == "calibration"
    )

    if not calibration_tomls:
        raise RuntimeError(
            f"anipose calibrate ran successfully but no calibration.toml files "
            f"were found under {inp.session_dir}."
        )

    return CalibrateOutput(calibration_tomls=calibration_tomls)


def evaluate(inp: EvaluateInput) -> EvaluateOutput:
    """Compute reprojection error and check against the configured threshold."""
    import numpy as np
    from video_calibration.reprojection_batch import run_reprojection_batch

    summary_df, _ = run_reprojection_batch(
        calibration_inputs=inp.calibration_tomls,
        label_csv_folder=inp.label_csv_folder,
        output_root=inp.output_root,
        image_size=inp.image_size,
    )

    mean_error_px = float(np.nanmean(summary_df["mean_reprojection_error_px"]))
    passed = mean_error_px <= inp.reprojection_error_threshold_px

    summary_csv = (
        inp.output_root
        / inp.label_csv_folder.name
        / "reprojection_summary.csv"
    )

    return EvaluateOutput(
        summary_csv=summary_csv,
        mean_error_px=mean_error_px,
        passed=passed,
    )


def run(
    zip_path: Path,
    cfg: PipelineConfig,
    session_name: str | None = None,
) -> EvaluateOutput | None:
    """Run the full calibration pipeline step end-to-end.

    Parameters
    ----------
    zip_path : Path
        Path to the downloaded .zip session archive.
    cfg : PipelineConfig
        Pipeline configuration (paths, thresholds, fps, etc.).
    session_name : str, optional
        Override for the session folder name. Defaults to the zip stem.

    Returns
    -------
    EvaluateOutput or None
        Reprojection summary, mean error, and pass/fail result.
        Returns ``None`` if no label CSVs exist yet (evaluate step is skipped).

    Notes
    -----
    ``label_csv_folder`` is derived as ``cfg.label_csv_root / zip_path.stem``.
    This assumes the label CSV subfolder is named after the zip file stem.
    If your naming differs, call :func:`evaluate` directly with a custom path.
    """
    # Derive session-level paths from config + zip stem.
    label_csv_folder = cfg.label_csv_root / zip_path.stem
    output_root = cfg.experiment_dir / "reprojection_outputs"

    prepare_out = prepare(PrepareInput(
        zip_path=zip_path,
        experiment_dir=cfg.experiment_dir,
        calibration_fps=cfg.calibration_fps,
        box_subfolders=cfg.box_subfolders,
        session_name=session_name,
    ))

    calibrate_out = calibrate(CalibrateInput(
        session_dir=prepare_out.session_dir,
    ))

    if not label_csv_folder.exists():
        print(
            f"[info] No label CSVs found at {label_csv_folder} — skipping evaluate.\n"
            "Label frames manually in DLC, then re-run with --evaluate-only."
        )
        return None

    evaluate_out = evaluate(EvaluateInput(
        calibration_tomls=calibrate_out.calibration_tomls,
        label_csv_folder=label_csv_folder,
        output_root=output_root,
        reprojection_error_threshold_px=cfg.reprojection_error_threshold_px,
        image_size=cfg.image_size,
    ))

    if not evaluate_out.passed:
        print(
            f"[warn] Reprojection error {evaluate_out.mean_error_px:.3f}px exceeds "
            f"threshold {cfg.reprojection_error_threshold_px}px."
        )

    return evaluate_out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the video calibration pipeline."
    )
    parser.add_argument(
        "zip_filename",
        type=str,
        help="Filename of the session .zip archive inside download_root (e.g. session1.zip)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline.toml"),
        help="Path to pipeline.toml (default: ./pipeline.toml)",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Override session folder name (defaults to zip stem)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    zip_path = cfg.experiment_dir / args.zip_filename

    result = run(
        zip_path=zip_path,
        cfg=cfg,
        session_name=args.session_name,
    )
    if result is not None:
        status = "PASSED" if result.passed else "FAILED"
        print(f"[{status}] Mean reprojection error: {result.mean_error_px:.3f}px")
        print(f"Summary: {result.summary_csv}")


if __name__ == "__main__":
    main()
