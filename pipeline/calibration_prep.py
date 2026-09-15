#!/usr/bin/env python3
"""
prepare_calibration_dataset.py
==============================
CLI tool and importable library for preparing calibration video datasets in
the Rat Lockbox project.

Given a downloaded ``.zip`` archive the script:

1. Extracts the archive into ``target_dir``.
2. Scans each box subfolder (e.g. BOX2, BOX3, BOX4), groups files by their
   leading timestamp, and selects the first timestamp group that contains at
   least 5 videos and 1 ``.txt`` annotation file as the calibration set.
3. Moves calibration files to ``<box>/calibration/`` and the remaining
   experiment videos to ``<box>/videos-raw/``.
4. Downsamples each calibration video to the requested FPS using OpenCV,
   placing the full-framerate originals in ``<box>/calibration/originals/``.

Usage (CLI)
-----------
    python prepare_calibration_dataset.py <zip_path> <target_dir> [--fps 2.0]

Usage (library)
---------------
    from pipeline.calibration_prep import (
        extract_zip, restructure_and_downsample, downsample_dir
    )
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}


@dataclass
class VideoMeta:
    """Metadata extracted from a single video file."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float  # seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unzip video dataset, build calibration/videos-raw folders, "
            "downsample calibration videos to 2 fps, and archive originals."
        )
    )
    parser.add_argument("zip_path", type=Path, help="Path to input .zip file")
    parser.add_argument(
        "target_dir", type=Path, help="Directory where extracted folder will be placed"
    )
    parser.add_argument(
        "--fps", type=float, default=2.0, help="Output FPS for calibration videos"
    )
    parser.add_argument(
        "--experiment",
        choices=("mechanical_lockbox", "sliding_lockbox"),
        default="mechanical_lockbox",
        help="Input directory layout (default: mechanical_lockbox)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        nargs=2,
        metavar=("COLUMNS", "ROWS"),
        default=(10, 7),
        help="Checkerboard inner-corner dimensions used as a detection fallback",
    )
    parser.add_argument(
        "--detection-samples",
        type=int,
        default=12,
        help="Number of evenly spaced frames inspected in each video",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete temporary extraction folder",
    )
    return parser.parse_args()


def get_timestamp_prefix(filename: str) -> str | None:
    """Return the leading ``YYYY-MM-DD_HH-MM-SS`` timestamp from *filename*, or ``None``."""
    match = TIMESTAMP_RE.match(filename)
    if not match:
        return None
    return match.group(1)


def extract_or_prepare_folder(
    zip_path: str,
    target_dir: str,
    unzip: bool = True,
    keep_temp: bool = False,
) -> Path:
    """Extract *zip_path* into ``target_dir / <stem>`` and return the destination.

    If the archive (or source folder, when *unzip* is ``False``) contains a
    single top-level directory it is moved directly to the destination;
    otherwise a new directory with the stem name is created and all
    top-level entries are moved into it.

    Parameters
    ----------
    zip_path : str or Path
        Path to the ``.zip`` archive, or to an already-extracted folder when
        *unzip* is ``False``.
    target_dir : str or Path
        Parent directory that will receive the extracted folder.
    unzip : bool
        When ``True`` (default), *zip_path* is treated as a zip archive and
        extracted to a temporary directory first. When ``False``, unzipping
        is skipped and *zip_path* is treated as a directory whose contents
        are organized directly into *target_dir*.
    keep_temp : bool
        When ``True`` the temporary extraction directory is preserved.
        Useful for debugging. Only relevant when *unzip* is ``True``.

    Returns
    -------
    Path
        Path to the extracted/organized session folder inside *target_dir*.

    Raises
    ------
    FileExistsError
        If the destination directory already exists.
    """
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / zip_path.stem
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    if unzip:
        with zipfile.ZipFile(zip_path, "r") as zf:
            temp_dir = Path(tempfile.mkdtemp(prefix="rat_lockbox_unzip_"))
            zf.extractall(temp_dir)
    else:
        temp_dir = zip_path

    entries = [p for p in temp_dir.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        shutil.move(str(entries[0]), str(destination))
    else:
        destination.mkdir(parents=True, exist_ok=False)
        for entry in entries:
            shutil.move(str(entry), str(destination / entry.name))

    if unzip and temp_dir.exists() and not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return destination


def extract_zip(zip_path: str | Path, target_dir: str | Path,
                keep_temp: bool = False) -> Path:
    """Compatibility name for extracting a session zip."""
    return extract_or_prepare_folder(str(zip_path), str(target_dir), unzip=True, keep_temp=keep_temp)


def collect_files(folder: Path) -> tuple[list[Path], list[Path]]:
    """Return ``(videos, txts)`` found directly inside *folder*, sorted by name."""
    videos = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda p: p.name,
    )
    txts = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: p.name,
    )
    return videos, txts


def select_calibration_files(videos: list[Path], txts: list[Path]) -> tuple[list[Path], Path]:
    """Select the calibration video set and annotation file from a flat file list.

    Groups *videos* and *txts* by their leading timestamp prefix and returns
    the first timestamp group (chronologically) that contains at least 5 videos
    and at least 1 ``.txt`` file.

    Parameters
    ----------
    videos : list[Path]
        All video files found in the box folder.
    txts : list[Path]
        All ``.txt`` files found in the box folder.

    Returns
    -------
    tuple[list[Path], Path]
        ``(calibration_videos[:5], calibration_txt)``.

    Raises
    ------
    ValueError
        If no valid timestamp group is found.
    """
    grouped_videos: dict[str, list[Path]] = {}
    grouped_txts: dict[str, list[Path]] = {}

    for video in videos:
        ts = get_timestamp_prefix(video.name)
        if ts is not None:
            grouped_videos.setdefault(ts, []).append(video)
    for txt in txts:
        ts = get_timestamp_prefix(txt.name)
        if ts is not None:
            grouped_txts.setdefault(ts, []).append(txt)

    valid_timestamps = sorted(
        [
            ts
            for ts in grouped_videos
            if len(grouped_videos.get(ts, [])) >= 5 and len(grouped_txts.get(ts, [])) >= 1
        ]
    )
    if not valid_timestamps:
        raise ValueError(
            "Could not find a timestamp group with at least 5 videos and 1 txt file."
        )

    ts = valid_timestamps[0]
    return grouped_videos[ts][:5], grouped_txts[ts][0]


def calibration_target_detection_rate(
    video_path: Path,
    board_size: tuple[int, int],
    sample_count: int = 12,
) -> float:
    """Return the fraction of frames containing a ChArUco or checkerboard target.

    ChArUco detection uses the 4x4_50 dictionary configured for the current
    experiment. Plain checkerboard detection is retained as a fallback.
    """
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        cap.release()
        raise RuntimeError(f"Invalid frame count for video: {video_path}")

    if sample_count == 1:
        frame_indices = [frame_count // 2]
    else:
        last_frame = frame_count - 1
        frame_indices = sorted(
            set(
                int(round(i * last_frame / (sample_count - 1)))
                for i in range(sample_count)
            )
        )

    detections = 0
    inspected = 0
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    aruco = getattr(cv2, "aruco", None)
    aruco_dictionary = None
    if aruco is not None:
        dictionary_id = getattr(aruco, "DICT_4X4_50", None)
        if dictionary_id is not None:
            if hasattr(aruco, "getPredefinedDictionary"):
                aruco_dictionary = aruco.getPredefinedDictionary(dictionary_id)
            else:
                aruco_dictionary = aruco.Dictionary_get(dictionary_id)

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        inspected += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        target_found = False
        if aruco_dictionary is not None:
            _, marker_ids, _ = aruco.detectMarkers(gray, aruco_dictionary)
            target_found = marker_ids is not None and len(marker_ids) >= 4

        if not target_found:
            target_found, _ = cv2.findChessboardCorners(gray, board_size, flags)

        if target_found:
            detections += 1

    cap.release()
    if inspected == 0:
        raise RuntimeError(f"Could not read sampled frames from: {video_path}")
    return detections / inspected


def classify_sliding_lockbox_videos(
    videos: list[Path],
    board_size: tuple[int, int] = (10, 7),
    sample_count: int = 12,
    min_detected_cameras: int = 3,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Classify name-sorted five-camera blocks using calibration-target detection.

    A complete block is calibration when the ChArUco/checkerboard target is found at least
    once in ``min_detected_cameras`` videos. A block with fewer non-zero
    detections is considered ambiguous and returned as a candidate. Blocks
    without detections are raw experiment videos. Incomplete final blocks are
    never accepted automatically.
    """
    if not 1 <= min_detected_cameras <= 5:
        raise ValueError("min_detected_cameras must be between 1 and 5")

    calibration: list[Path] = []
    candidates: list[Path] = []
    raw: list[Path] = []

    for start in range(0, len(videos), 5):
        group = videos[start : start + 5]
        rates = [
            calibration_target_detection_rate(video, board_size, sample_count)
            for video in group
        ]
        detected_cameras = sum(rate > 0 for rate in rates)
        rates_text = ", ".join(
            f"{video.name}={rate:.0%}" for video, rate in zip(group, rates)
        )
        print(f"Calibration-target scan: {rates_text}")

        if len(group) == 5 and detected_cameras >= min_detected_cameras:
            calibration.extend(group)
        elif detected_cameras:
            candidates.extend(group)
        else:
            raw.extend(group)

    return calibration, candidates, raw


def get_video_meta(video_path: Path) -> VideoMeta:
    """Open *video_path* with OpenCV and return a populated :class:`VideoMeta`."""
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count for video: {video_path}")

    duration = frame_count / fps
    return VideoMeta(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration=duration,
    )


def build_sampling_times(metas: list[VideoMeta], out_fps: float) -> list[float]:
    """Compute evenly-spaced sample timestamps (seconds) limited to the shortest video.

    Uses the minimum duration across all clips so that the same set of
    wall-clock times is valid for every camera.

    Parameters
    ----------
    metas : list[VideoMeta]
        Metadata for all calibration videos (one per camera).
    out_fps : float
        Desired output frame rate.

    Returns
    -------
    list[float]
        Monotonically increasing timestamps in seconds.
    """
    min_duration = min(meta.duration for meta in metas)
    step = 1.0 / out_fps
    times: list[float] = []
    t = 0.0
    while t < min_duration:
        times.append(t)
        t += step
    if not times:
        raise RuntimeError("No frames available after applying output FPS.")
    return times


def downsample_video(meta: VideoMeta, times: list[float], out_path: Path, out_fps: float) -> None:
    """Write a subsampled copy of *meta.path* at the timestamps in *times*.

    Parameters
    ----------
    meta : VideoMeta
        Source video metadata (path, fps, dimensions).
    times : list[float]
        Wall-clock timestamps (seconds) of frames to include in the output.
    out_path : Path
        Destination file path.  Extension must be ``.avi`` or ``.mp4``.
    out_fps : float
        Playback frame rate for the output video.
    """
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

    cap = cv2.VideoCapture(str(meta.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {meta.path}")

    # in downsample_video(...)
    ext = out_path.suffix.lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # or "MJPG"
    elif ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        raise RuntimeError(f"Unsupported output extension: {ext}")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (meta.width, meta.height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer: {out_path}")

    for t in times:
        frame_idx = int(round(t * meta.fps))
        frame_idx = min(frame_idx, meta.frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            writer.release()
            cap.release()
            print(f"Failed reading frame {frame_idx} from {meta.path}")
        writer.write(frame)

    writer.release()
    cap.release()


def downsample_dir(
    calibration_dir: Path, out_fps: float, require_annotation: bool = True
) -> None:
    """Downsample all calibration videos in *calibration_dir* in-place.

    Moves the original calibration videos and the ``.txt`` annotation file
    into a ``calibration_dir/originals/`` sub-directory, then writes
    downsampled copies back to *calibration_dir* with the suffix
    ``_<fps>fps`` appended to each stem.

    If no videos are found directly in *calibration_dir*, this will check
    for a pre-existing ``calibration_dir/originals/`` folder and, if videos
    are found there, move them up into *calibration_dir* before proceeding
    as normal.

    Parameters
    ----------
    calibration_dir : Path
        Directory containing one or more complete five-camera calibration
        groups and, for the old layout, one ``.txt`` file.
    out_fps : float
        Target frame rate for the downsampled videos.
    require_annotation : bool
        Whether exactly one ``.txt`` annotation file is required. The older
        mechanical-lockbox layout includes one; the sliding-lockbox layout
        contains videos only.

    Raises
    ------
    RuntimeError
        If the expected number of videos or annotation files is not found.
    """
    originals_dir = calibration_dir / "originals"

    current_cal_videos = sorted(
        [p for p in calibration_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda p: p.name,
    )

    if not current_cal_videos and originals_dir.is_dir():
        originals_videos = sorted(
            [p for p in originals_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
            key=lambda p: p.name,
        )
        if originals_videos:
            originals_txts = sorted(
                [p for p in originals_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
                key=lambda p: p.name,
            )
            for f in originals_videos + originals_txts:
                shutil.move(str(f), str(calibration_dir / f.name))

            current_cal_videos = sorted(
                [p for p in calibration_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
                key=lambda p: p.name,
            )

    originals_dir.mkdir(exist_ok=True)

    current_cal_txts = sorted(
        [p for p in calibration_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: p.name,
    )
    if len(current_cal_videos) < 5 or len(current_cal_videos) % 5:
        raise RuntimeError(
            f"Expected a positive multiple of 5 calibration videos in "
            f"{calibration_dir}, found {len(current_cal_videos)}."
        )
    expected_txts = 1 if require_annotation else 0
    if len(current_cal_txts) != expected_txts:
        raise RuntimeError(
            f"Expected {expected_txts} calibration txt files in {calibration_dir}, "
            f"found {len(current_cal_txts)}."
        )

    for f in current_cal_videos + current_cal_txts:
        shutil.move(str(f), str(originals_dir / f.name))

    metas = [get_video_meta(p) for p in sorted(originals_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    sampling_times = build_sampling_times(metas, out_fps=out_fps)

    for meta in metas:
        out_name = f"{meta.path.stem}_{out_fps}fps{meta.path.suffix.lower()}"  # preserves original container format
        #change extension to .avi
        out_path = calibration_dir / out_name
        out_path = out_path.with_suffix(".avi")
        downsample_video(meta=meta, times=sampling_times, out_path=out_path, out_fps=out_fps)
        print(f"Created downsampled calibration video: {out_path}")


def restructure_and_downsample(
    session_folder: Path,
    out_fps: float,
    experiment: str = "mechanical_lockbox",
    board_size: tuple[int, int] = (10, 7),
    detection_samples: int = 12,
) -> None:
    """Restructure all box subfolders in *session_folder* and downsample calibration videos.

    For ``mechanical_lockbox``, each subfolder is expected to use the old
    timestamp-grouped layout (BOX2, BOX3, BOX4). For ``sliding_lockbox``, the
    session is expected to contain CAGE1 and CAGE2. Name-sorted blocks of five
    videos are classified by sampling frames and detecting the calibration target.

    * Creates ``calibration/`` and ``videos-raw/`` subdirectories.
    * Moves the first valid group of 5 calibration videos + 1 ``.txt`` file
      into ``calibration/``.
    * Moves all remaining videos and text files into ``videos-raw/``.
    * Calls :func:`downsample_dir` to produce low-FPS copies.

    Parameters
    ----------
    session_folder : Path
        Root directory of a single recording session.
    out_fps : float
        Target frame rate for downsampled calibration videos.
    experiment : {"mechanical_lockbox", "sliding_lockbox"}
        Selects how calibration files are identified.
    board_size : tuple[int, int]
        Checkerboard inner-corner dimensions used as a fallback when no
        ChArUco markers are found.
    detection_samples : int
        Number of evenly spaced frames inspected per sliding-lockbox video.
    """
    if experiment not in ("mechanical_lockbox", "sliding_lockbox"):
        raise ValueError(f"Unsupported experiment: {experiment}")

    subfolders = sorted([p for p in session_folder.iterdir() if p.is_dir()], key=lambda p: p.name)
    expected_subfolders = 3 if experiment == "mechanical_lockbox" else 2
    if len(subfolders) != expected_subfolders:
        print(
            f"Warning: expected {expected_subfolders} subfolders for {experiment}, "
            f"found {len(subfolders)} in {session_folder}. "
            "Proceeding anyway."
        )

    for subfolder in subfolders:
        videos, txts = collect_files(subfolder)
        if not videos:
            print(f"Skipping {subfolder}: no videos found.")
            continue

        if experiment == "mechanical_lockbox":
            calibration_videos, calibration_txt = select_calibration_files(videos, txts)
            calibration_txts = [calibration_txt]
            raw_txts = [t for t in txts if t != calibration_txt]
        else:
            if len(videos) < 5:
                raise ValueError(
                    f"Expected at least 5 videos in {subfolder}, found {len(videos)}."
                )
            calibration_videos, candidate_videos, raw_videos = (
                classify_sliding_lockbox_videos(
                    videos,
                    board_size=board_size,
                    sample_count=detection_samples,
                )
            )
            calibration_txts = []
            raw_txts = txts

        calibration_dir = subfolder / "calibration"
        videos_raw_dir = subfolder / "videos-raw"
        calibration_dir.mkdir(exist_ok=True)
        videos_raw_dir.mkdir(exist_ok=True)

        if experiment == "mechanical_lockbox":
            cal_set = set(calibration_videos)
            raw_videos = [v for v in videos if v not in cal_set]
            candidate_videos = []

        candidates_dir = subfolder / "calibration-candidates"
        if candidate_videos:
            candidates_dir.mkdir(exist_ok=True)

        for f in raw_videos + raw_txts:
            shutil.move(str(f), str(videos_raw_dir / f.name))

        for f in calibration_videos + calibration_txts:
            shutil.move(str(f), str(calibration_dir / f.name))

        for f in candidate_videos:
            shutil.move(str(f), str(candidates_dir / f.name))

        if calibration_videos:
            downsample_dir(
                calibration_dir=calibration_dir,
                out_fps=out_fps,
                require_annotation=experiment == "mechanical_lockbox",
            )
        elif experiment == "sliding_lockbox":
            print(
                f"Warning: no calibration groups confidently detected in {subfolder}. "
                "Review calibration-candidates manually."
            )

        print(f"Finished: {subfolder}")


def main() -> None:
    args = parse_args()
    zip_path = args.zip_path.resolve()
    target_dir = args.target_dir.resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    extracted_folder = extract_zip(
        zip_path=zip_path, target_dir=target_dir, keep_temp=args.keep_temp
    )
    print(f"Extracted to: {extracted_folder}")

    restructure_and_downsample(
        session_folder=extracted_folder,
        out_fps=args.fps,
        experiment=args.experiment,
        board_size=tuple(args.board_size),
        detection_samples=args.detection_samples,
    )
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
