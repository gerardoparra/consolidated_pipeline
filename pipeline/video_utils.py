"""
video_utils.py
==============
Utility functions for video frame extraction and file discovery used across
the Rat Lockbox calibration pipeline.

Functions
---------
extract_frame
    Extract a single frame from each of a list of camera videos and write
    the result as a JPEG under the DLC ``labeled-data/`` directory tree.
find_video_files
    Glob for raw experiment videos across a set of named box subfolders.
"""

import cv2
import glob
import re
from pathlib import Path


def extract_frame(wd, video_files, frame_index):
    """Extract one frame from each camera video and save it for DLC labeling.

    For every video in *video_files* the frame at *frame_index* is decoded and
    saved as a JPEG under ``<wd>/labeled-data/<date_str>/``.  The output
    filename is ``<camera_number>_img<frame_index:05d>.jpg``.

    The function expects video filenames to follow the Rat Lockbox convention::

        <YYYY-MM-DD_HH-MM-SS>_<camera_serial>[_...].avi

    If a filename does not match, the first two underscore-separated tokens are
    used as the date string and the third token (if present) as the camera
    identifier.

    Parameters
    ----------
    wd : str or Path
        Working directory — typically the root of a DLC project.  A
        ``labeled-data/`` sub-directory will be created here.
    video_files : list[str | Path]
        Paths to the camera video files from which to extract the frame.
    frame_index : int
        Zero-based index of the frame to extract.

    Returns
    -------
    set[Path]
        The set of output directories that were written to (one per unique date
        string found in the video filenames).
    """
    labeled_data_dir = Path(wd) / "labeled-data"
    # Matches filenames like: 2026-02-25_10-43-38_12345678.avi
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)")

    out_dirs = set()
    for video_path_str in video_files:
        video_path = Path(video_path_str)
        video_stem = video_path.stem

        match = pattern.match(video_stem)
        if match:
            date_str, camera_number = match.groups()
        else:
            parts = video_stem.split("_")
            date_str = "_".join(parts[:2]) if len(parts) >= 2 else video_stem
            camera_number = parts[2] if len(parts) >= 3 else "cam"

        out_dir = labeled_data_dir / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.add(out_dir)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            print(f"Could not read frame {frame_index} from: {video_path}")
            continue

        out_file = out_dir / f"{camera_number}_img{frame_index:05d}.jpg"
        cv2.imwrite(str(out_file), frame)
        print(f"Saved: {out_file}")
        
    return out_dirs


def find_video_files(experiment_directory, subfolders, ext="avi"):
    """Return a flat list of raw video files across multiple box subfolders.

    Walks ``<experiment_directory>/<subfolder>/videos-raw/`` for each entry in
    *subfolders* and collects every file matching ``*.{ext}``.

    Parameters
    ----------
    experiment_directory : str or Path
        Root directory of a single recording session (contains the box
        subfolders, e.g. BOX2, BOX3, BOX4).
    subfolders : list[str]
        Names of box subfolders to search, e.g. ``['BOX2', 'BOX3', 'BOX4']``.
    ext : str, optional
        Video file extension to glob for (default ``'avi'``).

    Returns
    -------
    list[str]
        Sorted list of absolute paths to all matching video files.
    """
    video_files = []
    for folder in subfolders:
        folder_path = f"{experiment_directory}//{folder}//videos-raw//"
        video_files.extend(glob.glob(folder_path + f"*.{ext}"))
    return video_files