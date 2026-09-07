"""
video_calibration
=================
Tools for multi-camera calibration, reprojection error evaluation, and
frame management in the Rat Lockbox project.

Public API
----------
prepare_calibration_dataset
    extract_zip              -- unzip a downloaded recording archive
    restructure_and_downsample -- partition videos into calibration / raw and
                                  downsample calibration clips
    downsample_dir           -- downsample an already-structured calibration dir

reprojection_batch
    run_reprojection_batch   -- batch reprojection error across multiple
                                calibration files and manual label CSVs
    copy_dlc_label_csvs_for_date -- copy DLC labeled-data CSVs for a given date

video_utils
    extract_frame            -- extract a single frame from each camera video
                                and save as JPEG for DLC labeling
    find_video_files         -- glob for video files across experiment subfolders
"""

from .prepare_calibration_dataset import (
    extract_zip,
    restructure_and_downsample,
    downsample_dir,
)
from .reprojection_batch import (
    run_reprojection_batch,
    copy_dlc_label_csvs_for_date,
)
from .video_utils import extract_frame, find_video_files

__all__ = [
    "extract_zip",
    "restructure_and_downsample",
    "downsample_dir",
    "run_reprojection_batch",
    "copy_dlc_label_csvs_for_date",
    "extract_frame",
    "find_video_files",
]
