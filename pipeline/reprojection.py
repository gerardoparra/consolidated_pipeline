"""
reprojection_batch.py
=====================
Batch reprojection error evaluation for Anipose multi-camera calibrations.

Given one or more ``calibration.toml`` files (or session / calibration folder
paths) and a folder of DLC-style manual label CSVs, this module:

* Triangulates the manually-labeled 2-D points with each calibration.
* Reprojects the resulting 3-D points back onto each camera plane.
* Computes the mean reprojection error (pixels) per (calibration, label) pair.
* Generates scatter-plot figures (manual vs. reprojected) for every camera.
* Generates ChArUco board overlay figures using ``cv2.solvePnP`` on the
  detection snapshots stored in ``detections.pickle``.
* Saves all figures under a structured output directory and writes a summary
  CSV with one row per (calibration × label CSV) combination.

The public API consists of two functions:

* :func:`run_reprojection_batch` — the main batch entry point.
* :func:`copy_dlc_label_csvs_for_date` — helper to stage DLC labeled-data
  CSVs from the DLC project into the labels input folder.
"""

from pathlib import Path
from math import ceil, sqrt
import glob
import re
import pickle
import shutil
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_calibration_tomls(calibration_inputs):
    """
    Accepts a list containing any mix of:
      - calibration.toml files
      - calibration folders
      - session folders (searches recursively for */calibration/calibration.toml)
    Returns a sorted list of unique calibration.toml paths.
    """
    resolved = set()

    for item in calibration_inputs:
        p = Path(item)
        if not p.exists():
            print(f"[skip] Not found: {p}")
            continue

        if p.is_file() and p.name.lower() == "calibration.toml":
            resolved.add(str(p.resolve()))
            continue

        if p.is_dir():
            direct = p / "calibration.toml"
            if direct.exists():
                resolved.add(str(direct.resolve()))
                continue

            for found in p.rglob("calibration.toml"):
                if found.parent.name.lower() == "calibration":
                    resolved.add(str(found.resolve()))

    return sorted(resolved)


def _resolve_label_csvs(label_csv_folder):
    """Returns all CSV files in a folder as sorted absolute paths."""
    csv_dir = Path(label_csv_folder)
    if not csv_dir.exists() or not csv_dir.is_dir():
        raise ValueError(f"label_csv_folder does not exist or is not a folder: {csv_dir}")

    csvs = sorted([str(p.resolve()) for p in csv_dir.glob("*.csv")])
    if not csvs:
        raise ValueError(f"No CSV files found in label_csv_folder: {csv_dir}")

    return csvs


def _extract_serial_token(text):
    """
    Extracts a camera serial-like numeric token from text.
    """
    value = str(text)
    camera = re.search(r"camera([0-9]{2})", value, flags=re.IGNORECASE)
    if camera:
        return camera.group(1)
    serial = re.search(r"(\d{6,})", value)
    return serial.group(1) if serial else None


def _camera_serials_from_calibration_dir(calibration_dir, n_cams_total):
    """
    Attempts to infer camera serials in camera-index order from calibration videos.
    """
    video_files = sorted(glob.glob(str(Path(calibration_dir) / "*.avi")))
    serials = [_extract_serial_token(Path(v).name) for v in video_files]
    if len(serials) < n_cams_total:
        serials = serials + [None] * (n_cams_total - len(serials))
    return serials[:n_cams_total]


def _load_manual_points_csv(points_csv, n_cams_total, camera_serials=None, index_cols=3):
    """
    Loads a DLC-style manual label CSV and reshapes to (n_cams_total, n_points_total, 2).

    Expected primary format: one row per camera, and x/y columns per labeled point.
    """
    labels = pd.read_csv(points_csv, header=[1, 2])
    coords = labels.iloc[:, index_cols:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    n_rows, n_coord_cols = coords.shape
    if n_coord_cols % 2 != 0:
        raise ValueError(f"CSV shape {coords.shape} has an odd number of coordinate columns.")

    # Preferred mapping: use serials from row image names to place rows into camera slots.
    if camera_serials and any(s is not None for s in camera_serials):
        n_points = n_coord_cols // 2
        points_2d_full = np.full((n_cams_total, n_points, 2), np.nan, dtype=np.float64)
        serial_to_idx = {s: i for i, s in enumerate(camera_serials) if s is not None}

        row_name_col = labels.iloc[:, 2].astype(str) if labels.shape[1] > 2 else pd.Series([""] * n_rows)
        placed = 0
        for row_idx in range(n_rows):
            row_serial = _extract_serial_token(row_name_col.iloc[row_idx])
            cam_idx = serial_to_idx.get(row_serial, None)
            if cam_idx is None or cam_idx >= n_cams_total:
                continue
            points_2d_full[cam_idx, :, :] = coords[row_idx].reshape(n_points, 2)
            placed += 1

        if placed > 0:
            if placed < n_rows:
                print(
                    f"[warn] {Path(points_csv).name}: mapped {placed}/{n_rows} rows by serial; "
                    f"unmapped rows were skipped."
                )
            return points_2d_full, placed

    # Primary fallback interpretation: rows are cameras, columns are points*(x,y).
    if n_rows <= n_cams_total:
        n_points = n_coord_cols // 2
        points_2d_csv = coords.reshape(n_rows, n_points, 2)

        if n_rows == n_cams_total:
            return points_2d_csv, n_rows

        points_2d_full = np.full((n_cams_total, n_points, 2), np.nan, dtype=np.float64)
        points_2d_full[:n_rows, :, :] = points_2d_csv
        print(f"[warn] {Path(points_csv).name}: found {n_rows} camera row(s); padding to {n_cams_total}.")
        return points_2d_full, n_rows

    # Fallback interpretation for legacy/multi-frame layouts.
    candidate_camera_counts = [
        n for n in range(1, n_cams_total + 1)
        if n_coord_cols % (n * 2) == 0
    ]
    if not candidate_camera_counts:
        raise ValueError(
            f"CSV shape {coords.shape} is not compatible with calibration camera count {n_cams_total}."
        )

    n_cams_csv = max(candidate_camera_counts)
    n_points_per_row = n_coord_cols // (n_cams_csv * 2)
    reshaped = coords.reshape(n_rows, n_cams_csv, n_points_per_row, 2)
    points_2d_csv = reshaped.transpose(1, 0, 2, 3).reshape(n_cams_csv, n_rows * n_points_per_row, 2)

    if n_cams_csv == n_cams_total:
        return points_2d_csv, n_cams_csv

    points_2d_full = np.full((n_cams_total, points_2d_csv.shape[1], 2), np.nan, dtype=np.float64)
    points_2d_full[:n_cams_csv, :, :] = points_2d_csv
    print(f"[warn] {Path(points_csv).name}: inferred {n_cams_csv} camera(s) in fallback mode; padding to {n_cams_total}.")
    return points_2d_full, n_cams_csv


def _plot_points_vs_reprojection(points_2d, proj_2d, title_prefix="", image_size=(1920, 1080)):
    n_cams = points_2d.shape[0]
    n_cols = int(ceil(sqrt(n_cams)))
    n_rows = int(ceil(n_cams / n_cols))

    width, height = image_size
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axs = np.array(axs).reshape(-1)

    for cam_idx in range(n_cams):
        ax = axs[cam_idx]
        ax.scatter(points_2d[cam_idx, :, 0], height - points_2d[cam_idx, :, 1], s=18, color="tab:blue", label="Manual")
        ax.scatter(proj_2d[cam_idx, :, 0], height - proj_2d[cam_idx, :, 1], s=18, color="tab:red", marker="x", label="Reprojected")
        ax.set_title(f"{title_prefix}Camera {cam_idx}")
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.legend()

    for ax in axs[n_cams:]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def _create_charuco_board(square_length = 0.026, marker_length = 0.013):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    squares_x, squares_y = 10, 7
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)


def _charuco_reprojection_figures(calibration_dir, cgroup, max_frames_per_cam=1):
    """
    Returns {camera_index: [fig1, fig2, ...]} for valid detections in detections.pickle.
    """
    calibration_dir = Path(calibration_dir)
    detections_path = calibration_dir / "detections.pickle"
    if not detections_path.exists():
        print(f"[warn] No detections file: {detections_path}")
        return {}

    with open(detections_path, "rb") as f:
        detections = pickle.load(f)

    video_files = sorted(glob.glob(str(calibration_dir / "*.avi")))
    if len(video_files) == 0:
        print(f"[warn] No .avi files in: {calibration_dir}")
        return {}

    board = _create_charuco_board()
    board_corners = board.getChessboardCorners()

    out = {}
    for cam_idx, cam in enumerate(cgroup.cameras):
        if cam_idx >= len(detections) or cam_idx >= len(video_files):
            continue

        cam_dets = detections[cam_idx]
        if cam_dets is None:
            continue

        fig_list = []
        used = 0

        for det in cam_dets:
            if det is None:
                continue
            if used >= max_frames_per_cam:
                break

            corners = det.get("corners", None)
            ids = det.get("ids", None)
            framenum = det.get("framenum", None)

            if corners is None or ids is None or framenum is None:
                continue

            ids = np.array(ids).flatten().astype(int)
            if len(ids) == 0:
                continue

            valid = (ids >= 0) & (ids < len(board_corners))
            ids = ids[valid]
            if len(ids) == 0:
                continue

            det_pts = np.array(corners).reshape(-1, 2)
            if len(det_pts) != len(valid):
                continue
            det_pts = det_pts[valid]

            obj_pts = board_corners[ids]

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                det_pts,
                cam.matrix,
                cam.dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue

            proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam.matrix, cam.dist)
            proj_pts = proj_pts.reshape(-1, 2)

            frame_idx = int(framenum[1] if isinstance(framenum, (tuple, list, np.ndarray)) else framenum)
            cap = cv2.VideoCapture(video_files[cam_idx])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img = cap.read()
            cap.release()
            if not ret:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frame_reproj_error = float(np.mean(np.linalg.norm(det_pts - proj_pts, axis=1)))

            fig = plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.scatter(det_pts[:, 0], det_pts[:, 1], c="lime", s=15, label="Detected")
            plt.scatter(proj_pts[:, 0], proj_pts[:, 1], c="red", s=10, label="Reprojected")
            plt.legend()
            plt.title(
                f"Charuco reprojection | cam {cam_idx} | frame {frame_idx} | "
                f"err {frame_reproj_error:.3f}px"
            )
            plt.axis("off")
            plt.tight_layout()

            fig_list.append(fig)
            used += 1

        if fig_list:
            out[cam_idx] = fig_list

    return out


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def _session_box_names(calibration_path):
    cal_toml = Path(calibration_path).resolve()
    calibration_dir = cal_toml.parent

    if calibration_dir.name.lower() == "calibration" and calibration_dir.parent != calibration_dir:
        box_dir = calibration_dir.parent
        session_dir = box_dir.parent if box_dir.parent != box_dir else calibration_dir
        return session_dir.name, box_dir.name

    return calibration_dir.parent.name or "session", calibration_dir.name


def _calibration_day_box(calibration_path):
    """
    Returns (calibration_day, calibration_box) parsed from calibration.toml location.
    """
    cal_toml = Path(calibration_path).resolve()
    calibration_dir = cal_toml.parent

    if calibration_dir.name.lower() == "calibration" and calibration_dir.parent != calibration_dir:
        box_dir = calibration_dir.parent
        day_dir = box_dir.parent if box_dir.parent != box_dir else calibration_dir
        day_name = day_dir.name
        box_name = box_dir.name
    else:
        day_name = calibration_dir.parent.name or "session"
        box_name = calibration_dir.name

    m = re.search(r"(box\s*\d+)", box_name, flags=re.IGNORECASE)
    box_norm = m.group(1).replace(" ", "").upper() if m else box_name.upper()
    return day_name, box_norm


def _extract_leading_yymmdd(text, field_name):
    """
    Extract the leading YYMMDD token from strings like:
    260226_Rat_Lockbox_...
    """
    value = str(text)
    # MLB2 sliding-lockbox sessions use DD_MM_YYYY; older rat sessions use YYMMDD.
    local = re.match(r"^(\d{2})[_-](\d{2})[_-](\d{4})", value)
    if local:
        day, month, year = local.groups()
        return year[-2:] + month + day
    iso = re.match(r"^(\d{4})[_-]?(\d{2})[_-]?(\d{2})", value)
    if iso:
        year, month, day = iso.groups()
        return year[-2:] + month + day
    old = re.match(r"^(\d{6})", value)
    if old:
        return old.group(1)
    raise ValueError(f"Could not extract a date from {field_name}: {text}")


def _summary_calibration_label(calibration_path):
    """
    Returns a compact calibration label: <day_folder>/<box_folder>.
    """
    cal_toml = Path(calibration_path).resolve()
    calibration_dir = cal_toml.parent
    if calibration_dir.name.lower() == "calibration" and calibration_dir.parent != calibration_dir:
        box_dir = calibration_dir.parent
        day_dir = box_dir.parent if box_dir.parent != box_dir else calibration_dir
        return f"{day_dir.name}/{box_dir.name}"
    return f"{calibration_dir.parent.name}/{calibration_dir.name}"


def _summary_label_csv(label_csv):
    """
    Returns a compact label csv path: <day_folder>/<csv_filename>.
    """
    p = Path(label_csv).resolve()
    return f"{p.parent.name}/{p.name}"


def _save_figures_for_calibration(output_root, label_day_name, calibration_path, manual_entries, charuco_entries):
    """
    Saves figures to output_root/label_day_name/calibration_day/box_name and closes them.
    """
    calibration_day, box_name = _calibration_day_box(calibration_path)
    box_output_dir = (
        Path(output_root)
        / _safe_name(label_day_name)
        / _safe_name(calibration_day)
        / _safe_name(box_name)
    )
    box_output_dir.mkdir(parents=True, exist_ok=True)

    for label_csv, fig in manual_entries:
        csv_stem = _safe_name(Path(label_csv).stem)
        out_path = box_output_dir / f"manual_vs_reproj__{csv_stem}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    for cam_idx, fig_list in charuco_entries.items():
        for i, fig in enumerate(fig_list, start=1):
            out_path = box_output_dir / f"charuco_cam{cam_idx}_sample{i}.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

    return box_output_dir


def run_reprojection_batch(
    calibration_inputs,
    label_csv_folder,
    output_root,
    image_size=(1920, 1080),
    max_charuco_frames_per_cam=1,
):
    """Run reprojection error evaluation across multiple calibrations and label CSVs.

    For each ``calibration.toml`` resolved from *calibration_inputs* and each
    label CSV in *label_csv_folder* the function:

    1. Triangulates the manually-labeled 2-D points using the calibration.
    2. Reprojects the 3-D points back to every camera plane.
    3. Computes the mean reprojection error in pixels.
    4. Produces a scatter-plot figure (manual vs. reprojected, one sub-plot per
       camera) and ChArUco board overlay figures for up to
       *max_charuco_frames_per_cam* frames per camera.
    5. Saves all figures to
       ``output_root/<label_day>/<calibration_day>/<box_name>/``.
    6. Writes a summary CSV at
       ``output_root/<label_day>/reprojection_summary.csv``.

    Parameters
    ----------
    calibration_inputs : list[str | Path]
        Any mix of: paths to ``calibration.toml`` files, calibration
        directories, or session directories.  Session directories are searched
        recursively for ``*/calibration/calibration.toml``.
    label_csv_folder : str | Path
        Directory containing DLC-style manual label CSVs (``*.csv``).
        The folder name is expected to start with a 6-digit ``YYMMDD`` prefix.
    output_root : str | Path
        Root directory where figure sub-directories and the summary CSV will
        be written.  Created if it does not exist.
    image_size : tuple[int, int], optional
        ``(width, height)`` in pixels of the camera images.  Used to set axis
        limits on scatter plots (default ``(1920, 1080)``).
    max_charuco_frames_per_cam : int, optional
        Maximum number of ChArUco frame overlays to generate per camera
        (default ``1``).

    Returns
    -------
    summary_df : pandas.DataFrame
        One row per ``(calibration, label_csv)`` pair with columns:
        ``label_day``, ``calibration_day``, ``calibration_box``,
        ``calibration_path``, ``label_csv``, ``n_cameras``,
        ``n_cameras_in_csv``, ``n_points_total``,
        ``mean_reprojection_error_px``.
    figures : dict
        ``'manual_vs_reproj'``: ``{(calibration_path, label_csv): fig}``
        ``'charuco'``: ``{calibration_path: {cam_idx: [fig, ...]}}``
    """
    from aniposelib.cameras import CameraGroup

    calibration_paths = _resolve_calibration_tomls(calibration_inputs)
    if not calibration_paths:
        raise ValueError("No calibration.toml files resolved from calibration_inputs.")

    label_csvs = _resolve_label_csvs(label_csv_folder)
    label_day_name = Path(label_csv_folder).resolve().name
    label_day = _extract_leading_yymmdd(label_day_name, "label_csv_folder")

    figures = {
        "manual_vs_reproj": {},
        "charuco": {}
    }
    rows = []

    for calibration_path in calibration_paths:
        cgroup = CameraGroup.load(calibration_path)
        n_cams = len(cgroup.cameras)
        calibration_day_name, calibration_box = _calibration_day_box(calibration_path)
        calibration_day = _extract_leading_yymmdd(calibration_day_name, "calibration_path")

        calibration_dir = str(Path(calibration_path).parent)
        camera_serials = _camera_serials_from_calibration_dir(calibration_dir, n_cams)
        charuco_for_calib = _charuco_reprojection_figures(
            calibration_dir,
            cgroup,
            max_frames_per_cam=max_charuco_frames_per_cam
        )
        figures["charuco"][calibration_path] = charuco_for_calib

        manual_for_calib = []

        for label_csv in label_csvs:
            points_2d, n_cams_in_csv = _load_manual_points_csv(
                label_csv,
                n_cams_total=n_cams,
                camera_serials=camera_serials,
            )
            points_3d = cgroup.triangulate(points_2d, progress=False)
            proj_2d = cgroup.project(points_3d)

            reprojerr = cgroup.reprojection_error(points_3d, points_2d, mean=True)
            mean_error = float(np.nanmean(reprojerr))

            fig = _plot_points_vs_reprojection(
                points_2d,
                proj_2d,
                title_prefix=(
                    f"Calib {calibration_day_name}/{calibration_box} | "
                    f"Labels {label_day_name} | "
                    f"Mean err {mean_error:.3f}px | "
                ),
                image_size=image_size
            )

            key = (calibration_path, str(Path(label_csv).resolve()))
            figures["manual_vs_reproj"][key] = fig
            manual_for_calib.append((label_csv, fig))

            rows.append({
                "label_day": label_day,
                "calibration_day": calibration_day,
                "calibration_box": calibration_box,
                "calibration_path": _summary_calibration_label(calibration_path),
                "label_csv": _summary_label_csv(label_csv),
                "n_cameras": n_cams,
                "n_cameras_in_csv": int(n_cams_in_csv),
                "n_points_total": int(points_2d.shape[1]),
                "mean_reprojection_error_px": mean_error
            })

            print(f"[done] {Path(calibration_path).parent.parent.name} + {Path(label_csv).name} -> {mean_error:.4f}px")

        saved_dir = _save_figures_for_calibration(
            output_root=output_root,
            label_day_name=label_day_name,
            calibration_path=calibration_path,
            manual_entries=manual_for_calib,
            charuco_entries=charuco_for_calib,
        )
        print(f"[saved] {saved_dir}")

    summary_df = pd.DataFrame(rows).sort_values(
        ["calibration_day", "calibration_box", "label_csv"]
    ).reset_index(drop=True)

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_root_path / _safe_name(label_day_name) / "reprojection_summary.csv"
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[saved] {summary_csv_path}")

    return summary_df, figures


def copy_dlc_label_csvs_for_date(date_str, dlc_model_path, target_folder, target_root=None):
    """Copy DLC label CSVs from a labeled-data directory that match a given date.

    Searches ``<dlc_model_path>/labeled-data/`` for subdirectories whose names
    start with *date_str* (``YYYY-MM-DD``) and copies all ``*.csv`` files found
    within them to ``<target_root>/<target_folder>/``.  Duplicate filenames are
    resolved by prepending the source subdirectory name.

    Parameters
    ----------
    date_str : str
        Recording date in ``YYYY-MM-DD`` format.  Folders in
        ``labeled-data/`` whose names begin with this string are included.
    dlc_model_path : str | Path
        Root directory of the DLC project.  Must contain a ``labeled-data/``
        sub-directory.
    target_folder : str | Path
        Sub-folder name (or relative path) inside *target_root* where the CSVs
        should be placed.
    target_root : str | Path, optional
        Base destination directory.  Defaults to
        ``<this file's directory>/manual_test_labels``.

    Returns
    -------
    target_dir : Path
        Absolute path to the destination directory.
    copied_files : list[str]
        Absolute paths of all copied CSV files.

    Raises
    ------
    ValueError
        If *date_str* is not in ``YYYY-MM-DD`` format, the ``labeled-data``
        directory cannot be found, no matching folders exist, or no CSV files
        were found in those folders.
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date_str '{date_str}'. Expected YYYY-MM-DD.") from exc

    dlc_root = Path(dlc_model_path).resolve()
    labeled_data_dir = dlc_root / "labeled-data"
    if not labeled_data_dir.is_dir():
        raise ValueError(f"Could not find labeled-data directory at: {labeled_data_dir}")

    if target_root is None:
        target_root = Path(__file__).resolve().parent / "manual_test_labels"

    target_dir = Path(target_root).resolve() / target_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    matched_dirs = sorted(
        p for p in labeled_data_dir.iterdir()
        if p.is_dir() and p.name.startswith(date_str)
    )
    if not matched_dirs:
        raise ValueError(
            f"No folders in {labeled_data_dir} start with '{date_str}'."
        )

    copied_files = []
    used_names = set()

    for src_dir in matched_dirs:
        csv_paths = sorted(src_dir.rglob("*.csv"))
        for src in csv_paths:
            dst_name = src.name

            if dst_name in used_names or (target_dir / dst_name).exists():
                dst_name = f"{src_dir.name}__{src.name}"

            if dst_name in used_names or (target_dir / dst_name).exists():
                stem = Path(dst_name).stem
                suffix = Path(dst_name).suffix
                i = 1
                while True:
                    candidate = f"{stem}__{i}{suffix}"
                    if candidate not in used_names and not (target_dir / candidate).exists():
                        dst_name = candidate
                        break
                    i += 1

            dst = target_dir / dst_name
            shutil.copy2(src, dst)
            copied_files.append(str(dst))
            used_names.add(dst_name)

    if not copied_files:
        raise ValueError(
            f"No CSV files found under date-matched folders for '{date_str}' in {labeled_data_dir}."
        )

    print(f"[copied] {len(copied_files)} CSV(s) -> {target_dir}")
    return target_dir, copied_files
