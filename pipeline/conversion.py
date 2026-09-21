"""Stage complete SuperAnimal five-camera trials for Anipose."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .pose_inference import track_output_status


@dataclass
class ConversionResult:
    converted: list[Path] = field(default_factory=list)
    ready_trials: list[str] = field(default_factory=list)
    skipped_trials: dict[str, str] = field(default_factory=dict)


def _trial_name(video: Path, camera_pattern: str) -> str:
    if not re.search(camera_pattern, video.stem):
        raise ValueError(f"Camera name does not match {camera_pattern!r}: {video.name}")
    return re.sub(camera_pattern, "", video.stem).strip()


def _camera_name(video: Path, camera_pattern: str) -> str:
    match = re.search(camera_pattern, video.stem)
    if not match:
        raise ValueError(f"Camera name does not match {camera_pattern!r}: {video.name}")
    return match.group(1)


def _track_h5(video: Path, destination: Path) -> Path:
    candidates = sorted(destination.glob(f"{video.stem}*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No SuperAnimal HDF5 track for {video.name}")
    if len(candidates) == 1:
        return candidates[0]
    adapted = [path for path in candidates if "after_adapt" in path.stem]
    if len(adapted) == 1:
        return adapted[0]
    raise ValueError(f"Multiple HDF5 tracks for {video.name}: {', '.join(p.name for p in candidates)}")


def normalize_superanimal_h5(source: Path) -> pd.DataFrame:
    """Return a single-animal, three-header-level DLC/Anipose frame table."""
    frame = pd.read_hdf(source)
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels not in {3, 4}:
        raise ValueError(f"Expected three or four HDF5 header levels in {source}")
    tuples = []
    for column in frame.columns:
        if frame.columns.nlevels == 4:
            _, _, bodypart, coordinate = column
        else:
            _, bodypart, coordinate = column
        tuples.append(("mlb2_superanimal", str(bodypart), str(coordinate)))
    if len(tuples) != len(set(tuples)):
        raise ValueError(f"Multiple animals or duplicate keypoints in {source}")
    frame.columns = pd.MultiIndex.from_tuples(tuples, names=["scorer", "bodyparts", "coords"])
    return frame


def convert_session(session_dir: str | Path, cage_maps: dict[str, dict[str, str]],
                    video_extension: str, camera_pattern: str) -> ConversionResult:
    """Write only complete, synchronized, compatible trials into pose-2d."""
    session = Path(session_dir)
    result = ConversionResult()
    suffix = "." + video_extension.lstrip(".").lower()
    for cage, mapping in cage_maps.items():
        raw = session / cage / "videos-raw"
        if not raw.is_dir():
            continue
        tracks = raw.parent / "tracks"
        output = raw.parent / "pose-2d"
        groups: dict[str, list[Path]] = {}
        for video in sorted(raw.iterdir()):
            if video.is_file() and video.suffix.lower() == suffix:
                try:
                    trial = _trial_name(video, camera_pattern)
                except ValueError as exc:
                    result.skipped_trials[f"{cage}/{video.stem}"] = str(exc)
                    continue
                groups.setdefault(trial, []).append(video)
        expected = {re.search(camera_pattern, token).group(1) for token in mapping}
        for trial, videos in groups.items():
            label = f"{cage}/{trial}"
            camera_names = {_camera_name(video, camera_pattern) for video in videos}
            if camera_names != expected or len(videos) != len(expected):
                result.skipped_trials[label] = f"Need cameras {sorted(expected)}; found {sorted(camera_names)}"
                continue
            missing_outputs = []
            for video in videos:
                h5, adapted = track_output_status(video, tracks)
                missing = []
                if not h5:
                    missing.append("HDF5")
                if not adapted:
                    missing.append("adaptation JSON")
                if missing:
                    camera = f"camera{_camera_name(video, camera_pattern)}"
                    missing_outputs.append(f"{camera} ({', '.join(missing)})")
            if missing_outputs:
                result.skipped_trials[label] = (
                    "2D tracks are still incomplete; missing " + "; ".join(missing_outputs)
                )
                continue
            try:
                sources = {video: _track_h5(video, tracks) for video in videos}
                frames = {video: normalize_superanimal_h5(source) for video, source in sources.items()}
                lengths = {len(frame) for frame in frames.values()}
                if len(lengths) != 1:
                    raise ValueError(f"Camera track frame counts differ: {sorted(lengths)}")
                bodyparts = [set(frame.columns.get_level_values("bodyparts")) for frame in frames.values()]
                shared = set.intersection(*bodyparts)
                usable = sorted(bp for bp in shared if all(
                    ("mlb2_superanimal", bp, coordinate) in frame.columns
                    for frame in frames.values() for coordinate in ("x", "y", "likelihood")
                ))
                if not usable:
                    raise ValueError("No common x/y/likelihood bodyparts across cameras")
                columns = pd.MultiIndex.from_product(
                    [["mlb2_superanimal"], usable, ["x", "y", "likelihood"]],
                    names=["scorer", "bodyparts", "coords"],
                )
                output.mkdir(parents=True, exist_ok=True)
                replacements: list[tuple[Path, Path]] = []
                for video, frame in frames.items():
                    destination = output / f"{video.stem}.h5"
                    selected = frame.loc[:, columns]
                    if destination.is_file():
                        try:
                            existing = pd.read_hdf(destination)
                            if (existing.shape == selected.shape
                                    and existing.columns.equals(selected.columns)
                                    and existing.index.equals(selected.index)
                                    and destination.stat().st_mtime >= sources[video].stat().st_mtime):
                                continue
                        except (OSError, ValueError, KeyError):
                            pass
                    temporary = destination.with_suffix(".tmp.h5")
                    selected.to_hdf(temporary, key="df", mode="w")
                    replacements.append((temporary, destination))
                for temporary, destination in replacements:
                    temporary.replace(destination)
                    result.converted.append(destination)
                result.ready_trials.append(label)
            except (OSError, ValueError, KeyError, ImportError) as exc:
                if output.is_dir():
                    for temporary in output.glob("*.tmp.h5"):
                        temporary.unlink(missing_ok=True)
                result.skipped_trials[label] = str(exc)
    return result
