"""Run SuperAnimal inference using cameras and models from setup.json."""

from __future__ import annotations

import inspect
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


def normalize_extension(extension: str) -> str:
    value = extension.strip()
    if not value or value == ".":
        raise ValueError("Video extension cannot be empty")
    return value if value.startswith(".") else f".{value}"


def discover_videos(video_root: str | Path, extension: str,
                    video_directory_name: str = "videos-raw") -> list[Path]:
    root = Path(video_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Video root is not a directory: {root}")
    suffix = normalize_extension(extension).lower()
    return sorted(
        path.resolve() for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == suffix
        and path.parent.name == video_directory_name
    )


def cage_name_for_video(video: Path, video_root: str | Path,
                        cage_names: Iterable[str]) -> str:
    root = Path(video_root).expanduser().resolve()
    try:
        relative = video.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Video {video} is outside video root {root}") from exc
    configured = set(cage_names)
    for directory in reversed(relative.parts[:-1]):
        if directory in configured:
            return directory
    raise ValueError(f"No configured cage directory above {video}: {sorted(configured)}")


def determine_perspective(video: Path, perspective_map: Mapping[str, str]) -> str | None:
    matches = [(camera, perspective) for camera, perspective in perspective_map.items()
               if camera in video.stem]
    if len(matches) > 1:
        raise ValueError(f"Multiple camera names match {video.name}")
    return matches[0][1] if matches else None


def video_output_folder(video: Path, video_root: str | Path,
                        result_root: str | Path, output_layout: str = "sibling-tracks") -> Path:
    if output_layout == "sibling-tracks":
        if video.parent.name != "videos-raw":
            raise ValueError(f"Expected videos-raw parent for {video}")
        return video.parent.parent / "tracks"
    if output_layout != "mirror":
        raise ValueError(f"Unknown output layout {output_layout!r}")
    root = Path(video_root).expanduser().resolve()
    try:
        relative = video.resolve().relative_to(root).parent
    except ValueError as exc:
        raise ValueError(f"Video {video} is outside video root {root}") from exc
    return Path(result_root).expanduser().resolve() / relative


def track_output_status(video: Path, destination: Path) -> tuple[bool, bool]:
    """Report whether a video has nonempty HDF5 and adaptation artifacts."""
    h5 = any(path.stat().st_size > 0 for path in destination.glob(f"{video.stem}*.h5")
             if path.is_file())
    adapted = any(path.stat().st_size > 0
                  for path in destination.glob(f"{video.stem}*_after_adapt.json")
                  if path.is_file())
    return h5, adapted


def adapted_track_h5s(video: Path, destination: Path) -> list[Path]:
    """Return nonempty HDF5 tracks paired with nonempty after-adaptation JSON files."""
    suffix = "_after_adapt.json"
    matches: list[Path] = []
    for report in destination.glob(f"{video.stem}*{suffix}"):
        if not report.is_file() or report.stat().st_size == 0:
            continue
        track = report.with_name(report.name[:-len(suffix)] + ".h5")
        if track.is_file() and track.stat().st_size > 0:
            matches.append(track)
    return sorted(set(matches))


def has_existing_track(video: Path, destination: Path, video_adapt: bool = True) -> bool:
    if video_adapt:
        return bool(adapted_track_h5s(video, destination))
    h5, _ = track_output_status(video, destination)
    return h5


def delete_completed_labeled_videos(video: Path, destination: Path) -> int:
    """Remove only labeled preview MP4s for a completed raw video."""
    if not has_existing_track(video, destination) or not destination.is_dir():
        return 0
    removed = 0
    for path in destination.iterdir():
        if (path.is_file() and path.suffix.lower() == ".mp4"
                and path.name.startswith(f"{video.stem}_")
                and "_labeled" in path.stem[len(video.stem) + 1:]):
            try:
                path.unlink()
            except OSError as exc:
                print(f"Could not remove labeled preview {path}: {exc}")
            else:
                removed += 1
    return removed


def group_videos(videos: Iterable[Path], video_root: str | Path,
                 result_root: str | Path, cage_maps: Mapping[str, Mapping[str, str]],
                 *, skip_existing: bool = True,
                 output_layout: str = "sibling-tracks") -> tuple[dict[tuple[str, Path], list[Path]], list[str]]:
    groups: dict[tuple[str, Path], list[Path]] = defaultdict(list)
    errors: list[str] = []
    for video in videos:
        if not video.is_file():
            errors.append(f"Missing video: {video}")
            continue
        try:
            cage = cage_name_for_video(video, video_root, cage_maps)
            perspective = determine_perspective(video, cage_maps[cage])
            if perspective is None:
                errors.append(f"No camera perspective configured for {video}")
                continue
            destination = video_output_folder(video, video_root, result_root, output_layout)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if skip_existing and has_existing_track(video, destination):
            continue
        groups[(perspective, destination)].append(video)
    return groups, errors


def run_inference(videos: Iterable[Path], video_root: str | Path,
                  result_root: str | Path, extension: str,
                  *, cage_maps: Mapping[str, Mapping[str, str]],
                  model_configs: Mapping[str, str], model_name: str,
                  detector_name: str, max_individuals: int,
                  skip_existing: bool = True,
                  output_layout: str = "sibling-tracks",
                  inference_batch_size: int = 1,
                  detector_batch_size: int = 1,
                  adapt_batch_size: int = 4,
                  create_labeled_video: bool = True,
                  delete_labeled_videos_after_inference: bool = False) -> None:
    """Apply current SuperAnimal models to incomplete videos inside the DLC container."""
    try:
        from deeplabcut.modelzoo.video_inference import video_inference_superanimal
    except ImportError as exc:
        raise RuntimeError("DeepLabCut SuperAnimal inference is unavailable in this Python environment") from exc

    result = Path(result_root).expanduser().resolve()
    result.mkdir(parents=True, exist_ok=True)
    videos = list(videos)
    groups, errors = group_videos(
        videos, video_root, result, cage_maps,
        skip_existing=skip_existing, output_layout=output_layout,
    )
    (result / "error_vid.txt").write_text(
        "".join(f"{message}\n" for message in errors), encoding="utf-8"
    )
    if delete_labeled_videos_after_inference:
        pending_videos = {video for paths in groups.values() for video in paths}
        for video in videos:
            if video not in pending_videos:
                try:
                    cage = cage_name_for_video(video, video_root, cage_maps)
                    if determine_perspective(video, cage_maps[cage]) is None:
                        continue
                    destination = video_output_folder(video, video_root, result, output_layout)
                except ValueError:
                    continue
                delete_completed_labeled_videos(video, destination)

    parameters = inspect.signature(video_inference_superanimal).parameters
    supports_labeled_option = (
        "create_labeled_video" in parameters
        or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    )
    if not supports_labeled_option:
        print("Installed SuperAnimal does not accept create_labeled_video; using its default output behavior")
    for (perspective, destination), paths in sorted(
        groups.items(), key=lambda item: (item[0][0], str(item[0][1]))
    ):
        if perspective not in model_configs:
            raise ValueError(f"No SuperAnimal model configured for perspective {perspective!r}")
        destination.mkdir(parents=True, exist_ok=True)
        print(f"{len(paths)} file(s): {perspective} -> {destination}")
        labeled_option = {"create_labeled_video": create_labeled_video} if supports_labeled_option else {}
        video_inference_superanimal(
            [str(path) for path in paths], model_configs[perspective],
            model_name=model_name, detector_name=detector_name,
            videotype=normalize_extension(extension), video_adapt=True,
            batch_size=inference_batch_size,
            detector_batch_size=detector_batch_size,
            video_adapt_batch_size=adapt_batch_size,
            scale_list=[], dest_folder=str(destination),
            max_individuals=max_individuals,
            **labeled_option,
        )
        if delete_labeled_videos_after_inference:
            removed = sum(delete_completed_labeled_videos(video, destination) for video in paths)
            if removed:
                print(f"Removed {removed} labeled preview video(s) from {destination}")
