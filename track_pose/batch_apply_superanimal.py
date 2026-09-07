"""Run DeepLabCut SuperAnimal inference over a batch of videos.

The preferred interface is ``--video-path`` plus ``--video-extension``. It
recursively finds videos and mirrors their directory layout below the result
folder. The older labels and video-list arguments remain available.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

SUPPORTED_PERSPECTIVES = frozenset({"top", "front", "side", "left", "right", "back"})
PERSPECTIVE_PRESETS = {
    "default": {"23499137": "top", "23422463": "front", "23501391": "side"},
    "rats": {"23499131": "top", "23501387": "front", "24622086": "left", "24806000": "right", "25272920": "back"},
    "mlb2_cage1": {"camera06": "top", "camera07": "front", "camera08": "left", "camera09": "right", "camera10": "back"},
    "mlb2_cage2": {"camera01": "left", "camera02": "front", "camera03": "right", "camera04": "top", "camera05": "back"},
}
MODEL_CONFIGS = {perspective: "superanimal_quadruped" for perspective in SUPPORTED_PERSPECTIVES}
MODEL_CONFIGS["top"] = "superanimal_topviewmouse"


def normalize_extension(extension: str) -> str:
    """Return an extension with one leading dot."""
    value = extension.strip()
    if not value:
        raise ValueError("Video extension cannot be empty.")
    return value if value.startswith(".") else f".{value}"


def discover_videos(video_root: str | Path, extension: str, video_directory_name: str | None = None) -> list[Path]:
    """Recursively return matching videos, optionally only from named folders."""
    root = Path(video_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Video path is not a directory: {root}")
    suffix = normalize_extension(extension).lower()
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == suffix
        and (video_directory_name is None or path.parent.name == video_directory_name)
    )


def load_perspective_map(preset: str = "default", map_file: str | Path | None = None) -> dict[str, str]:
    """Load and validate either a built-in perspective map or a JSON map."""
    if map_file is None:
        return dict(PERSPECTIVE_PRESETS[preset])
    path = Path(map_file).expanduser().resolve()
    try:
        raw_map = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"Could not read perspective map {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Perspective map {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("Perspective map must be a non-empty JSON object.")
    if not all(isinstance(serial, str) and isinstance(perspective, str) for serial, perspective in raw_map.items()):
        raise ValueError("Perspective map keys and values must be strings.")
    invalid = sorted(set(raw_map.values()) - SUPPORTED_PERSPECTIVES)
    if invalid:
        raise ValueError(f"Unsupported perspective(s): {', '.join(invalid)}")
    return dict(raw_map)


def load_cage_maps(map_file: str | Path) -> dict[str, dict[str, str]]:
    """Load a JSON ``cage-directory -> preset or camera map`` configuration.

    Example::

        {"cage1": "mlb2_cage1", "cage2": "mlb2_cage2"}
    """
    path = Path(map_file).expanduser().resolve()
    try:
        raw_maps = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"Could not read cage map {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cage map {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw_maps, dict) or not raw_maps:
        raise ValueError("Cage map must be a non-empty JSON object.")

    result: dict[str, dict[str, str]] = {}
    for cage, map_definition in raw_maps.items():
        if not isinstance(cage, str) or not cage:
            raise ValueError("Cage-map keys must be non-empty cage directory names.")
        if isinstance(map_definition, str):
            if map_definition not in PERSPECTIVE_PRESETS:
                raise ValueError(f"Cage {cage!r} references unknown preset {map_definition!r}.")
            result[cage] = load_perspective_map(map_definition)
        elif isinstance(map_definition, dict):
            # Reuse the normal map validator after serializing the inline map.
            if not map_definition or not all(isinstance(serial, str) and isinstance(perspective, str) for serial, perspective in map_definition.items()):
                raise ValueError(f"Cage {cage!r} must use a preset name or a camera-map object.")
            invalid = sorted(set(map_definition.values()) - SUPPORTED_PERSPECTIVES)
            if invalid:
                raise ValueError(f"Cage {cage!r} has unsupported perspective(s): {', '.join(invalid)}")
            result[cage] = dict(map_definition)
        else:
            raise ValueError(f"Cage {cage!r} must use a preset name or a camera-map object.")
    return result


def determine_perspective(video: str | Path, perspective_map: Mapping[str, str]) -> str | None:
    """Find the mapped camera serial in a video filename/path."""
    for serial, perspective in perspective_map.items():
        if serial in str(video):
            return perspective
    return None


def cage_name_for_video(video: Path, video_root: str | Path, cage_names: Iterable[str]) -> str:
    """Return the nearest configured cage directory above *video*.

    This supports both ``experiment/CAGE2/file`` and more deeply nested
    layouts such as ``experiment/session/CAGE2/videos/file``.
    """
    root = Path(video_root).expanduser().resolve()
    try:
        relative = video.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Video {video} is outside video root {root}") from exc
    configured_cages = set(cage_names)
    for directory in reversed(relative.parts[:-1]):
        if directory in configured_cages:
            return directory
    choices = ", ".join(sorted(configured_cages))
    raise ValueError(f"No configured cage directory ({choices}) found above video: {video}")


def find_paths(vids: Iterable[str], folders: Iterable[str], master_folder: str | Path, batches: Iterable[int | str] | None = None) -> list[Path]:
    """Legacy labels.csv lookup, returning absolute paths."""
    root = Path(master_folder).expanduser().resolve()
    batch_numbers = list(batches or [1, 2, 3])
    result: list[Path] = []
    for video, folder in zip(vids, folders):
        candidates = [root / f"batch{batch}" / str(folder) / str(video) for batch in batch_numbers]
        candidates.append(root / str(folder) / str(video))
        for candidate in candidates:
            if candidate.exists():
                result.append(candidate.resolve())
                break
    return result


def read_video_list(video_list: str | Path, video_root: str | Path) -> list[Path]:
    """Read one-path-per-line lists and ensure every entry is below root."""
    root = Path(video_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Video path is not a directory: {root}")
    with Path(video_list).open(newline="") as handle:
        entries = [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]
    videos, outside = [], []
    for entry in entries:
        candidate = Path(entry.replace("\\", "/")).expanduser()
        path = (root / candidate if not candidate.is_absolute() else candidate).resolve()
        try:
            path.relative_to(root)
            videos.append(path)
        except ValueError:
            outside.append(path)
    if outside:
        shown = "\n".join(str(path) for path in outside[:10])
        raise ValueError(f"Video-list paths must be inside {root}:\n{shown}" + ("\n..." if len(outside) > 10 else ""))
    return videos


def legacy_label_videos(labels_file: str | Path, video_root: str | Path, batch_id: str | None, lockbox_id: str | None, lockbox_type: str) -> list[Path]:
    import pandas as pd

    labels = pd.read_csv(labels_file)
    labels = labels[labels["type"] == int(lockbox_type)]
    if batch_id is not None:
        labels = labels[labels["group"].astype(str).str.contains(f"-{batch_id}")]
    if lockbox_id is not None:
        labels = labels[labels["lock_box"] == float(lockbox_id)]
    return find_paths(labels["filename"], labels["folder"], video_root, [batch_id] if batch_id else None)


def video_output_folder(
    video: Path,
    video_root: str | Path,
    result_root: str | Path,
    output_layout: str = "mirror",
    video_directory_name: str = "videos-raw",
    tracks_directory_name: str = "tracks",
) -> Path:
    """Return a mirrored destination or the sibling tracks directory."""
    if output_layout == "sibling-tracks":
        if video.parent.name != video_directory_name:
            raise ValueError(
                f"Expected {video_directory_name!r} as the video parent for sibling-tracks output: {video}"
            )
        return video.parent.parent / tracks_directory_name
    root = Path(video_root).expanduser().resolve()
    try:
        relative_parent = video.resolve().relative_to(root).parent
    except ValueError as exc:
        raise ValueError(f"Video {video} is outside video root {root}") from exc
    return Path(result_root).expanduser().resolve() / relative_parent


def has_existing_track(video: Path, destination: Path, video_adapt: bool = True) -> bool:
    """Return whether the final expected output exists, not merely partial output."""
    has_h5 = any(destination.glob(f"{video.stem}*.h5"))
    if not video_adapt:
        return has_h5
    return has_h5 and any(destination.glob(f"{video.stem}*_after_adapt.json"))


def group_videos(
    videos: Iterable[Path], video_root: str | Path, result_root: str | Path,
    perspective_map: Mapping[str, str] | None = None, skip_existing: bool = False,
    cage_maps: Mapping[str, Mapping[str, str]] | None = None,
    output_layout: str = "mirror", video_directory_name: str = "videos-raw",
    tracks_directory_name: str = "tracks",
) -> tuple[dict[tuple[str, Path], list[Path]], list[str]]:
    """Group videos by perspective and their mirrored DLC destination."""
    groups: dict[tuple[str, Path], list[Path]] = defaultdict(list)
    errors: list[str] = []
    for video in videos:
        if not video.is_file():
            errors.append(f"Missing video: {video}")
            continue
        try:
            video_map = perspective_map
            if cage_maps is not None:
                cage = cage_name_for_video(video, video_root, cage_maps)
                video_map = cage_maps.get(cage)
                if video_map is None:
                    errors.append(f"No camera map configured for cage {cage!r}: {video}")
                    continue
            if video_map is None:
                errors.append(f"No camera map configured for: {video}")
                continue
            perspective = determine_perspective(video, video_map)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if perspective is None:
            errors.append(f"No camera perspective configured for: {video}")
            continue
        try:
            destination = video_output_folder(
                video, video_root, result_root, output_layout,
                video_directory_name, tracks_directory_name,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if skip_existing and has_existing_track(video, destination):
            continue
        groups[(perspective, destination)].append(video)
    return groups, errors


def run_inference(
    videos: Iterable[Path], video_root: str | Path, result_root: str | Path, extension: str,
    perspective_map: Mapping[str, str] | None = None, skip_existing: bool = False,
    cage_maps: Mapping[str, Mapping[str, str]] | None = None,
    output_layout: str = "mirror", video_directory_name: str = "videos-raw",
    tracks_directory_name: str = "tracks", inference_batch_size: int = 1,
    detector_batch_size: int = 1, adapt_batch_size: int = 4,
) -> None:
    """Run SuperAnimal inference, mirroring source directories in result_root."""
    # Local unit tests and login nodes need not import the container-only DLC package.
    from deeplabcut.modelzoo.video_inference import video_inference_superanimal

    result = Path(result_root).expanduser().resolve()
    result.mkdir(parents=True, exist_ok=True)
    groups, errors = group_videos(
        videos, video_root, result, perspective_map, skip_existing, cage_maps,
        output_layout, video_directory_name, tracks_directory_name,
    )
    with (result / "error_vid.txt").open("w") as handle:
        handle.writelines(f"{message}\n" for message in errors)
    for (perspective, destination), paths in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        destination.mkdir(parents=True, exist_ok=True)
        print(f"{len(paths)} file(s): {perspective} -> {destination}")
        video_inference_superanimal(
            [str(path) for path in paths], MODEL_CONFIGS[perspective], model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2", videotype=normalize_extension(extension),
            video_adapt=True, batch_size=inference_batch_size,
            detector_batch_size=detector_batch_size, video_adapt_batch_size=adapt_batch_size,
            scale_list=[], dest_folder=str(destination), max_individuals=1,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply SuperAnimal models to videos.")
    parser.add_argument("labels_file", nargs="?", default="/wd/gparra/mlb/labels.csv", help="Legacy labels CSV.")
    parser.add_argument("vid_folder", nargs="?", default="/wd/mlb24/v", help="Legacy video root.")
    parser.add_argument("vid_list", nargs="?", default=None, help="Legacy one-video-per-line CSV.")
    parser.add_argument("-b", "--batch", default=None, help="Legacy batch number.")
    parser.add_argument("-l", "--lockbox", default=None, help="Legacy lockbox number.")
    parser.add_argument("-y", "--type", default="2", help="Legacy lockbox type.")
    parser.add_argument("-r", "--result-folder", default="/wd/gparra/mlb/tracks/", help="Output root.")
    parser.add_argument("-p", "--skip-existing", action="store_true", help="Skip videos with an existing HDF5 track.")
    parser.add_argument("--restore-structure", action="store_true", help="Deprecated; structure is always preserved.")
    parser.add_argument("--video-path", help="Root to recursively scan for videos.")
    parser.add_argument("--video-extension", help="Extension to scan/pass to DLC, e.g. avi or .mp4.")
    parser.add_argument("--video-directory-name", help="Only scan videos directly inside folders with this name.")
    parser.add_argument("--output-layout", choices=("mirror", "sibling-tracks"), default="mirror", help="Where to write outputs.")
    parser.add_argument("--tracks-directory-name", default="tracks", help="Sibling output directory name for sibling-tracks layout.")
    parser.add_argument("--inference-batch-size", type=int, default=1, help="Pose inference batch size.")
    parser.add_argument("--detector-batch-size", type=int, default=1, help="Detector inference batch size.")
    parser.add_argument("--adapt-batch-size", type=int, default=4, help="Video-adaptation training batch size.")
    map_group = parser.add_mutually_exclusive_group()
    map_group.add_argument("--perspective-preset", choices=sorted(PERSPECTIVE_PRESETS), default="default", help="Built-in camera map.")
    map_group.add_argument("--perspective-map-file", help="JSON camera-serial-to-perspective map.")
    map_group.add_argument("--cage-map-file", help="JSON mapping cage directory names to presets or camera maps.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video_path and not args.video_extension:
        parser.error("--video-extension is required with --video-path")
    if min(args.inference_batch_size, args.detector_batch_size, args.adapt_batch_size) < 1:
        parser.error("All batch sizes must be positive integers")
    if args.output_layout == "sibling-tracks" and not args.video_path:
        parser.error("--output-layout sibling-tracks requires --video-path")
    extension = args.video_extension or ".avi"
    video_root = Path(args.video_path or args.vid_folder).expanduser().resolve()
    try:
        if args.video_path:
            videos = discover_videos(video_root, extension, args.video_directory_name)
        elif args.vid_list:
            videos = read_video_list(args.vid_list, video_root)
        else:
            videos = legacy_label_videos(args.labels_file, video_root, args.batch, args.lockbox, args.type)
        cage_maps = load_cage_maps(args.cage_map_file) if args.cage_map_file else None
        perspective_map = None if cage_maps else load_perspective_map(args.perspective_preset, args.perspective_map_file)
        run_inference(
            videos, video_root, args.result_folder, extension, perspective_map,
            args.skip_existing, cage_maps, args.output_layout,
            args.video_directory_name or "videos-raw", args.tracks_directory_name,
            args.inference_batch_size, args.detector_batch_size, args.adapt_batch_size,
        )
    except (OSError, ValueError, NotADirectoryError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
