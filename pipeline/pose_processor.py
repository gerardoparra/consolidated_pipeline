"""SuperAnimal pose inference, track conversion, and 3D triangulation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .anipose_runner import AniposeRunner
from .conversion import ConversionResult, convert_session
from .pose_inference import discover_videos, run_inference


@dataclass
class TriangulationResult:
    outputs: list[Path] = field(default_factory=list)
    skipped_trials: dict[str, str] = field(default_factory=dict)
    anipose_output: str = ""


@dataclass
class FilteringResult:
    outputs: list[Path] = field(default_factory=list)
    ready_trials: list[str] = field(default_factory=list)
    skipped_trials: dict[str, str] = field(default_factory=dict)
    anipose_output: str = ""


@dataclass
class VisualizationResult:
    outputs: list[Path] = field(default_factory=list)
    skipped_trials: dict[str, str] = field(default_factory=dict)
    anipose_output: str = ""


class PoseProcessor(AniposeRunner):

    @staticmethod
    def _settings_digest(settings: object) -> str:
        encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _read_state(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _write_state(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _archive(path: Path, label: str = "stale") -> Path:
        archived = path.with_name(path.name + f".{label}")
        suffix = 1
        while archived.exists():
            archived = path.with_name(path.name + f".{label}.{suffix}")
            suffix += 1
        path.rename(archived)
        return archived

    @staticmethod
    def _valid_pose2d(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            with pd.HDFStore(path, mode="r") as store:
                return bool(store.keys())
        except (OSError, ValueError, KeyError):
            return False

    def _render_preview(self, pose_csv: Path, source_video: Path,
                        output: Path, preview_fps: float) -> str:
        """Run the sampled renderer in the configured Anipose environment."""
        command = self._anipose_command()
        if command[-1] != "anipose":
            raise ValueError(
                "The configured Anipose command must end in 'anipose' to run label-3d"
            )
        worker = Path(__file__).with_name("label_3d_preview.py")
        command = command[:-1] + [
            "python", str(worker),
            "--config", str(self.experiment_dir / "config.toml"),
            "--pose-csv", str(pose_csv),
            "--source-video", str(source_video),
            "--output", str(output),
            "--fps", str(preview_fps),
        ]
        print(f"Rendering 3D preview: {pose_csv}", flush=True)
        completed = subprocess.run(command, cwd=self.experiment_dir)
        if completed.returncode:
            raise RuntimeError(
                f"3D preview rendering failed for {pose_csv} "
                f"(exit {completed.returncode}); see the renderer output above."
            )
        return f"Rendered {pose_csv.name} at {preview_fps:g} FPS"

    def detect_keypoints(self, video_root: str | Path | None = None,
                         result_root: str | Path | None = None,
                         extension: str | None = None,
                         inference_batch_size: int | None = None,
                         detector_batch_size: int | None = None,
                         adapt_batch_size: int | None = None) -> None:
        """Run the worker directly inside the DLC GPU container."""
        root = Path(video_root or self.experiment_dir)
        pose = self.setup["pose"]
        ext = extension or pose["video_extension"]
        videos = discover_videos(root, ext, "videos-raw")
        if not videos:
            print(f"No {ext} files found under {root}/.../videos-raw")
            return
        run_inference(
            videos, root, result_root or root, ext,
            skip_existing=pose["skip_existing"], cage_maps=self.setup["cages"],
            output_layout="sibling-tracks",
            inference_batch_size=inference_batch_size or pose["inference_batch_size"],
            detector_batch_size=detector_batch_size or pose["detector_batch_size"],
            adapt_batch_size=adapt_batch_size or pose["adapt_batch_size"],
            model_configs=pose["perspective_models"], model_name=pose["model_name"],
            detector_name=pose["detector_name"], max_individuals=pose["max_individuals"],
            create_labeled_video=pose["create_labeled_video"],
            delete_labeled_videos_after_inference=pose["delete_labeled_videos_after_inference"],
        )

    def convert_dlc_output_to_anipose(self, session_dir: str | Path) -> ConversionResult:
        return convert_session(
            session_dir, self.setup["cages"], self.setup["pose"]["video_extension"],
            self.setup["anipose"]["cam_regex"],
        )

    def filter_2d(self, session_dir: str | Path,
                  conversion: ConversionResult) -> FilteringResult:
        """Temporally filter converted camera tracks and resume valid outputs."""
        session = Path(session_dir)
        result = FilteringResult(skipped_trials=dict(conversion.skipped_trials))
        if not self.setup["filter"]["enabled"]:
            result.ready_trials = list(conversion.ready_trials)
            return result

        digest = self._settings_digest(self.setup["filter"])
        pending: list[tuple[str, list[tuple[Path, Path]]]] = []
        touched_cages: set[str] = set()
        for label in conversion.ready_trials:
            cage, trial = label.split("/", 1)
            source_folder = session / cage / "pose-2d"
            output_folder = session / cage / "pose-2d-filtered"
            sources = sorted(source_folder.glob(f"{trial}*.h5"))
            if len(sources) != self.setup["anipose"]["num_cams"]:
                result.skipped_trials[label] = (
                    f"Need {self.setup['anipose']['num_cams']} converted camera files; "
                    f"found {len(sources)}"
                )
                continue
            pairs = [(source, output_folder / source.name) for source in sources]
            state = self._read_state(output_folder / ".pipeline-filter.json")
            settings_match = state.get("settings_digest") == digest
            valid = settings_match and all(
                self._valid_pose2d(output)
                and output.stat().st_mtime_ns >= source.stat().st_mtime_ns
                for source, output in pairs
            )
            if valid:
                result.outputs.extend(output for _, output in pairs)
                result.ready_trials.append(label)
                continue
            for _, output in pairs:
                if output.exists():
                    # Filtered tracks are reproducible from pose-2d and can be
                    # large, so replace stale copies instead of accumulating them.
                    output.unlink()
            pending.append((label, pairs))
            touched_cages.add(cage)

        if pending:
            result.anipose_output = self._run_anipose("filter") or ""
            for label, pairs in pending:
                outputs = [output for _, output in pairs]
                if all(self._valid_pose2d(output) for output in outputs):
                    result.outputs.extend(outputs)
                    result.ready_trials.append(label)
                else:
                    result.skipped_trials[label] = (
                        "Anipose did not create every filtered camera HDF5"
                    )
            for cage in touched_cages:
                self._write_state(
                    session / cage / "pose-2d-filtered" / ".pipeline-filter.json",
                    {"settings_digest": digest},
                )
        return result

    @staticmethod
    def _valid_pose3d(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            frame = pd.read_csv(path, nrows=1)
        except (OSError, ValueError, pd.errors.ParserError):
            return False
        columns = set(frame.columns)
        return (len(frame) == 1 and "fnum" in columns
                and any(name.endswith("_x") and name[:-2] + "_y" in columns
                        and name[:-2] + "_z" in columns for name in columns))

    def triangulate(self, session_dir: str | Path,
                    inputs: ConversionResult | FilteringResult) -> TriangulationResult:
        session = Path(session_dir)
        result = TriangulationResult(skipped_trials=dict(inputs.skipped_trials))
        pending: list[tuple[Path, str]] = []
        digest = self._settings_digest({
            "filter": self.setup["filter"],
            "triangulation": self.setup["anipose"],
        })
        input_folder_name = (
            "pose-2d-filtered" if self.setup["filter"]["enabled"] else "pose-2d"
        )
        touched_cages: set[str] = set()
        for label in inputs.ready_trials:
            cage, trial = label.split("/", 1)
            output = session / cage / "pose-3d" / f"{trial}.csv"
            state = self._read_state(output.parent / ".pipeline-triangulation.json")
            sources = list((session / cage / input_folder_name).glob(f"{trial}*.h5"))
            calibration = session / cage / "calibration" / "calibration.toml"
            dependencies = sources + ([calibration] if calibration.is_file() else [])
            latest_input = max(
                (path.stat().st_mtime_ns for path in dependencies), default=0
            )
            valid = (
                state.get("settings_digest") == digest
                and self._valid_pose3d(output)
                and output.stat().st_mtime_ns >= latest_input
            )
            if valid:
                result.outputs.append(output)
            else:
                if output.exists():
                    self._archive(output)
                preview = session / cage / "videos-3d" / f"{trial}.mp4"
                if preview.exists():
                    self._archive(preview)
                pending.append((output, cage))
                touched_cages.add(cage)
        if pending:
            anipose_output = self._run_anipose("triangulate") or ""
            missing_after_run = False
            for output, _ in pending:
                label = f"{output.parent.parent.name}/{output.stem}"
                if self._valid_pose3d(output):
                    result.outputs.append(output)
                else:
                    missing_after_run = True
                    result.skipped_trials[label] = "Anipose did not create a 3D CSV"
            if missing_after_run:
                result.anipose_output = anipose_output
            for cage in touched_cages:
                self._write_state(
                    session / cage / "pose-3d" / ".pipeline-triangulation.json",
                    {"settings_digest": digest},
                )
        return result

    def render_3d(self, session_dir: str | Path) -> VisualizationResult:
        """Render optional Anipose skeleton videos for existing 3D CSV files."""
        session = Path(session_dir)
        result = VisualizationResult()
        pending: list[tuple[str, Path, Path, Path]] = []
        extension = self.setup["pose"]["video_extension"]
        for cage in self.setup["cages"]:
            pose_folder = session / cage / "pose-3d"
            for pose_csv in sorted(pose_folder.glob("*.csv")):
                label = f"{cage}/{pose_csv.stem}"
                output = session / cage / "videos-3d" / f"{pose_csv.stem}.mp4"
                if output.is_file() and output.stat().st_size > 0:
                    result.outputs.append(output)
                else:
                    videos = sorted(
                        (session / cage / "videos-raw").glob(
                            f"{pose_csv.stem}*.{extension.lstrip('.')}"
                        )
                    )
                    if not videos:
                        result.skipped_trials[label] = (
                            "No matching raw video is available to determine source FPS"
                        )
                        continue
                    pending.append((label, pose_csv, videos[0], output))
        if not pending:
            return result

        missing_tools = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
        if missing_tools:
            raise RuntimeError(
                "Anipose label-3d requires the FFmpeg executables on PATH; missing: "
                + ", ".join(missing_tools)
                + ". Load the HPC FFmpeg module or install the FFmpeg system binaries. "
                  "Installing a Python package named ffmpeg is not sufficient."
            )

        preview_fps = float(self.setup["visualization"]["preview_fps"])
        diagnostics: list[str] = []
        for label, pose_csv, source_video, output in pending:
            try:
                diagnostic = self._render_preview(
                    pose_csv, source_video, output, preview_fps
                )
            except Exception:
                output.unlink(missing_ok=True)
                raise
            if diagnostic:
                diagnostics.append(diagnostic)
            if output.is_file() and output.stat().st_size > 0:
                result.outputs.append(output)
            else:
                result.skipped_trials[label] = "Anipose did not create a 3D video"
        result.anipose_output = "\n".join(diagnostics)
        return result
