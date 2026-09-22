"""SuperAnimal pose inference, track conversion, and 3D triangulation."""

from __future__ import annotations

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


class PoseProcessor(AniposeRunner):

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

    def triangulate(self, session_dir: str | Path, conversion: ConversionResult) -> TriangulationResult:
        session = Path(session_dir)
        result = TriangulationResult(skipped_trials=dict(conversion.skipped_trials))
        pending: list[Path] = []
        for label in conversion.ready_trials:
            cage, trial = label.split("/", 1)
            output = session / cage / "pose-3d" / f"{trial}.csv"
            if self._valid_pose3d(output):
                result.outputs.append(output)
            else:
                if output.exists():
                    archived = output.with_name(output.name + ".invalid")
                    suffix = 1
                    while archived.exists():
                        archived = output.with_name(output.name + f".invalid.{suffix}")
                        suffix += 1
                    output.rename(archived)
                pending.append(output)
        if pending:
            anipose_output = self._run_anipose("triangulate") or ""
            missing_after_run = False
            for output in pending:
                label = f"{output.parent.parent.name}/{output.stem}"
                if self._valid_pose3d(output):
                    result.outputs.append(output)
                else:
                    missing_after_run = True
                    result.skipped_trials[label] = "Anipose did not create a 3D CSV"
            if missing_after_run:
                result.anipose_output = anipose_output
        return result
