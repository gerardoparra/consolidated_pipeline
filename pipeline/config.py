"""Load the single setup file and materialize tool-specific settings."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Cannot put {type(value).__name__} in Anipose config")


def _toml_section(name: str, settings: dict[str, Any]) -> str:
    return f"[{name}]\n" + "\n".join(
        f"{key} = {_toml_value(value)}" for key, value in settings.items()
    ) + "\n"


def _write_if_changed(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    return path


def hpc_repository_paths(hpc: dict[str, Any]) -> tuple[PurePosixPath, PurePosixPath, PurePosixPath]:
    """Return the configured checkout, setup file, and Slurm worker script."""
    repository = PurePosixPath(hpc["repository_dir"])
    return (
        repository,
        repository / "config" / "setup.json",
        repository / "pipeline" / "dlc_sbatch_superanimal.sh",
    )


@dataclass(frozen=True)
class Setup:
    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Setup":
        source = Path(path).expanduser().resolve()
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load setup JSON {source}: {exc}") from exc
        for key in ("hosts", "cages", "calibration", "evaluation", "pose", "anipose", "slurm"):
            if not isinstance(data.get(key), dict):
                raise ValueError(f"setup.json needs an object named {key!r}")
        if data.get("experiment") not in {"mechanical_lockbox", "sliding_lockbox"}:
            raise ValueError("experiment must be mechanical_lockbox or sliding_lockbox")
        if not data["cages"]:
            raise ValueError("At least one cage/box camera map is required")
        for host_name in ("local", "hpc"):
            host = data["hosts"].get(host_name)
            if not isinstance(host, dict):
                raise ValueError(f"setup.json needs hosts.{host_name}")
            for key in ("experiment_dir", "model_folder", "anipose_command"):
                if not host.get(key):
                    raise ValueError(f"hosts.{host_name}.{key} is required")
            if not isinstance(host["anipose_command"], list):
                raise ValueError(f"hosts.{host_name}.anipose_command must be a list")
        for key in ("repository_dir", "sandbox", "cache_dir", "tmp_dir", "singularity_module"):
            if not data["hosts"]["hpc"].get(key):
                raise ValueError(f"hosts.hpc.{key} is required")
        perspectives = {"top", "front", "side", "left", "right", "back"}
        for cage, mapping in data["cages"].items():
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"Camera map for {cage} must be a non-empty object")
            if set(mapping.values()) - perspectives:
                raise ValueError(f"Camera map for {cage} has unsupported perspectives")
        if data["calibration"].get("fps", 0) <= 0:
            raise ValueError("calibration.fps must be positive")
        if data["anipose"].get("num_cams", 0) < 2:
            raise ValueError("anipose.num_cams must be at least two")
        camera_pattern = data["anipose"].get("cam_regex", "")
        try:
            if re.compile(camera_pattern).groups != 1:
                raise ValueError("anipose.cam_regex must have one capture group")
        except re.error as exc:
            raise ValueError(f"Invalid anipose.cam_regex: {exc}") from exc
        for cage, mapping in data["cages"].items():
            if len(mapping) != data["anipose"]["num_cams"]:
                raise ValueError(f"{cage} needs {data['anipose']['num_cams']} cameras")
            if any(re.search(camera_pattern, token) is None for token in mapping):
                raise ValueError(f"Camera names for {cage} must match anipose.cam_regex")
        return cls(source, data)

    def host_name(self, environment: str = "auto") -> str:
        name = ("local" if os.name == "nt" else "hpc") if environment == "auto" else environment
        if name not in self.data["hosts"]:
            raise ValueError(f"Unknown environment {name!r}")
        return name

    def host(self, environment: str = "auto") -> dict[str, Any]:
        return self.data["hosts"][self.host_name(environment)]

    def experiment_dir(self, environment: str = "auto", override: str | Path | None = None) -> Path:
        return Path(override or self.host(environment)["experiment_dir"]).expanduser().resolve()

    def anipose_config_text(self, environment: str = "auto") -> str:
        cfg = self.data["anipose"]
        cal = self.data["calibration"]
        pose = self.data["pose"]
        host = self.host(environment)
        header = {
            "project": cfg["project"],
            "model_folder": host["model_folder"],
            "nesting": cfg["nesting"],
            "video_extension": pose["calibration_video_extension"],
        }
        content = "# Generated from config/setup.json; edit setup.json instead.\n"
        content += "\n".join(f"{key} = {_toml_value(value)}" for key, value in header.items()) + "\n\n"
        content += _toml_section("pipeline", {
            "videos_raw": "videos-raw", "pose_2d": "pose-2d", "pose_3d": "pose-3d",
            "calibration_videos": "calibration", "calibration_results": "calibration",
        }) + "\n"
        content += _toml_section("calibration", {
            key: cal[key] for key in ("board_type", "board_size", "board_marker_bits",
                                  "board_marker_dict_number", "board_square_side_length", "board_marker_length")
        }) + "\n"
        content += _toml_section("manual_verification", {"manually_verify": cal["manually_verify"]}) + "\n"
        content += _toml_section("labeling", {"scheme": cfg["scheme"]}) + "\n"
        content += _toml_section("filter", {"enabled": False}) + "\n"
        content += _toml_section("triangulation", {
            key: cfg[key] for key in ("cam_regex", "num_cams", "ransac", "optim",
                                   "constraints", "reproj_error_threshold", "score_threshold")
        })
        return content

    def materialize(self, experiment_dir: Path, environment: str = "auto",
                    session_name: str | None = None) -> dict[str, Path]:
        root = Path(experiment_dir)
        config = _write_if_changed(root / "config.toml", self.anipose_config_text(environment))
        cage_map = _write_if_changed(
            root / "cage_map.json", json.dumps(self.data["cages"], indent=2) + "\n"
        )
        hpc = self.host("hpc")
        _, _, wrapper = hpc_repository_paths(hpc)
        video_root = PurePosixPath(hpc["experiment_dir"])
        if session_name:
            video_root /= session_name
        jobs = {"mlb2_experiment": {
            "video_root": str(video_root),
            "extension": self.data["pose"]["video_extension"],
            "result_root": str(video_root),
            "cage_map_file": str(PurePosixPath(hpc["experiment_dir"]) / "cage_map.json"),
            "wrapper": str(wrapper),
            "job_name": self.data["slurm"]["job_name"],
            "experiment_layout": True,
            "skip_existing": self.data["pose"]["skip_existing"],
            "inference_batch_size": self.data["pose"]["inference_batch_size"],
            "detector_batch_size": self.data["pose"]["detector_batch_size"],
            "adapt_batch_size": self.data["pose"]["adapt_batch_size"],
        }}
        job_file = _write_if_changed(root / "jobs.json", json.dumps(jobs, indent=2) + "\n")
        return {"anipose": config, "cages": cage_map, "jobs": job_file}
