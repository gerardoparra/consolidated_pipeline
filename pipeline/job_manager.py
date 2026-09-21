"""Preview, submit, and remember one SuperAnimal Slurm job per experiment."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import hpc_repository_paths


def _is_windows() -> bool:
    return os.name == "nt"


@dataclass(frozen=True)
class JobResult:
    status: str
    command: str
    job_id: str | None = None


class JobManager:
    def __init__(self, experiment_dir: str | Path, setup: dict,
                 session_dir: str | Path | None = None):
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.setup = setup
        self.session_dir = Path(session_dir).expanduser().resolve() if session_dir else None

    @property
    def state_path(self) -> Path:
        return (self.session_dir or self.experiment_dir) / "pipeline_state.json"

    def _saved_job_id(self) -> str | None:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8")).get("pose_job_id")
        except (OSError, json.JSONDecodeError):
            return None

    def _save_job_id(self, job_id: str) -> None:
        self.state_path.write_text(json.dumps({"pose_job_id": job_id}, indent=2) + "\n", encoding="utf-8")

    def command(self) -> list[str]:
        hpc = self.setup["hosts"]["hpc"]
        pose = self.setup["pose"]
        slurm = self.setup["slurm"]
        root = PurePosixPath(hpc["experiment_dir"])
        if self.session_dir:
            root /= self.session_dir.name
        repository, setup_json, wrapper = hpc_repository_paths(hpc)
        exports = ["ALL"] + [f"{key}={value}" for key, value in (
            ("PIPELINE_REPOSITORY", repository),
            ("PIPELINE_SETUP", setup_json),
            ("PIPELINE_SANDBOX", hpc["sandbox"]),
            ("PIPELINE_CACHE", hpc["cache_dir"]),
            ("PIPELINE_TMP", hpc["tmp_dir"]),
            ("PIPELINE_MODULE", hpc["singularity_module"]),
            ("PIPELINE_INFERENCE_BATCH_SIZE", pose["inference_batch_size"]),
            ("PIPELINE_DETECTOR_BATCH_SIZE", pose["detector_batch_size"]),
            ("PIPELINE_ADAPT_BATCH_SIZE", pose["adapt_batch_size"]),
        )]
        return [
            "sbatch", "--job-name", slurm["job_name"],
            "--partition", slurm["partition"],
            "--gres", slurm["gres"],
            "--constraint", slurm["constraint"],
            "--time", slurm["time"],
            "--chdir", str(repository),
            "--output", str(repository / "logs" / "submission-%x.%j.out"),
            "--error", str(repository / "logs" / "submission-%x.%j.err"),
            "--export=" + ",".join(exports),
            str(wrapper), str(root), pose["video_extension"], str(root),
        ]

    def preview(self) -> JobResult:
        command = shlex.join(self.command())
        saved = self._saved_job_id()
        if saved:
            return JobResult("pending", command, saved)
        return JobResult("preview", command)

    def submit(self, *, retry: bool = False) -> JobResult:
        if _is_windows():
            raise RuntimeError("Slurm submission must run on the HPC host; use preview on Windows")
        saved = self._saved_job_id()
        if saved and not retry:
            return JobResult("pending", shlex.join(self.command()), saved)
        hpc = self.setup["hosts"]["hpc"]
        repository, setup_json, wrapper = hpc_repository_paths(hpc)
        root = Path(hpc["experiment_dir"])
        if self.session_dir:
            root /= self.session_dir.name
        for path in (root, Path(repository), Path(wrapper), Path(setup_json), Path(hpc["sandbox"])):
            if not path.exists():
                raise FileNotFoundError(f"HPC job dependency is missing: {path}")
        Path(repository / "logs").mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(self.command(), check=True, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError("sbatch is unavailable; run this command on an HPC login node") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"sbatch failed: {exc.stderr or exc.stdout}") from exc
        match = re.search(r"Submitted batch job (\d+)", completed.stdout)
        if not match:
            raise RuntimeError(f"Could not read a Slurm job ID from: {completed.stdout!r}")
        self._save_job_id(match.group(1))
        return JobResult("submitted", shlex.join(self.command()), match.group(1))
