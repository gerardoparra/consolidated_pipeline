"""Submit pipeline stages using the shared Slurm settings."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
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
    output_log: str | None = None
    error_log: str | None = None


def _submit(command: list[str], script: str | None = None) -> str:
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True,
                                   **({"input": script} if script is not None else {}))
    except FileNotFoundError as exc:
        raise RuntimeError("sbatch is unavailable; run this command on an HPC login node") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"sbatch failed: {exc.stderr or exc.stdout}") from exc
    match = re.fullmatch(r"(?:Submitted batch job )?(\d+)(?:;[^\s;]+)?", completed.stdout.strip())
    if not match:
        raise RuntimeError(f"Could not read a Slurm job ID from: {completed.stdout!r}")
    return match.group(1)


def _slurm_options(setup: dict) -> list[str]:
    """Use the same resource settings for host stages and container tracking."""
    options = []
    for key, flag in (("job_name", "--job-name"), ("partition", "--partition"),
                      ("gres", "--gres"), ("constraint", "--constraint"),
                      ("time", "--time"), ("cpus", "--cpus-per-task"),
                      ("memory", "--mem")):
        value = setup["slurm"].get(key)
        if value is not None:
            options.extend([flag, str(value)])
    return options


def submit_stage(setup: dict, stage: str, arguments: list[str]) -> JobResult:
    """Queue a host-Python stage without running any session processing."""
    if _is_windows():
        raise RuntimeError("Slurm submission must run on the HPC host")
    hpc = setup["hosts"]["hpc"]
    repository = Path(hpc["repository_dir"])
    python = Path(hpc.get("python_executable") or sys.executable)
    for path in (repository, repository / "pipeline" / "main.py", python):
        if not path.exists():
            raise FileNotFoundError(f"HPC job dependency is missing: {path}")
    logs = repository / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = str(logs / f"{stage}-%j.out")
    error = str(logs / f"{stage}-%j.err")
    command = ["sbatch", "--parsable", *_slurm_options(setup),
               "--chdir", str(repository), "--output", output, "--error", error,
               "--export=ALL"]
    worker = [str(python), "-m", "pipeline.main", stage, *arguments]
    script = "#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(worker) + "\n"
    job_id = _submit(command, script)
    # Separate records also work for ZIPs and raw folders that the worker moves.
    record = logs / f"submission-{job_id}.json"
    record.write_text(json.dumps({"stage": stage, "job_id": job_id,
                                 "command": worker}, indent=2) + "\n", encoding="utf-8")
    return JobResult("submitted", shlex.join(command), job_id,
                     output.replace("%j", job_id), error.replace("%j", job_id))


class JobManager:
    def __init__(self, experiment_dir: str | Path, setup: dict,
                 session_dir: str | Path | None = None, *, setup_path: Path | None = None):
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.setup = setup
        self.session_dir = Path(session_dir).expanduser().resolve() if session_dir else None
        self.setup_path = setup_path

    @property
    def state_path(self) -> Path:
        return (self.session_dir or self.experiment_dir) / "pipeline_state.json"

    def _saved_job_id(self) -> str | None:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8")).get("pose_job_id")
        except (OSError, json.JSONDecodeError):
            return None

    def _save_job_id(self, job_id: str) -> None:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        state["pose_job_id"] = job_id
        self.state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def _root(self) -> Path | PurePosixPath:
        if self.setup_path is not None:
            return self.session_dir or self.experiment_dir
        root = PurePosixPath(self.setup["hosts"]["hpc"]["experiment_dir"])
        return root / self.session_dir.name if self.session_dir else root

    def _result(self, status: str, job_id: str) -> JobResult:
        command = self.command()
        name = self.setup["slurm"]["job_name"]
        logs = [command[command.index(flag) + 1].replace("%x", name).replace("%j", job_id)
                for flag in ("--output", "--error")]
        return JobResult(status, shlex.join(command), job_id, *logs)

    def command(self) -> list[str]:
        hpc = self.setup["hosts"]["hpc"]
        pose = self.setup["pose"]
        root = self._root()
        repository, setup_json, wrapper = hpc_repository_paths(hpc)
        setup_json = self.setup_path or setup_json
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
            "sbatch", "--parsable", *_slurm_options(self.setup),
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
            return self._result("pending", saved)
        hpc = self.setup["hosts"]["hpc"]
        repository, setup_json, wrapper = hpc_repository_paths(hpc)
        setup_json = self.setup_path or setup_json
        root = Path(self._root())
        for path in (root, Path(repository), Path(wrapper), Path(setup_json), Path(hpc["sandbox"])):
            if not path.exists():
                raise FileNotFoundError(f"HPC job dependency is missing: {path}")
        Path(repository / "logs").mkdir(parents=True, exist_ok=True)
        job_id = _submit(self.command())
        self._save_job_id(job_id)
        return self._result("submitted", job_id)
