"""Reusable, dry-run-first helpers for submitting SuperAnimal Slurm jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from typing import Iterable


MAP_PRESETS = frozenset({"default", "rats", "mlb2_cage1", "mlb2_cage2"})


@dataclass(frozen=True)
class JobSpec:
    """Inputs accepted by the SuperAnimal Slurm wrappers."""

    video_root: str | Path
    extension: str
    result_root: str | Path
    perspective_map: str | Path = "default"
    wrapper: str | Path = "dlc_sbatch_gera.sh"
    job_name: str | None = None
    cage_map_file: str | Path | None = None
    experiment_layout: bool = False
    skip_existing: bool = False
    inference_batch_size: int | None = None
    detector_batch_size: int | None = None
    adapt_batch_size: int | None = None

    def validated(self) -> "JobSpec":
        video_root = Path(self.video_root).expanduser().resolve()
        result_root = Path(self.result_root).expanduser().resolve()
        wrapper = Path(self.wrapper).expanduser().resolve()
        if not video_root.is_dir():
            raise ValueError(f"Video root is not a directory: {video_root}")
        if not self.extension or not self.extension.strip("."):
            raise ValueError("A non-empty video extension is required.")
        if not wrapper.is_file():
            raise ValueError(f"Slurm wrapper does not exist: {wrapper}")
        batch_sizes = (self.inference_batch_size, self.detector_batch_size, self.adapt_batch_size)
        if any(
            size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1)
            for size in batch_sizes
        ):
            raise ValueError("Configured batch sizes must be positive integers.")
        map_value = str(self.perspective_map)
        if map_value not in MAP_PRESETS:
            map_path = Path(map_value).expanduser().resolve()
            if not map_path.is_file():
                raise ValueError(f"Perspective map is neither a preset nor a file: {map_path}")
            map_value = str(map_path)
        cage_map_file = None
        if self.cage_map_file is not None:
            cage_path = Path(self.cage_map_file).expanduser().resolve()
            if not cage_path.is_file():
                raise ValueError(f"Cage map file does not exist: {cage_path}")
            cage_map_file = str(cage_path)
        return JobSpec(
            video_root, self.extension, result_root, map_value, wrapper, self.job_name,
            cage_map_file, self.experiment_layout, self.skip_existing,
            self.inference_batch_size, self.detector_batch_size, self.adapt_batch_size,
        )


def build_sbatch_command(job: JobSpec) -> list[str]:
    """Return a validated ``sbatch`` command without running it."""
    spec = job.validated()
    command = ["sbatch"]
    exported = ["ALL"]
    if spec.experiment_layout:
        exported.append("SUPERANIMAL_EXPERIMENT_LAYOUT=1")
    if spec.skip_existing:
        exported.append("SUPERANIMAL_SKIP_EXISTING=1")
    for variable, value in (
        ("SUPERANIMAL_INFERENCE_BATCH_SIZE", spec.inference_batch_size),
        ("SUPERANIMAL_DETECTOR_BATCH_SIZE", spec.detector_batch_size),
        ("SUPERANIMAL_ADAPT_BATCH_SIZE", spec.adapt_batch_size),
    ):
        if value is not None:
            exported.append(f"{variable}={value}")
    if len(exported) > 1:
        command.append(f"--export={','.join(exported)}")
    if spec.job_name:
        command.extend(["--job-name", spec.job_name])
    command.extend([str(spec.wrapper), str(spec.video_root), spec.extension, str(spec.result_root), str(spec.perspective_map)])
    if spec.cage_map_file:
        command.append(str(spec.cage_map_file))
    return command


def preview_job(job: JobSpec) -> str:
    """Render the exact shell-safe command that would be submitted."""
    return shlex.join(build_sbatch_command(job))


def submit_job(job: JobSpec, *, submit: bool = False) -> str | subprocess.CompletedProcess[str]:
    """Preview by default; set ``submit=True`` to call Slurm."""
    command = build_sbatch_command(job)
    if not submit:
        return shlex.join(command)
    return subprocess.run(command, check=True, text=True, capture_output=True)


def submit_many(jobs: Iterable[JobSpec], *, submit: bool = False) -> list[str] | list[subprocess.CompletedProcess[str]]:
    """Validate and preview all jobs before optionally submitting any of them."""
    job_list = list(jobs)
    commands = [build_sbatch_command(job) for job in job_list]
    previews = [shlex.join(command) for command in commands]
    if not submit:
        return previews
    for preview in previews:
        print(preview)
    return [subprocess.run(command, check=True, text=True, capture_output=True) for command in commands]
