"""Shared Anipose command execution for calibration and triangulation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class AniposeRunner:
    def __init__(self, experiment_dir: str | Path, setup: dict, *, environment: str = "auto"):
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.setup = setup
        self.environment = "local" if environment == "auto" and os.name == "nt" else (
            "hpc" if environment == "auto" else environment
        )

    def _anipose_command(self) -> list[str]:
        command = self.setup["hosts"][self.environment]["anipose_command"]
        if not isinstance(command, list) or not command:
            raise ValueError("Host anipose_command must be a non-empty list")
        executable = str(command[0])
        if Path(executable).is_absolute():
            if not Path(executable).is_file():
                raise FileNotFoundError(f"Anipose executable is missing: {executable}")
        elif shutil.which(executable) is None:
            raise RuntimeError(f"Anipose command {executable!r} is unavailable on this host")
        return [str(part) for part in command]

    def _run_anipose(self, action: str) -> None:
        command = self._anipose_command() + [action]
        try:
            completed = subprocess.run(
                command, cwd=self.experiment_dir, text=True, capture_output=True
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start Anipose {action}: {exc}") from exc
        if completed.returncode:
            raise RuntimeError(
                f"Anipose {action} failed (exit {completed.returncode}).\n"
                f"{completed.stderr or completed.stdout or 'No output from Anipose.'}"
            )
