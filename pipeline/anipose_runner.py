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

    def _run_anipose(self, action: str) -> str:
        command = self._anipose_command() + [action]
        env = None
        if action == "filter":
            # Anipose 1.1.24 calls DataFrame.to_hdf(path, key, ...), while
            # current pandas makes ``key`` keyword-only.  Load a narrowly
            # scoped sitecustomize shim in the Anipose subprocess so users do
            # not need to modify site-packages or downgrade pandas.
            compat_dir = Path(__file__).with_name("_anipose_compat")
            env = os.environ.copy()
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(compat_dir) + (
                os.pathsep + existing if existing else ""
            )
        try:
            completed = subprocess.run(
                command, cwd=self.experiment_dir, text=True,
                capture_output=True, env=env,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start Anipose {action}: {exc}") from exc
        if completed.returncode:
            raise RuntimeError(
                f"Anipose {action} failed (exit {completed.returncode}).\n"
                f"{completed.stderr or completed.stdout or 'No output from Anipose.'}"
            )
        return "\n".join(
            output.strip() for output in (completed.stdout, completed.stderr)
            if output and output.strip()
        )
