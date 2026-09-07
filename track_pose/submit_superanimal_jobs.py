#!/usr/bin/env python3
"""Prepare or submit named SuperAnimal Slurm jobs from a JSON profile file.

Examples:
    python submit_superanimal_jobs.py jobs.json --list
    python submit_superanimal_jobs.py jobs.json mlb2_day1
    python submit_superanimal_jobs.py jobs.json mlb2_day1 --submit
    python submit_superanimal_jobs.py jobs.json --all --submit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from superanimal_submit import JobSpec, preview_job, submit_job, submit_many


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_profile_path(value: str | Path, config_directory: Path) -> Path:
    path = Path(value).expanduser()
    return (config_directory / path if not path.is_absolute() else path).resolve()


def load_profiles(config_file: str | Path) -> tuple[dict[str, dict[str, Any]], Path]:
    """Load named profiles, accepting either a direct object or ``profiles`` key."""
    path = Path(config_file).expanduser().resolve()
    try:
        raw = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"Could not read job profile file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Job profile file {path} is not valid JSON: {exc}") from exc
    profiles = raw.get("profiles", raw) if isinstance(raw, dict) else None
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Job profile file must contain a non-empty object of named profiles.")
    if not all(isinstance(name, str) and isinstance(profile, dict) for name, profile in profiles.items()):
        raise ValueError("Every job profile must have a string name and an object value.")
    return profiles, path.parent


def job_from_profile(name: str, profile: dict[str, Any], config_directory: Path) -> JobSpec:
    """Convert one JSON profile to a validated-job-ready specification."""
    required = ("video_root", "extension", "result_root")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"Profile {name!r} is missing: {', '.join(missing)}")
    wrapper = profile.get("wrapper", "dlc_sbatch_gera.sh")
    wrapper_path = resolve_profile_path(wrapper, config_directory)
    # If omitted, wrappers are supplied by this project rather than the caller's cwd.
    if not Path(wrapper).is_absolute() and not wrapper_path.exists():
        wrapper_path = PROJECT_ROOT / str(wrapper)
    map_value = profile.get("map", profile.get("perspective_map", "default"))
    if isinstance(map_value, str) and map_value not in {"default", "rats", "mlb2_cage1", "mlb2_cage2"}:
        map_value = resolve_profile_path(map_value, config_directory)
    cage_map = profile.get("cage_map_file")
    if cage_map is not None:
        cage_map = resolve_profile_path(cage_map, config_directory)
    return JobSpec(
        video_root=resolve_profile_path(profile["video_root"], config_directory),
        extension=str(profile["extension"]),
        result_root=resolve_profile_path(profile["result_root"], config_directory),
        perspective_map=map_value,
        wrapper=wrapper_path,
        job_name=profile.get("job_name", name),
        cage_map_file=cage_map,
        experiment_layout=bool(profile.get("experiment_layout", False)),
        skip_existing=bool(profile.get("skip_existing", False)),
        inference_batch_size=profile.get("inference_batch_size"),
        detector_batch_size=profile.get("detector_batch_size"),
        adapt_batch_size=profile.get("adapt_batch_size"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview or submit named SuperAnimal Slurm profiles.")
    parser.add_argument("config", help="JSON file containing named job profiles.")
    parser.add_argument("profile", nargs="?", help="Name of one profile to prepare.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Prepare every profile in the config file.")
    selection.add_argument("--list", action="store_true", help="List profile names and exit.")
    parser.add_argument("--submit", action="store_true", help="Submit after previewing; default only prints commands.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        profiles, config_directory = load_profiles(args.config)
        if args.list:
            print(*sorted(profiles), sep="\n")
            return
        if args.all:
            names = list(profiles)
        elif args.profile:
            if args.profile not in profiles:
                raise ValueError(f"Unknown profile {args.profile!r}. Use --list to see available profiles.")
            names = [args.profile]
        else:
            parser.error("provide PROFILE, --all, or --list")
        jobs = [job_from_profile(name, profiles[name], config_directory) for name in names]
        if not args.submit:
            print(*(preview_job(job) for job in jobs), sep="\n")
            return
        if len(jobs) == 1:
            print(preview_job(jobs[0]))
            completed = submit_job(jobs[0], submit=True)
            if completed.stdout:
                print(completed.stdout, end="")
        else:
            completed_jobs = submit_many(jobs, submit=True)
            for completed in completed_jobs:
                if completed.stdout:
                    print(completed.stdout, end="")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
