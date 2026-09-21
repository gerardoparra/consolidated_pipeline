"""Public end-to-end API and command-line entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from .calibration_processor import CalibrationProcessor, CalibrationResult
from .config import Setup
from .conversion import ConversionResult
from .file_handler import FileHandler
from .job_manager import JobManager, JobResult
from .pose_processor import PoseProcessor, TriangulationResult
from .video_processor import VideoProcessor


DEFAULT_SETUP = Path(__file__).resolve().parent.parent / "config" / "setup.json"


@dataclass
class PipelineRunResult:
    status: str
    session_dir: Path
    calibration: CalibrationResult
    job: JobResult | None = None
    conversion: ConversionResult = field(default_factory=ConversionResult)
    triangulation: TriangulationResult = field(default_factory=TriangulationResult)
    pending_videos: list[Path] = field(default_factory=list)


def run_pose_detection_pipeline(
    source_path: str | Path,
    experiment_dir: str | Path | None,
    setup_json: str | Path = DEFAULT_SETUP,
    *,
    submit_jobs: bool = False,
    environment: str = "auto",
) -> PipelineRunResult:
    """Advance one session as far as completed outputs and requested jobs allow.

    Preview is the default for Slurm. Reinvoke after the job finishes to convert
    2D tracks and triangulate complete trials. Manual-label calibration evaluation
    is available only through the separate ``evaluate`` command.
    """
    setup = Setup.load(setup_json)
    root = setup.experiment_dir(environment, experiment_dir)
    file_handler = FileHandler(root, setup.data)
    video_processor = VideoProcessor(root, setup.data)
    calibration_processor = CalibrationProcessor(root, setup.data, environment=setup.host_name(environment))
    pose_processor = PoseProcessor(root, setup.data, environment=setup.host_name(environment))
    session = file_handler.import_raw_data(source_path)
    job_manager = JobManager(root, setup.data, session)
    video_processor.prepare(session)
    video_processor.downsample(session)
    setup.materialize(root, environment, session.name)
    calibration = calibration_processor.calibrate_cameras(session)

    pending = file_handler.pending_tracks(session)
    job = None
    if pending:
        job = job_manager.submit() if submit_jobs else job_manager.preview()
    conversion = pose_processor.convert_dlc_output_to_anipose(session)
    triangulation = pose_processor.triangulate(session, conversion)
    if not file_handler.raw_videos(session):
        status = "empty_session"
    elif pending:
        status = "partial" if triangulation.outputs else "pending_pose"
    elif conversion.skipped_trials or triangulation.skipped_trials or calibration.skipped_cages:
        status = "partial"
    else:
        status = "complete"
    return PipelineRunResult(
        status, session, calibration, job, conversion, triangulation, pending
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_SETUP,
                        help="Single setup JSON (default: config/setup.json)")
    parser.add_argument("--environment", choices=("auto", "local", "hpc"), default="auto")
    parser.add_argument("--experiment-dir", type=Path,
                        help="Override this host's experiment directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLB2 calibration and pose pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "Advance a raw or prepared session through all ready stages"),
        ("prepare", "Import, organize, and downsample a session"),
        ("downsample", "Resume downsampling an organized session"),
        ("calibrate", "Run missing Anipose camera calibrations"),
        ("labels", "Extract raw-video frames for manual reprojection labeling"),
        ("evaluate", "Evaluate camera calibration from manual label CSVs"),
        ("pose", "Preview or submit a SuperAnimal Slurm job"),
        ("convert", "Stage complete SuperAnimal trials in pose-2d"),
        ("triangulate", "Triangulate ready pose-2d trials"),
    ):
        sub = commands.add_parser(name, help=help_text)
        _add_common(sub)
        if name in {"run", "prepare"}:
            sub.add_argument("source", type=Path, help="Zip, raw session folder, or prepared session")
        else:
            sub.add_argument("session", type=Path, help="Prepared session directory")
        if name in {"run", "pose"}:
            sub.add_argument("--submit", action="store_true", help="Submit Slurm job on HPC")
        if name == "pose":
            sub.add_argument("--retry-job", action="store_true", help="Submit again despite saved job ID")
        if name == "evaluate":
            sub.add_argument("--labels", type=Path, help="Manual label CSV directory")
        if name == "labels":
            sub.add_argument("--frame-index", type=int, default=100)
            sub.add_argument("--dlc-project", type=Path, help="Destination DLC project")
    worker = commands.add_parser("worker", help="Internal DLC GPU worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--environment", choices=("local", "hpc"), default="hpc")
    worker.add_argument("--video-root", type=Path, required=True)
    worker.add_argument("--result-root", type=Path, required=True)
    worker.add_argument("--video-extension", required=True)
    worker.add_argument("--inference-batch-size", type=int)
    worker.add_argument("--detector-batch-size", type=int)
    worker.add_argument("--adapt-batch-size", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_pose_detection_pipeline(
            args.source, args.experiment_dir, args.config,
            submit_jobs=args.submit, environment=args.environment,
        )
        print(f"{result.status}: {result.session_dir}")
        print(f"Camera calibrations: {len(result.calibration.tomls)}")
        if result.job:
            print(f"Slurm {result.job.status}: {result.job.command}")
            if result.job.job_id:
                print(f"Job ID: {result.job.job_id}")
        print(f"2D videos pending: {len(result.pending_videos)}; 3D trials: {len(result.triangulation.outputs)}")
        for trial, reason in sorted(result.triangulation.skipped_trials.items()):
            print(f"[skip] {trial}: {reason}")
        return

    setup = Setup.load(args.config)
    if args.command == "worker":
        PoseProcessor(args.video_root, setup.data, environment=args.environment).detect_keypoints(
            args.video_root, args.result_root, args.video_extension,
            args.inference_batch_size, args.detector_batch_size, args.adapt_batch_size,
        )
        return
    root = setup.experiment_dir(args.environment, args.experiment_dir)
    files = FileHandler(root, setup.data)
    video = VideoProcessor(root, setup.data)
    calibration = CalibrationProcessor(root, setup.data, environment=setup.host_name(args.environment))
    pose = PoseProcessor(root, setup.data, environment=setup.host_name(args.environment))
    if args.command == "prepare":
        session = files.import_raw_data(args.source)
        video.prepare(session)
        video.downsample(session)
        setup.materialize(root, args.environment, session.name)
        print(f"Prepared: {session}")
        return
    session = args.session.expanduser().resolve()
    if not session.is_dir():
        parser.error(f"Session directory does not exist: {session}")
    if args.command == "downsample":
        print(f"Downsampled {len(video.downsample(session))} calibration directories")
        return
    if args.command != "labels":
        setup.materialize(root, args.environment, session.name)
    if args.command == "calibrate":
        result = calibration.calibrate_cameras(session)
        print(f"Camera calibrations: {len(result.tomls)}")
        for cage, reason in result.skipped_cages.items():
            print(f"[skip] {cage}: {reason}")
    elif args.command == "labels":
        from .video_utils import extract_frame
        project = args.dlc_project or Path(setup.host(args.environment)["model_folder"])
        outputs = extract_frame(project, files.raw_videos(session), args.frame_index)
        print(f"Manual label frame folders: {len(outputs)}")
    elif args.command == "evaluate":
        result = calibration.evaluate(session, files.calibration_tomls(session), args.labels)
        print(f"Quality: {result.status}; mean error: {result.mean_error_px}; summary: {result.summary_csv}")
    elif args.command == "pose":
        pending = files.pending_tracks(session)
        if not pending:
            print("All 2D tracks already exist")
        else:
            manager = JobManager(root, setup.data, session)
            result = manager.submit(retry=args.retry_job) if args.submit else manager.preview()
            print(f"{len(pending)} videos pending; Slurm {result.status}: {result.command}")
            if result.job_id:
                print(f"Job ID: {result.job_id}")
    elif args.command == "convert":
        result = pose.convert_dlc_output_to_anipose(session)
        print(f"Converted {len(result.converted)} files; ready trials: {len(result.ready_trials)}")
        for trial, reason in sorted(result.skipped_trials.items()):
            print(f"[skip] {trial}: {reason}")
    elif args.command == "triangulate":
        converted = pose.convert_dlc_output_to_anipose(session)
        result = pose.triangulate(session, converted)
        print(f"3D CSVs: {len(result.outputs)}")
        for trial, reason in sorted(result.skipped_trials.items()):
            print(f"[skip] {trial}: {reason}")


if __name__ == "__main__":
    main()
