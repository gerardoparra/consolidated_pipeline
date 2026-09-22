import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.calibration_processor import CalibrationProcessor
from pipeline.config import Setup
from pipeline.main import (
    _print_triangulation_issues,
    build_parser,
    run_pose_detection_pipeline,
)
from pipeline.pose_processor import TriangulationResult
from pipeline.pose_processor import PoseProcessor


SETUP_PATH = Path(__file__).resolve().parent.parent / "config" / "setup.json"


def prepared_session(root: Path, *, tracks_for_cage1: bool = False) -> Path:
    session = root / "11_08_2026_Day7"
    setup = Setup.load(SETUP_PATH).data
    for cage, mapping in setup["cages"].items():
        calibration = session / cage / "calibration"
        raw = session / cage / "videos-raw"
        calibration.mkdir(parents=True)
        raw.mkdir()
        (calibration / "calibration.toml").write_text("existing", encoding="utf-8")
        for token in mapping:
            video = raw / f"20260811_102757_{token}.mkv"
            video.write_bytes(b"video")
            if tracks_for_cage1 and cage == "CAGE1":
                tracks = raw.parent / "tracks"
                tracks.mkdir(exist_ok=True)
                scorer = "top_model" if mapping[token] == "top" else "quadruped_model"
                columns = pd.MultiIndex.from_product(
                    [[scorer], ["single"], ["nose", "tail_base"], ["x", "y", "likelihood"]]
                )
                frame = pd.DataFrame([[1.0] * len(columns), [2.0] * len(columns)], columns=columns)
                frame.to_hdf(tracks / f"{video.stem}DLC_model.h5", key="df")
                (tracks / f"{video.stem}DLC_model_after_adapt.json").write_text("{}")
    return session


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "experiment"
        self.root.mkdir()

    def test_run_previews_pending_pose_and_resumes_without_recalibrating(self):
        session = prepared_session(self.root)
        with patch.object(PoseProcessor, "_run_anipose") as anipose:
            result = run_pose_detection_pipeline(session, self.root, SETUP_PATH)
            again = run_pose_detection_pipeline(session, self.root, SETUP_PATH)
        anipose.assert_not_called()
        self.assertEqual(result.status, "pending_pose")
        self.assertEqual(len(result.pending_videos), 10)
        self.assertEqual(result.job.status, "preview")
        self.assertIn(session.name, result.job.command)
        self.assertEqual(again.status, "pending_pose")

    def test_run_skips_manual_evaluation_even_with_configured_labels(self):
        session = prepared_session(self.root)
        labels = self.root / "labels" / session.name
        labels.mkdir(parents=True)
        (labels / "manual.csv").write_text("placeholder", encoding="utf-8")
        setup_data = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
        setup_data["evaluation"]["label_csv_root"] = str(labels.parent)
        setup_path = self.root / "setup.json"
        setup_path.write_text(json.dumps(setup_data), encoding="utf-8")

        with patch.object(CalibrationProcessor, "evaluate", side_effect=AssertionError("manual-only")):
            result = run_pose_detection_pipeline(session, self.root, setup_path)

        self.assertEqual(result.status, "pending_pose")
        self.assertFalse(hasattr(result, "evaluation"))
        args = build_parser().parse_args(["evaluate", str(session), "--labels", str(labels)])
        self.assertEqual(args.labels, labels)

    def test_ready_trial_converts_then_triangulates_while_other_cage_waits(self):
        session = prepared_session(self.root, tracks_for_cage1=True)
        actions = []

        def complete_triangulation(processor, action):
            actions.append(action)
            output = session / "CAGE1" / "pose-3d"
            output.mkdir()
            (output / "20260811_102757_.csv").write_text("fnum,nose_x,nose_y,nose_z\n0,1,2,3\n")

        with patch.object(PoseProcessor, "_run_anipose", complete_triangulation):
            result = run_pose_detection_pipeline(session, self.root, SETUP_PATH)
        self.assertEqual(actions, ["triangulate"])
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.pending_videos), 5)
        self.assertEqual(len(result.conversion.converted), 5)
        self.assertEqual(len(result.triangulation.outputs), 1)
        converted = session / "CAGE1" / "pose-2d" / "20260811_102757_camera06.h5"
        self.assertEqual(pd.read_hdf(converted).columns.nlevels, 3)
        with patch.object(PoseProcessor, "_run_anipose") as anipose:
            rerun = run_pose_detection_pipeline(session, self.root, SETUP_PATH)
        anipose.assert_not_called()
        self.assertEqual(rerun.conversion.converted, [])

    def test_missing_3d_output_preserves_anipose_diagnostic(self):
        session = prepared_session(self.root, tracks_for_cage1=True)
        processor = PoseProcessor(self.root, Setup.load(SETUP_PATH).data)
        conversion = processor.convert_dlc_output_to_anipose(session)
        diagnostic = "Traceback (most recent call last):\nValueError: camera names do not match"
        with patch.object(processor, "_run_anipose", return_value=diagnostic):
            result = processor.triangulate(session, conversion)
        self.assertFalse(result.outputs)
        self.assertEqual(result.anipose_output, diagnostic)
        self.assertIn("CAGE1/20260811_102757_", result.skipped_trials)

    def test_standalone_triangulation_output_includes_anipose_diagnostic(self):
        result = TriangulationResult(
            skipped_trials={"CAGE2/trial": "Anipose did not create a 3D CSV"},
            anipose_output="ValueError: camera names do not match",
        )
        output = io.StringIO()
        with redirect_stdout(output):
            _print_triangulation_issues(result)
        self.assertIn("[skip] CAGE2/trial", output.getvalue())
        self.assertIn("Anipose output:\nValueError", output.getvalue())

    def test_calibration_handoff_reports_missing_cage(self):
        session = self.root / "session"
        calibration = session / "CAGE1" / "calibration"
        calibration.mkdir(parents=True)
        for token in Setup.load(SETUP_PATH).data["cages"]["CAGE1"]:
            (calibration / f"20260811_{token}_3fps.avi").write_bytes(b"video")

        def make_calibration(processor, action):
            self.assertEqual(action, "calibrate")
            (calibration / "calibration.toml").write_text("done")

        with patch.object(CalibrationProcessor, "_run_anipose", make_calibration):
            result = CalibrationProcessor(self.root, Setup.load(SETUP_PATH).data).calibrate_cameras(session)
        self.assertEqual(len(result.tomls), 1)
        self.assertIn("CAGE2", result.skipped_cages)

    def test_manual_label_reprojection_warns_on_high_calibration_error(self):
        session = prepared_session(self.root)
        labels = self.root / "labels"
        labels.mkdir()
        (labels / "manual.csv").write_text("placeholder", encoding="utf-8")
        calibration_tomls = [session / "CAGE1" / "calibration" / "calibration.toml"]
        summary = pd.DataFrame({"mean_reprojection_error_px": [8.0]})

        with patch("pipeline.reprojection.run_reprojection_batch", return_value=(summary, None)):
            result = CalibrationProcessor(self.root, Setup.load(SETUP_PATH).data).evaluate(
                session, calibration_tomls, labels
            )

        self.assertEqual(result.status, "warning")
        self.assertEqual(result.mean_error_px, 8.0)


if __name__ == "__main__":
    unittest.main()
