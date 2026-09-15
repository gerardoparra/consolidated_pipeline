import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.config import Setup
from pipeline.main import run_pose_detection_pipeline
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
                (tracks / f"{video.stem}_after_adapt.json").write_text("{}")
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

    def test_calibration_handoff_reports_missing_cage(self):
        session = self.root / "session"
        calibration = session / "CAGE1" / "calibration"
        calibration.mkdir(parents=True)
        for token in Setup.load(SETUP_PATH).data["cages"]["CAGE1"]:
            (calibration / f"20260811_{token}_3fps.avi").write_bytes(b"video")

        def make_calibration(processor, action):
            self.assertEqual(action, "calibrate")
            (calibration / "calibration.toml").write_text("done")

        with patch.object(PoseProcessor, "_run_anipose", make_calibration):
            result = PoseProcessor(self.root, Setup.load(SETUP_PATH).data).calibrate_cameras(session)
        self.assertEqual(len(result.tomls), 1)
        self.assertIn("CAGE2", result.skipped_cages)


if __name__ == "__main__":
    unittest.main()
