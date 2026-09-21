import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from pipeline.config import Setup
from pipeline.conversion import convert_session
from pipeline.job_manager import JobManager
from pipeline.pose_processor import PoseProcessor


SETUP_PATH = Path(__file__).resolve().parent.parent / "config" / "setup.json"


class JobsAndConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.setup = copy.deepcopy(Setup.load(SETUP_PATH).data)

    def test_submission_records_id_and_needs_explicit_retry(self):
        hpc_root = self.root / "hpc_experiment"
        project = self.root / "project"
        session = hpc_root / "day1"
        session.mkdir(parents=True)
        (project / "pipeline").mkdir(parents=True)
        (project / "config").mkdir()
        (project / "pipeline" / "dlc_sbatch_superanimal.sh").write_text("#!/bin/bash")
        (project / "config" / "setup.json").write_text("{}")
        sandbox = self.root / "dlc.sif"
        sandbox.write_bytes(b"container")
        self.setup["hosts"]["hpc"].update({
            "experiment_dir": str(hpc_root), "project_dir": str(project),
            "sandbox": str(sandbox),
        })
        manager = JobManager(hpc_root, self.setup, session)
        completed = SimpleNamespace(stdout="Submitted batch job 12345\n", stderr="")
        with patch("pipeline.job_manager._is_windows", return_value=False), \
             patch("pipeline.job_manager.subprocess.run", return_value=completed) as run:
            submitted = manager.submit()
            pending = manager.submit()
            retried = manager.submit(retry=True)
        self.assertEqual(submitted.job_id, "12345")
        self.assertEqual(pending.status, "pending")
        self.assertEqual(retried.status, "submitted")
        self.assertEqual(run.call_count, 2)
        self.assertIn("day1", submitted.command)
        self.assertTrue(manager.state_path.is_file())

    def test_incompatible_keypoints_skip_trial_without_staging_partial_pose(self):
        session = self.root / "day1"
        raw = session / "CAGE1" / "videos-raw"
        tracks = session / "CAGE1" / "tracks"
        raw.mkdir(parents=True)
        tracks.mkdir()
        for index, token in enumerate(self.setup["cages"]["CAGE1"]):
            video = raw / f"20260811_102757_{token}.mkv"
            video.write_bytes(b"video")
            bodypart = "nose" if index < 4 else "unrelated"
            columns = pd.MultiIndex.from_product(
                [["model"], ["single"], [bodypart], ["x", "y", "likelihood"]]
            )
            pd.DataFrame([[1.0, 2.0, 0.9]], columns=columns).to_hdf(
                tracks / f"{video.stem}DLC_model.h5", key="df"
            )
            (tracks / f"{video.stem}_after_adapt.json").write_text("{}")
        result = convert_session(
            session, self.setup["cages"], "mkv", self.setup["anipose"]["cam_regex"]
        )
        self.assertFalse(result.ready_trials)
        self.assertIn("No common", result.skipped_trials["CAGE1/20260811_102757_"])
        self.assertFalse((session / "CAGE1" / "pose-2d").exists())

    def test_incomplete_tracks_name_missing_camera_artifacts(self):
        session = self.root / "day1"
        raw = session / "CAGE1" / "videos-raw"
        tracks = session / "CAGE1" / "tracks"
        raw.mkdir(parents=True)
        tracks.mkdir()
        cameras = list(self.setup["cages"]["CAGE1"])
        for camera in cameras:
            (raw / f"20260811_102757_{camera}.mkv").write_bytes(b"video")
        (tracks / f"20260811_102757_{cameras[0]}_model.h5").write_bytes(b"track")
        (tracks / f"20260811_102757_{cameras[1]}_after_adapt.json").write_text("{}")

        result = convert_session(
            session, self.setup["cages"], "mkv", self.setup["anipose"]["cam_regex"]
        )

        reason = result.skipped_trials["CAGE1/20260811_102757_"]
        self.assertIn(f"{cameras[0]} (adaptation JSON)", reason)
        self.assertIn(f"{cameras[1]} (HDF5)", reason)
        self.assertIn(f"{cameras[2]} (HDF5, adaptation JSON)", reason)
        self.assertFalse(result.ready_trials)

    def test_missing_anipose_tool_reports_preflight_error(self):
        self.setup["hosts"]["local"]["anipose_command"] = [str(self.root / "missing_anipose.exe")]
        processor = PoseProcessor(self.root, self.setup, environment="local")
        with self.assertRaisesRegex(FileNotFoundError, "Anipose executable is missing"):
            processor._run_anipose("triangulate")


if __name__ == "__main__":
    unittest.main()
