import json
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pipeline.config import Setup, hpc_repository_paths
from pipeline.file_handler import FileHandler
from pipeline.reprojection import _extract_leading_yymmdd, _extract_serial_token
from pipeline.video_processor import VideoProcessor


SETUP_PATH = Path(__file__).resolve().parent.parent / "config" / "setup.json"


class ConfigAndFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.setup = Setup.load(SETUP_PATH)

    def test_setup_generates_tool_configs_and_replaces_stale_anipose_config(self):
        experiment = self.root / "experiment"
        experiment.mkdir()
        (experiment / "config.toml").write_text("old Anipose config", encoding="utf-8")
        derived = self.setup.materialize(experiment, "local")
        with derived["anipose"].open("rb") as handle:
            config = tomllib.load(handle)
        self.assertEqual(config["nesting"], 2)
        self.assertEqual(config["calibration"]["board_size"], [10, 7])
        self.assertEqual(config["triangulation"]["cam_regex"], "camera([0-9][0-9])")
        self.assertEqual(json.loads(derived["cages"].read_text())["CAGE1"]["camera06"], "top")
        jobs = json.loads(derived["jobs"].read_text())
        self.assertIn("mlb2_experiment", jobs)
        self.assertEqual(
            jobs["mlb2_experiment"]["wrapper"],
            str(hpc_repository_paths(self.setup.data["hosts"]["hpc"])[2]),
        )
        before = derived["anipose"].stat().st_mtime_ns
        self.setup.materialize(experiment, "local")
        self.assertEqual(before, derived["anipose"].stat().st_mtime_ns)

    def test_zip_and_prepared_folder_inputs(self):
        experiment = self.root / "experiment"
        experiment.mkdir()
        handler = FileHandler(experiment, self.setup.data)
        archive = self.root / "day1.zip"
        with zipfile.ZipFile(archive, "w") as zip_file:
            zip_file.writestr("recording/CAGE1/example.txt", "data")
        imported = handler.import_raw_data(archive)
        self.assertEqual(imported, experiment / "day1")
        self.assertTrue((imported / "CAGE1" / "example.txt").is_file())

        prepared = self.root / "day2"
        (prepared / "CAGE1" / "calibration").mkdir(parents=True)
        (prepared / "CAGE1" / "videos-raw").mkdir()
        moved = handler.import_raw_data(prepared)
        self.assertEqual(moved, experiment / "day2")
        self.assertTrue(handler.is_prepared(moved))
        self.assertFalse(prepared.exists())
        self.assertEqual(handler.import_raw_data(moved), moved)

    def test_video_preparation_preserves_current_calibration_parameters(self):
        experiment = self.root / "experiment"
        session = experiment / "day1"
        (session / "CAGE1").mkdir(parents=True)
        processor = VideoProcessor(experiment, self.setup.data)
        with patch("pipeline.video_processor.restructure_and_downsample") as prepare:
            processor.prepare(session)
        self.assertEqual(prepare.call_args.kwargs["experiment"], "sliding_lockbox")
        self.assertEqual(prepare.call_args.kwargs["out_fps"], 3.0)
        self.assertEqual(prepare.call_args.kwargs["board_size"], (10, 7))
        (session / "CAGE1" / "calibration").mkdir()
        (session / "CAGE1" / "videos-raw").mkdir()
        (session / "CAGE1" / "calibration" / "calibration.toml").write_text("done")
        with patch("pipeline.video_processor.restructure_and_downsample") as prepare:
            processor.prepare(session)
        prepare.assert_not_called()
        self.assertEqual(processor.downsample(session), [])

    def test_reprojection_matches_current_mlb2_dates_and_camera_names(self):
        self.assertEqual(_extract_leading_yymmdd("11_08_2026_Day7", "session"), "260811")
        self.assertEqual(_extract_leading_yymmdd("20260811_labels", "labels"), "260811")
        self.assertEqual(_extract_serial_token("20260811_124044_camera03_2fps.avi"), "03")
        self.assertEqual(_extract_serial_token("camera03_img00100.jpg"), "03")


if __name__ == "__main__":
    unittest.main()
