import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.config import Setup
from pipeline.pose_inference import group_videos
from pipeline.pose_processor import PoseProcessor


SETUP_PATH = Path(__file__).resolve().parent.parent / "config" / "setup.json"


class PoseWorkerTests(unittest.TestCase):
    def test_cameras_use_setup_models_and_sibling_track_folders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "day1" / "CAGE1" / "videos-raw"
            raw.mkdir(parents=True)
            videos = []
            for camera in ("camera06", "camera07"):
                path = raw / f"20260811_102757_{camera}.mkv"
                path.write_bytes(b"video")
                videos.append(path)
            setup = Setup.load(SETUP_PATH).data
            groups, errors = group_videos(videos, root, root, setup["cages"])
            self.assertFalse(errors)
            self.assertEqual({perspective for perspective, _ in groups}, {"top", "front"})
            self.assertEqual({destination for _, destination in groups}, {raw.parent / "tracks"})

            processor = PoseProcessor(root, setup)
            with patch("pipeline.pose_processor.run_inference") as inference:
                processor.detect_keypoints(root, root)
            self.assertEqual(inference.call_args.kwargs["model_configs"]["top"], "superanimal_topviewmouse")
            self.assertEqual(inference.call_args.kwargs["model_configs"]["front"], "superanimal_quadruped")
            self.assertEqual(inference.call_args.kwargs["output_layout"], "sibling-tracks")


if __name__ == "__main__":
    unittest.main()
