import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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
            self.assertFalse(inference.call_args.kwargs["create_labeled_video"])
            self.assertTrue(inference.call_args.kwargs["delete_labeled_videos_after_inference"])

    def test_labeled_video_setting_reaches_superanimal_for_both_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "day1" / "CAGE1" / "videos-raw"
            raw.mkdir(parents=True)
            (raw / "20260811_102757_camera06.mkv").write_bytes(b"video")
            setup = Setup.load(SETUP_PATH).data
            inference_module = types.ModuleType("deeplabcut.modelzoo.video_inference")
            inference_module.video_inference_superanimal = Mock()
            fake_modules = {
                "deeplabcut": types.ModuleType("deeplabcut"),
                "deeplabcut.modelzoo": types.ModuleType("deeplabcut.modelzoo"),
                "deeplabcut.modelzoo.video_inference": inference_module,
            }
            with patch.dict(sys.modules, fake_modules):
                for enabled in (False, True):
                    with self.subTest(create_labeled_video=enabled):
                        setup["pose"]["create_labeled_video"] = enabled
                        inference_module.video_inference_superanimal.reset_mock()
                        PoseProcessor(root, setup).detect_keypoints(root, root)
                        self.assertEqual(
                            inference_module.video_inference_superanimal.call_args.kwargs[
                                "create_labeled_video"
                            ],
                            enabled,
                        )

    def test_cleanup_only_removes_labeled_previews_for_complete_tracks(self):
        for delete_previews in (False, True):
            with self.subTest(delete_labeled_videos=delete_previews):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    raw = root / "day1" / "CAGE1" / "videos-raw"
                    raw.mkdir(parents=True)
                    video = raw / "20260811_102757_camera06.mkv"
                    video.write_bytes(b"raw video")
                    tracks = raw.parent / "tracks"
                    stem = video.stem
                    labeled = tracks / f"{stem}_superanimal_topviewmouse_labeled_after_adapt.mp4"
                    before = tracks / f"{stem}_superanimal_topviewmouse_labeled_before_adapt.mp4"
                    unrelated = tracks / f"{stem}_other.mp4"
                    other_camera = tracks / "20260811_102757_camera07_superanimal_labeled.mp4"

                    def fake_superanimal(videos, _model, **kwargs):
                        self.assertEqual(videos, [str(video)])
                        (tracks / f"{stem}DLC_model.h5").write_bytes(b"track")
                        (tracks / f"{stem}DLC_model_after_adapt.json").write_text("{}")
                        for path in (labeled, before, unrelated, other_camera):
                            path.write_bytes(b"preview")

                    setup = Setup.load(SETUP_PATH).data
                    setup["pose"]["delete_labeled_videos_after_inference"] = delete_previews
                    inference_module = types.ModuleType("deeplabcut.modelzoo.video_inference")
                    inference_module.video_inference_superanimal = fake_superanimal
                    fake_modules = {
                        "deeplabcut": types.ModuleType("deeplabcut"),
                        "deeplabcut.modelzoo": types.ModuleType("deeplabcut.modelzoo"),
                        "deeplabcut.modelzoo.video_inference": inference_module,
                    }
                    with patch.dict(sys.modules, fake_modules):
                        PoseProcessor(root, setup).detect_keypoints(root, root)
                    self.assertEqual(labeled.exists(), not delete_previews)
                    self.assertEqual(before.exists(), not delete_previews)
                    self.assertTrue(unrelated.exists())
                    self.assertTrue(other_camera.exists())
                    self.assertEqual(video.read_bytes(), b"raw video")
                    self.assertTrue((tracks / f"{stem}DLC_model.h5").exists())
                    self.assertTrue((tracks / f"{stem}DLC_model_after_adapt.json").exists())

    def test_cleanup_skips_incomplete_tracks_and_handles_older_dlc_signature(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "day1" / "CAGE1" / "videos-raw"
            raw.mkdir(parents=True)
            video = raw / "20260811_102757_camera06.mkv"
            video.write_bytes(b"raw video")
            complete_video = raw / "20260811_102757_camera07.mkv"
            complete_video.write_bytes(b"raw video")
            labeled = raw.parent / "tracks" / f"{video.stem}_superanimal_labeled.mp4"
            complete_labeled = raw.parent / "tracks" / f"{complete_video.stem}_superanimal_labeled.mp4"

            def legacy_superanimal(videos, _model, *, model_name, detector_name,
                                   videotype, video_adapt, batch_size,
                                   detector_batch_size, video_adapt_batch_size,
                                   scale_list, dest_folder, max_individuals):
                if videos == [str(complete_video)]:
                    (complete_labeled.parent / f"{complete_video.stem}DLC_model.h5").write_bytes(b"track")
                    (complete_labeled.parent / f"{complete_video.stem}DLC_model_after_adapt.json").write_text("{}")
                    complete_labeled.write_bytes(b"preview")
                else:
                    labeled.write_bytes(b"preview")

            inference_module = types.ModuleType("deeplabcut.modelzoo.video_inference")
            inference_module.video_inference_superanimal = legacy_superanimal
            fake_modules = {
                "deeplabcut": types.ModuleType("deeplabcut"),
                "deeplabcut.modelzoo": types.ModuleType("deeplabcut.modelzoo"),
                "deeplabcut.modelzoo.video_inference": inference_module,
            }
            with patch.dict(sys.modules, fake_modules):
                PoseProcessor(root, Setup.load(SETUP_PATH).data).detect_keypoints(root, root)
            self.assertTrue(labeled.exists())
            self.assertFalse(complete_labeled.exists())

    def test_cleanup_removes_previews_from_tracks_already_complete_on_worker_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "day1" / "CAGE1" / "videos-raw"
            raw.mkdir(parents=True)
            video = raw / "20260811_102757_camera06.mkv"
            video.write_bytes(b"raw video")
            tracks = raw.parent / "tracks"
            tracks.mkdir()
            (tracks / f"{video.stem}DLC_model.h5").write_bytes(b"track")
            (tracks / f"{video.stem}DLC_model_after_adapt.json").write_text("{}")
            labeled = tracks / f"{video.stem}_superanimal_labeled.mp4"
            labeled.write_bytes(b"preview")
            inference_module = types.ModuleType("deeplabcut.modelzoo.video_inference")
            inference_module.video_inference_superanimal = Mock()
            fake_modules = {
                "deeplabcut": types.ModuleType("deeplabcut"),
                "deeplabcut.modelzoo": types.ModuleType("deeplabcut.modelzoo"),
                "deeplabcut.modelzoo.video_inference": inference_module,
            }
            with patch.dict(sys.modules, fake_modules):
                PoseProcessor(root, Setup.load(SETUP_PATH).data).detect_keypoints(root, root)
            inference_module.video_inference_superanimal.assert_not_called()
            self.assertFalse(labeled.exists())


if __name__ == "__main__":
    unittest.main()
