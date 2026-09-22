import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.label_3d_preview import (
    patch_legacy_skvideo_writer,
    sampled_pose_csv,
)


class Label3dPreviewTests(unittest.TestCase):
    def test_sampling_reduces_frames_and_preserves_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pose.csv"
            destination = root / "sampled.csv"
            pd.DataFrame({
                "fnum": range(60),
                "nose_x": range(60),
                "nose_y": range(60),
                "nose_z": range(60),
            }).to_csv(source, index=False)

            stride, output_fps = sampled_pose_csv(
                source, target_fps=10.0, source_fps=30.0,
                destination=destination,
            )

            sampled = pd.read_csv(destination)
            self.assertEqual(stride, 3)
            self.assertEqual(output_fps, 10.0)
            self.assertEqual(len(sampled), 20)
            self.assertEqual(sampled["nose_x"].tolist(), list(range(0, 60, 3)))
            self.assertEqual(sampled["fnum"].tolist(), list(range(20)))
            self.assertEqual(len(sampled) / output_fps, 60 / 30.0)

    def test_target_above_source_does_not_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pose.csv"
            destination = root / "sampled.csv"
            pd.DataFrame({"fnum": range(3), "nose_x": range(3)}).to_csv(
                source, index=False
            )

            stride, output_fps = sampled_pose_csv(
                source, target_fps=60.0, source_fps=30.0,
                destination=destination,
            )

            self.assertEqual(stride, 1)
            self.assertEqual(output_fps, 30.0)
            self.assertEqual(len(pd.read_csv(destination)), 3)

    def test_target_is_a_ceiling_for_nondivisible_source_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pose.csv"
            destination = root / "sampled.csv"
            pd.DataFrame({"fnum": range(25), "nose_x": range(25)}).to_csv(
                source, index=False
            )

            stride, output_fps = sampled_pose_csv(
                source, target_fps=10.0, source_fps=25.0,
                destination=destination,
            )

            self.assertEqual(stride, 3)
            self.assertAlmostEqual(output_fps, 25 / 3)
            self.assertLessEqual(output_fps, 10.0)

    def test_legacy_skvideo_writer_uses_numpy_compatibility_view(self):
        class LegacyWriter:
            def writeFrame(self, image):
                self.received_compatibility_view = (
                    type(image).__name__ == "_ArrayWithTostring"
                )
                self.payload = image.clip(0, 255).astype("uint8").tostring()

        writer = LegacyWriter()
        patched = patch_legacy_skvideo_writer(LegacyWriter, _force=True)
        writer.writeFrame(np.zeros((2, 2, 3), dtype="uint8"))
        self.assertEqual(len(writer.payload), 12)
        self.assertTrue(writer.received_compatibility_view)
        self.assertTrue(patched)


if __name__ == "__main__":
    unittest.main()
