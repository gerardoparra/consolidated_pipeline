import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.label_3d_preview import sampled_pose_csv


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


if __name__ == "__main__":
    unittest.main()
