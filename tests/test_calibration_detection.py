import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from pipeline.calibration_prep import calibration_target_detection_rate


class CalibrationDetectionTests(unittest.TestCase):
    def detection_rate(self, frames):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = len(frames)
        capture.read.side_effect = [(True, frame) for frame in frames]
        with patch.object(cv2, "VideoCapture", return_value=capture):
            rate = calibration_target_detection_rate(
                Path("calibration.avi"), (5, 4), sample_count=len(frames)
            )
        capture.release.assert_called_once()
        return rate

    @unittest.skipUnless(
        hasattr(cv2.aruco, "ArucoDetector"), "Requires the modern OpenCV API"
    )
    def test_modern_api_without_module_detect_markers(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        markers = np.full((360, 360), 255, dtype=np.uint8)
        for marker_id, (y, x) in enumerate(
            [(30, 30), (30, 210), (210, 30), (210, 210)]
        ):
            markers[y:y + 100, x:x + 100] = cv2.aruco.generateImageMarker(
                dictionary, marker_id, 100
            )
        marker_frame = cv2.cvtColor(markers, cv2.COLOR_GRAY2BGR)
        blank_frame = np.full_like(marker_frame, 255)
        # Expose only the modern API to reproduce the failing HPC environment.
        modern_aruco = SimpleNamespace(
            DICT_4X4_50=cv2.aruco.DICT_4X4_50,
            getPredefinedDictionary=cv2.aruco.getPredefinedDictionary,
            ArucoDetector=cv2.aruco.ArucoDetector,
        )
        with patch.object(cv2, "aruco", modern_aruco):
            self.assertEqual(self.detection_rate([marker_frame, blank_frame]), 0.5)

    def test_legacy_api_and_dictionary_factory(self):
        dictionary = object()
        for factory in ("getPredefinedDictionary", "Dictionary_get"):
            with self.subTest(factory=factory):
                legacy_aruco = SimpleNamespace(
                    DICT_4X4_50=0,
                    detectMarkers=Mock(return_value=([], np.arange(4), [])),
                    **{factory: Mock(return_value=dictionary)},
                )
                frame = np.full((100, 100, 3), 255, dtype=np.uint8)
                with patch.object(cv2, "aruco", legacy_aruco):
                    self.assertEqual(self.detection_rate([frame]), 1.0)
                legacy_aruco.detectMarkers.assert_called_once()
                self.assertIs(legacy_aruco.detectMarkers.call_args.args[1], dictionary)

    def test_checkerboard_fallback_with_insufficient_markers(self):
        board = np.full((280, 320), 255, dtype=np.uint8)
        for y in range(5):
            for x in range(6):
                if (x + y) % 2 == 0:
                    board[40 + y * 40:80 + y * 40, 40 + x * 40:80 + x * 40] = 0
        frame = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        for marker_ids in (None, np.arange(3)):
            with self.subTest(marker_ids=marker_ids):
                detector = Mock()
                detector.detectMarkers.return_value = ([], marker_ids, [])
                modern_aruco = SimpleNamespace(
                    DICT_4X4_50=0,
                    getPredefinedDictionary=Mock(),
                    ArucoDetector=Mock(return_value=detector),
                )
                with patch.object(cv2, "aruco", modern_aruco):
                    self.assertEqual(self.detection_rate([frame]), 1.0)


if __name__ == "__main__":
    unittest.main()
