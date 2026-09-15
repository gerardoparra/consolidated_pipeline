"""Consolidated MLB2 calibration and pose pipeline."""

__all__ = ["PipelineRunResult", "run_pose_detection_pipeline"]


def __getattr__(name: str):
    if name in __all__:
        from . import main
        return getattr(main, name)
    raise AttributeError(name)
