# MLB2 consolidated pipeline

This package prepares multi-camera sessions, calibrates cameras with Anipose,
tracks 2D keypoints with the existing SuperAnimal camera models, and triangulates
complete five-camera trials. It accepts a zip archive, a raw session folder, or
an already organized session. Re-running a command skips verified outputs and
reports the next pending stage.
Input folders outside the selected experiment root are moved into that root;
zip archives are extracted there.

`config/setup.json` is the only file to edit for experiment, camera, model,
calibration, and cluster settings. It currently describes the local MLB2
experiment and the `/scratch/lbshks/mlb2/experiment` cluster copy. When a command
needs them, the pipeline generates `config.toml`, `cage_map.json`, and `jobs.json`
in the experiment root. The generated `config.toml` replaces an older one; make
changes in `setup.json` so they survive the next run.

## Run one session

From this repository, use Python 3.10 or newer:

```powershell
python -m pipeline.main run "C:\Users\Gerardo\Documents\Lockbox\mlb2\experiment\11_08_2026_Day7_Round2_Mouse_Lockbox2_Male_Cohort1_Pilot_10x10Task"
```

The default run previews a Slurm command for missing 2D tracks. It performs
preparation and any feasible camera calibration first, evaluates reprojection
error if manual label CSVs are configured, and converts and triangulates any
trials whose tracks are already complete. After the Slurm job finishes, invoke
the same command again to advance the remaining trials.

On the HPC login node, place `pipeline/` and `config/` directly under
`/scratch/lbshks/super_animal`, then run:

```bash
python3 -m pipeline.main run /scratch/lbshks/mlb2/experiment/SESSION_NAME --environment hpc --submit
```

`--submit` is required for Slurm submission. Jobs are scoped to the selected
session; the returned job ID is saved in that session's `pipeline_state.json`.
Repeated runs will show the saved ID instead of submitting a duplicate. If a
job has ended without producing tracks, `pose SESSION_NAME --submit --retry-job`
submits a replacement. Data transfer between local and HPC experiment roots is
handled outside this package.

The equivalent Python API is:

```python
from pipeline import run_pose_detection_pipeline

result = run_pose_detection_pipeline(
    source_path="session.zip",
    experiment_dir=None,  # use the selected host path in setup.json
    setup_json="config/setup.json",
    submit_jobs=False,
)
print(result.status, result.pending_videos, result.triangulation.outputs)
```

## Run individual stages

Every command accepts `--config`, `--environment`, and `--experiment-dir`:

```text
python -m pipeline.main prepare ZIP_OR_FOLDER
python -m pipeline.main downsample SESSION_DIR
python -m pipeline.main calibrate SESSION_DIR
python -m pipeline.main labels SESSION_DIR --frame-index 100
python -m pipeline.main evaluate SESSION_DIR --labels LABEL_CSV_DIR
python -m pipeline.main pose SESSION_DIR            # preview
python -m pipeline.main pose SESSION_DIR --submit   # HPC only
python -m pipeline.main convert SESSION_DIR
python -m pipeline.main triangulate SESSION_DIR
```

`labels` writes frames to the configured DLC project for manual labeling.
Reprojection evaluation is optional until `evaluation.label_csv_root` is set or
`--labels` is supplied. An error above `evaluation.threshold_px` warns and still
allows tracking. Trial conversion requires five synchronized cameras, finished
HDF5 tracks and adaptation reports, matching frame counts, and shared
`x`/`y`/`likelihood` bodyparts. Incomplete trials are listed with a reason.

For each cage, the stage folders are:

```text
SESSION/CAGE1/
  calibration/   # low-FPS videos and calibration.toml
  videos-raw/    # source experiment videos
  tracks/        # SuperAnimal outputs
  pose-2d/       # normalized Anipose HDF5 files
  pose-3d/       # Anipose trial CSV files
```

Camera selection and calibration-video downsampling preserve the prior
mechanical- and sliding-lockbox behavior. For the current MLB2 setup, the raw
video extension is `.mkv`, the calibration extension is `.avi`, and `CAGE1` and
`CAGE2` each map five cameras to top or lateral SuperAnimal models.

## Environments and verification

The local setup invokes Anipose through `conda run -n anipose`, which supplies
the environment's required DLL paths. The Python environment running this
package needs the dependencies in `pyproject.toml`; reprojection evaluation also
needs the `validation` extra. The Slurm worker runs DeepLabCut inside the
configured Singularity sandbox. On HPC, make the worker script readable by
Slurm and ensure the configured module, sandbox, cache, and project paths exist.

```text
python -m unittest discover -s tests -v
python tests/smoke_calibration.py
python tests/smoke_anipose.py
```

The calibration smoke command copies five existing MLB2 calibration videos and
runs real Anipose calibration in a temporary project. The triangulation smoke
command copies one existing MLB2 calibration into a temporary project,
generates synthetic five-camera 2D poses, runs real Anipose triangulation, and
removes the temporary project. GPU SuperAnimal inference requires the HPC DLC
container and was not run locally.
