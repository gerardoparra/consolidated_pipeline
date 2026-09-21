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

On HPC, place this repository at `/scratch/lbshks/super_animal`, activate the
host environment described below, and run from a CPU allocation:

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

## Install dependencies on HPC

There are two Python environments. Preparation, calibration, conversion, and
triangulation use a **host Python environment** with this package and Anipose.
The GPU Slurm worker uses Python **inside the Singularity sandbox**; installing
DeepLabCut only in the host environment will not make it available to the worker.
The current `run` command performs preparation and calibration in the shell
that invokes it, so use a CPU allocation for a full session rather than doing
that work on the login node. The GPU worker is submitted separately by `--submit`.

First, copy the whole repository to the `hosts.hpc.project_dir` in
`config/setup.json` (currently `/scratch/lbshks/super_animal`). Use a cluster
Python module or installation with Python 3.10 or newer and `venv`, then create
the host environment in a persistent location:

```bash
cd /scratch/lbshks/super_animal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . anipose
# Only if using the optional manual-label reprojection check:
python -m pip install -e '.[validation]'
```

Activate `.venv` in every shell or CPU job that runs the CLI. The configured
HPC `anipose_command` is `anipose`, so that executable must be on `PATH` in
the activated environment. Anipose's [installation guide](https://anipose.readthedocs.io/en/latest/installation.html)
also uses `pip install anipose`. Check imports and the CLI before processing a
session; this catches broken NumPy installations and missing ChArUco support:

```bash
python -c "import numpy, pandas, tables, cv2; assert hasattr(cv2, 'aruco')"
anipose --help
python -m pip check
```

The worker script loads the configured `singularity` module and executes
`hosts.hpc.sandbox` (currently `deeplabcut_sandbox`) with `--nv`. That sandbox
must have Python 3.10+, a GPU-compatible PyTorch installation, DeepLabCut with
SuperAnimal/Model Zoo support, and the package runtime dependencies (`numpy`,
`pandas`, `tables`, and `opencv-contrib-python`). When building or updating a
**writable** sandbox, install those Python packages inside it, using the
container's Python and a PyTorch build compatible with the cluster driver:

```bash
python3 -m pip install 'deeplabcut[gui,modelzoo]' numpy pandas tables opencv-contrib-python
```

Run that installation command in the sandbox build environment, not in the
host `.venv`. If the existing sandbox is already provisioned, verify it instead
of reinstalling. See the [DeepLabCut installation guide](https://deeplabcut.github.io/DeepLabCut/docs/installation.html)
for its current GPU/PyTorch instructions and the
[Apptainer GPU guide](https://apptainer.org/docs/user/main/gpu.html) for `--nv`.
On a GPU allocation, check the exact import the worker uses:

```bash
cd /scratch/lbshks/super_animal
module load singularity
singularity exec --nv --containall --bind "$PWD:$PWD:ro" \
  /scratch/lbshks/super_animal/deeplabcut_sandbox \
  env PYTHONPATH="$PWD" python3 -c \
  'import cv2, numpy, pandas, tables, torch; from deeplabcut.modelzoo.video_inference import video_inference_superanimal; assert torch.cuda.is_available()'
```

Finally, confirm that `sbatch`, the configured Singularity module and sandbox,
the model directory, and writable experiment/cache/temp paths exist. The
worker writes its logs under the project directory. SuperAnimal model weights
may download on first use, so the configured cache must also be usable from a
compute node. Transfer session data to HPC storage separately; the pipeline
does not copy it between hosts.

## Verification

The local setup invokes Anipose through `conda run -n anipose`, which supplies
the environment's required DLL paths. On HPC, the commands above verify the
separate host and worker environments before running a session.

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
