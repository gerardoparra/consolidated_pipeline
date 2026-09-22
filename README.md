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

## Pipeline stages

The `run` command advances through the stages below in order. It resumes from
completed outputs, previews missing GPU work unless `--submit` is supplied,
and can be run again after tracking finishes.

| Stage / CLI command | What it does | Main output |
| --- | --- | --- |
| Import and prepare / `prepare` | Extract a session zip, organize a raw folder, or accept a prepared session; sort calibration and behavioral videos into each cage. | `SESSION/CAGE*/calibration/` and `videos-raw/` |
| Downsample / `downsample` | Sample each calibration video at a synchronized low frame rate for Anipose. The MLB2 default is 3 FPS. | Calibration `.avi` files |
| Camera calibration / `calibrate` | Use the ChArUco-board videos to estimate the five cameras' geometry for each cage. | `calibration/calibration.toml` |
| 2D tracking / `pose` | Run SuperAnimal on the raw behavioral videos with the configured top-view or quadruped model. On HPC, preview or submit a GPU Slurm job. | Sibling `tracks/` HDF5 files and adaptation reports |
| Convert tracks / `convert` | Group synchronized five-camera trials, verify matching frames and shared keypoints, and write Anipose-compatible 2D tracks. | `pose-2d/` HDF5 files |
| 3D pose / `triangulate` | Use Anipose and the saved camera calibration to triangulate complete 2D trials. | `pose-3d/` trial CSV files |

The `prepare` command also performs downsampling; the separate `downsample`
command resumes that step for an organized session. The optional manual-label
calibration check is described under **Additional commands**.

In `config/setup.json`, `pose.create_labeled_video` controls whether SuperAnimal
also renders labeled preview videos. The MLB2 default is `false` to save space;
2D HDF5 tracks and adaptation JSON reports are still produced. Set it to `true`
to restore labeled videos. `pose.delete_labeled_videos_after_inference` defaults
to `true` as a second space-saving measure. After inference, the worker removes
matching labeled `.mp4` previews from `tracks/` only when that raw video's HDF5
track and `*_after_adapt.json` report are complete. It also cleans previews for
tracks already complete when the worker is rerun. It never deletes raw videos,
HDF5 tracks, or JSON reports. Set the cleanup option to `false` to keep previews.
If the installed DeepLabCut does not accept `create_labeled_video`, the worker
uses its default video behavior and the cleanup setting still applies. Neither
setting removes previews from sessions that are not rerun through the worker.

## Run one session

From this repository, use Python 3.10 or newer:

```powershell
python -m pipeline.main run "C:\Users\Gerardo\Documents\Lockbox\mlb2\experiment\11_08_2026_Day7_Round2_Mouse_Lockbox2_Male_Cohort1_Pilot_10x10Task"
```

The default run previews a Slurm command for missing 2D tracks. It performs
preparation and any feasible camera calibration first, then converts and
triangulates any trials whose tracks are already complete. It does not perform
the manual-label calibration check. After the Slurm job finishes, invoke the
same command again to advance the remaining trials.

On HPC, clone this repository inside `/scratch/lbshks/super_animal` so that
`pyproject.toml` is at the configured `hosts.hpc.repository_dir` (currently
`/scratch/lbshks/super_animal/consolidated_pipeline`). Activate the host
environment described below before running from a CPU allocation:

```bash
cd /scratch/lbshks/super_animal/consolidated_pipeline
source /scratch/lbshks/super_animal/.venv/bin/activate
python -m pipeline.main run /scratch/lbshks/mlb2/experiment/SESSION_NAME --environment hpc --submit
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

## Additional commands

The full `run` command handles the core stages. Use these commands to run or
resume one stage independently. On HPC, activate the host environment first:

```bash
cd /scratch/lbshks/super_animal/consolidated_pipeline
source /scratch/lbshks/super_animal/.venv/bin/activate
```

Every command accepts `--config`, `--environment`, and `--experiment-dir`:

```text
python -m pipeline.main prepare ZIP_OR_FOLDER
python -m pipeline.main downsample SESSION_DIR
python -m pipeline.main calibrate SESSION_DIR
python -m pipeline.main pose SESSION_DIR            # preview
python -m pipeline.main pose SESSION_DIR --submit   # HPC only
python -m pipeline.main convert SESSION_DIR
python -m pipeline.main triangulate SESSION_DIR
```

The manual calibration check is separate from the default run. After
calibration, extract frames for labeling and evaluate the resulting CSVs when
you want to check reprojection error:

```text
python -m pipeline.main labels SESSION_DIR --frame-index 100
python -m pipeline.main evaluate SESSION_DIR --labels LABEL_CSV_DIR
```

`labels` writes frames to the configured DLC project for manual labeling.
`evaluate` runs only when explicitly requested. It uses
`evaluation.label_csv_root` or its `--labels` argument to locate manual labels;
an error above `evaluation.threshold_px` reports a warning. This check
triangulates manually labeled points only to test camera geometry; the core 3D
stage triangulates SuperAnimal tracks from behavioral videos. Trial conversion
requires five synchronized cameras, finished HDF5 tracks and adaptation
reports, matching frame counts, and shared
`x`/`y`/`likelihood` bodyparts. Incomplete trials are listed with a reason.
If `convert` reports that 2D tracks are incomplete, check the sibling `tracks/`
directory: every raw camera video in that trial needs a nonempty HDF5 track and
a matching nonempty `*_after_adapt.json` report. A labeled preview video or
`*_before_adapt` output alone is not a completed track. Check the saved job ID
in `pipeline_state.json` and the Slurm logs to see whether inference is still
running or failed, then rerun `convert` after the missing outputs are ready.

With video adaptation enabled, two HDF5 files per raw video are expected. The
shorter pretrained-model filename contains the initial predictions used to make
pseudo-labels; the filename containing `snapshot-...` contains predictions from
the adapted detector and pose checkpoints. DeepLabCut adds `_after_adapt` to the
adapted JSON report but not to its HDF5 filename. Conversion pairs that report
with the same filename prefix and uses the corresponding adapted HDF5 file.

If triangulation finds ready `pose-2d` trials but produces no 3D CSV, the CLI
prints Anipose's captured output after the skipped trials. Anipose may catch a
per-trial `ValueError`, print its traceback, and still exit successfully, so an
exit code alone does not prove that it wrote `pose-3d` output.

If that output ends with `ValueError: Need at least one array to stack` from
`aniposelib/cameras.py`, the converted HDF5 structure and calibration are not
necessarily at fault. With `anipose.ransac` enabled, the JAX-based Aniposelib
implementation can construct an empty camera subset when a frame/bodypart has
no camera above `anipose.score_threshold`. One such point can abort the entire
trial, even when the trial has good multi-camera coverage overall. The progress
total is frames multiplied by bodyparts, so failure at `0%` indicates that the
first flattened point triggered this case.

The recommended workaround is to set `"ransac": false` under `anipose` in
`config/setup.json`, then rerun the pipeline's `triangulate` command. The command
regenerates `config.toml`; do not make the change only in that generated file.
Without RANSAC, Anipose triangulates points visible in at least two cameras and
leaves points without enough camera observations as `NaN` in the 3D CSV. Running
`anipose analyze` or `anipose filter` is not required for this error. Lowering
the score threshold is not a safe general workaround because it can admit the
`-1` coordinates used for missing detections.

```bash
# After changing config/setup.json:
python -m pipeline.main triangulate /scratch/lbshks/mlb2/experiment/SESSION_NAME
```

### Optional 3D visualization

The consolidated pipeline stops after writing `pose-3d` CSV files. Anipose can
render those coordinates as standalone skeleton animations with `label-3d`.
This command is optional, is not part of the default `run`, and requires
Anipose's Mayavi visualization dependencies described under **Install
dependencies on HPC**. The configured labeling scheme connects
`left_eye`–`nose`–`right_eye` and `nose`–`tail_base`–`tail_end`, matching the
current SuperAnimal output.

Run the command from the experiment root so Anipose finds the generated
`config.toml`. It writes animations under each cage's `videos-3d/` directory:

```bash
cd /scratch/lbshks/mlb2/experiment
anipose label-3d
```

On a headless compute node, use a virtual framebuffer if `xvfb-run` is
available:

```bash
cd /scratch/lbshks/mlb2/experiment
xvfb-run -a --server-args="-screen 0 1280x1024x24" anipose label-3d
```

`anipose label-combined` additionally requires retained 2D labeled videos, so
it is not compatible with the default space-saving settings that omit or delete
those previews.

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

First, clone the whole repository at the `hosts.hpc.repository_dir` in
`config/setup.json` (currently
`/scratch/lbshks/super_animal/consolidated_pipeline`). This must include
`pyproject.toml` at the repository root, alongside `pipeline/` and `config/`.
If the clone has a different name or location, edit `hosts.hpc.repository_dir`
before submitting a Slurm job. The job manager derives the wrapper script,
worker setup file, working directory, and log paths from that setting; the
sandbox, cache, and temp paths remain separate settings. Earlier deployments
that copied only `pipeline/` and `config/` need the manifest too, or
`pip install -e .` reports that this is not a Python project. Check the files
before installing:

```bash
cd /scratch/lbshks/super_animal/consolidated_pipeline
ls -l pyproject.toml pipeline/main.py config/setup.json
```

If `pyproject.toml` is missing, transfer it from this repository and rerun the
install command. Keep an existing `/scratch/lbshks/super_animal/.venv`; it does
not need to be recreated after cloning the repository. Use a cluster Python
module or installation with Python 3.10 or newer and `venv`. Create the host
environment alongside the checkout only if it does not already exist, then
activate it and install the package from the repository:

```bash
cd /scratch/lbshks/super_animal/consolidated_pipeline
python3 -m venv /scratch/lbshks/super_animal/.venv  # skip if it already exists
source /scratch/lbshks/super_animal/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . anipose
# Only if using the optional manual-label reprojection check:
python -m pip install -e '.[validation]'
# Only if using Anipose's optional label-3d visualization:
python -m pip install 'anipose[viz]'
```

The `viz` extra installs Mayavi and VTK, which are large optional dependencies.
Mayavi depends on VTK and a rendering backend; Linux compute nodes without a
display may also need the cluster's Xvfb package or module. Test the import
before starting a long render:

```bash
python -c 'from mayavi import mlab; print("Mayavi import OK")'
command -v xvfb-run
```

Mayavi is distributed from PyPI as source rather than a platform wheel. If it
does not build in the Python 3.13 host environment, create a separate Python
3.11 or 3.12 visualization environment rather than replacing the working pose
pipeline environment. The visualization environment only needs access to the
experiment's generated `config.toml` and `pose-3d` CSV files.

Activate the host `.venv` in every shell or CPU job that runs the CLI. The
configured HPC `anipose_command` is `anipose`, so it must be on `PATH` in
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
cd /scratch/lbshks/super_animal/consolidated_pipeline
module load singularity
singularity exec --nv --containall --bind "$PWD:$PWD:ro" \
  /scratch/lbshks/super_animal/deeplabcut_sandbox \
  env PYTHONPATH="$PWD" python3 -c \
  'import cv2, inspect, numpy, pandas, tables, torch; from deeplabcut.modelzoo.video_inference import video_inference_superanimal; assert torch.cuda.is_available(); print("create_labeled_video supported:", "create_labeled_video" in inspect.signature(video_inference_superanimal).parameters)'
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

## Working with limited HPC storage

ZIP import now extracts into a temporary directory beside the destination
session and removes that staging directory if extraction fails. Older versions
used the system temporary directory; a `No space left on device` error during
`extractall` may therefore mean `/tmp` filled rather than the experiment
scratch area. Before retrying a failed ZIP, check the available space and your
cluster quota, and inspect any old `rat_lockbox_unzip_*` directories left by a
failed run. A ZIP kept on scratch and its extracted session both consume space.

If scratch cannot hold the ZIP, extracted videos, tracking intermediates, and
outputs at once, prepare and downsample the session locally. Transfer one
prepared session at a time with `CAGE*/videos-raw/` and the low-FPS videos in
`CAGE*/calibration/`; `calibration/originals/` and the ZIP can stay on local
storage. Run `python -m pipeline.main prepare ZIP_PATH` locally, transfer the
resulting prepared session, then run
`python -m pipeline.main run SESSION_DIR --environment hpc --submit` from the
activated HPC environment. This runs
calibration and submits SuperAnimal tracking; rerun it after the GPU job finishes
to convert and triangulate ready trials. After verifying and backing up the
calibration and pose outputs, remove unneeded videos from scratch before
transferring the next session. The pipeline discovers trials from `videos-raw/`,
so deleting those files prevents a later `run` or `convert` from resuming that
session; finish and verify 3D output first.
