# MLB2 consolidated pipeline

Turn synchronized multi-camera recordings into calibrated 3D mouse poses, or develop and evaluate SuperAnimal models using labeled data.

## Overview

The project provides two workflows:

| Your goal | Workflow | Start here |
| --- | --- | --- |
| Process recorded experiments into 3D poses | Prepare data → calibrate cameras → track 2D keypoints → convert/filter → triangulate | [Process a session](#process-a-session) |
| Train or compare pose-estimation models | Initialize a DLC project → match/review keypoints → prepare training data → train → predict/evaluate | [Develop and evaluate models](#develop-and-evaluate-models) |

**DeepLabCut (DLC)** runs pose estimation using SuperAnimal models. **Anipose** calibrates cameras and reconstructs 3D positions from synchronized 2D tracks. **Slurm** schedules cluster jobs; **Singularity** runs the GPU software environment on the cluster.

The checked-in configuration describes MLB2: two cages, five cameras per cage, `.mkv` behavioral videos, and `.avi` calibration videos. Paths, camera names, and experiment settings must be reviewed before using it with another dataset. Model-development commands operate independently; training a model does not automatically replace the models used for session tracking.

### Guide contents

- [Installation](#installation)
- [Configure your paths and experiment](#configure-your-paths-and-experiment)
- [Process a session](#process-a-session)
- [Develop and evaluate models](#develop-and-evaluate-models)
- [Python interfaces](#python-interfaces)
- [Troubleshooting and verification](#troubleshooting-and-verification)

## Installation

### Choose where the work will run

| Environment | Used for | Required software |
| --- | --- | --- |
| Local computer or HPC host Python | Data preparation, camera calibration, conversion, filtering, triangulation, and job submission | Python 3.10+, this package, Anipose |
| HPC GPU container | Session tracking and queued model-development stages | Python 3.10+, DLC 3 with PyTorch/SuperAnimal support, pipeline dependencies, compatible GPU drivers |
| Local DLC environment, if available | Direct model matching, training, prediction, and evaluation | This package with model-development dependencies, DLC 3 and PyTorch |

On HPC, the host and GPU container are **separate Python environments**. Installing DLC in the host environment does not install it inside the container. Use the [HPC setup guide](docs/hpc-setup.md) to provision or verify both environments, including optional visualization dependencies.

### Install the pipeline

Clone or copy the **whole repository**, including `pyproject.toml`, `pipeline/`, and `config/`. Run installation commands from its root.

For a new local environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . anipose
```

For the existing HPC directory layout:

```bash
cd /scratch/lbshks/super_animal/consolidated_pipeline
python3 -m venv /scratch/lbshks/super_animal/.venv  # only if it does not exist
source /scratch/lbshks/super_animal/.venv/bin/activate
python -m pip install -e . anipose
```

If you already have a working Python/Anipose environment, activate it and run the installation command there. Configure `anipose_command` to use that environment as described below. Activate your environment again in each new terminal before running the pipeline.

Install extras only for the features you need:

| Feature | Installation command |
| --- | --- |
| Model-development configuration and reports | `python -m pip install -e ".[model-development]"` |
| Manual-label calibration evaluation | `python -m pip install -e ".[validation]"` |
| 3D skeleton preview videos | `python -m pip install "anipose[viz]"`, plus FFmpeg and a working rendering backend; see [HPC setup](docs/hpc-setup.md#optional-visualization-dependencies) |

The model-development extra provides reporting/configuration packages; DLC and PyTorch must also be installed in the environment executing GPU work.

Check the host installation:

```text
lb-pipeline --help
anipose --help
python -c "import numpy, pandas, tables, cv2; assert hasattr(cv2, 'aruco')"
python -m pip check
```

If `lb-pipeline` is unavailable, reinstall with `python -m pip install -e .` in the active environment. From the repository, `python -m pipeline.main` is an equivalent entry point.

## Configure your paths and experiment

Edit [config/setup.json](config/setup.json), or make a copy and pass it with `--config PATH`. It is the source of truth for configuration.

| Setting | What to review |
| --- | --- |
| `hosts.local` / `hosts.hpc` | Experiment storage, Anipose command, and the DLC project used for manual labeling (`model_folder`) |
| `hosts.hpc.repository_dir` | Full checkout containing `pyproject.toml`; also determines session-job log locations |
| `hosts.hpc.sandbox`, `cache_dir`, `tmp_dir`, `singularity_module` | GPU container and writable cluster paths |
| `cages` | Camera-name-to-perspective mappings for each cage |
| `calibration` | Calibration board dimensions and video sampling rate |
| `pose` | Video extensions, stock model selection, batch sizes, and preview retention |
| `filter` / `anipose` | Confidence filtering, camera count, skeleton constraints, and triangulation settings |
| `slurm` | Job partition, GPU request, time limit, and optional CPU/memory resources |
| `hosts.*.development_root` / `model_development` | Model-development projects and named experiments; see [the model guide](docs/model-development.md) |

The supplied Windows configuration invokes an existing Conda environment named `anipose`. If you installed Anipose in the same environment as the pipeline instead, set `hosts.local.anipose_command` to `["anipose"]` and activate that environment before running commands.

Use `--environment local` or `--environment hpc` to select host settings explicitly. Automatic selection uses `local` on Windows and `hpc` elsewhere. Session commands also accept `--experiment-dir PATH` to override session storage; model commands use `development_root` instead.

The pipeline generates `config.toml`, `cage_map.json`, and `jobs.json` in the experiment root as needed. **Edit setup JSON, not these generated files**: a later command can regenerate them. Data transfer between local and HPC storage is handled outside this package.

## Process a session

A **session** is one recording folder containing the synchronized camera videos and calibration recordings. Start with a ZIP, a raw session folder, or an already prepared session directory.

**Import behavior:** raw folders outside the selected experiment root are moved into that root; ZIPs are extracted there. Use a copy of a raw folder if the original must stay in place.

### Recommended HPC workflow

Once the host and GPU container are configured, activate the host environment and submit from the login node:

```bash
cd /scratch/lbshks/super_animal/consolidated_pipeline
source /scratch/lbshks/super_animal/.venv/bin/activate
lb-pipeline run /path/to/session.zip --environment hpc --slurm
```

This queues preparation and calibration, then submits a separate GPU tracking job if tracks are missing. Each submission reports its job ID and log paths. **It does not automatically schedule a continuation after GPU tracking.**

After tracking finishes, rerun with the **prepared session directory** under `hosts.hpc.experiment_dir`:

```bash
lb-pipeline run /scratch/lbshks/mlb2/experiment/SESSION_NAME --environment hpc --slurm
```

The rerun uses completed tracks and advances conversion, filtering, and triangulation. Successful trials produce CSV files under each cage's `pose-3d/`. Incomplete trials are reported with reasons.

### Other ways to run

| Command form | What runs immediately | What is queued |
| --- | --- | --- |
| `run INPUT --slurm` | Submission only | Host processing, then a separate GPU tracking job when needed |
| `run INPUT --slurm-pose` | Preparation/calibration and any ready downstream work | Missing GPU tracking only |
| `run INPUT` | Preparation/calibration and any ready downstream work | Nothing; missing GPU tracking is shown as a Slurm preview |
| `STAGE SESSION_DIR --slurm` | Submission only | The selected stage |

`--slurm` and `--slurm-pose` require HPC execution and cannot be combined. On HPC, run forms that do immediate processing inside a compute allocation. Use `--slurm` when submitting from the login node.

A session `run` without submission flags is **not a dry run**: it can move/extract data and perform CPU processing. To inspect only the missing tracking submission, use `lb-pipeline pose SESSION_DIR` on a prepared session.

For local preparation or processing of already available tracks:

```powershell
lb-pipeline prepare "C:\data\session.zip" --environment local
lb-pipeline run "C:\data\experiment\SESSION_NAME" --environment local
```

Replace these example paths with your input and the actual prepared directory. Session `pose` previews or submits a cluster job; it does not start local DLC inference automatically.

### Run individual stages

All session stages accept `--config`, `--environment`, `--experiment-dir`, and `--slurm`. Use `lb-pipeline STAGE --help` for stage-specific options.

| Command | Purpose | Main output |
| --- | --- | --- |
| `prepare ZIP_OR_FOLDER` | Import and organize camera videos, then downsample calibration videos | `calibration/`, `videos-raw/` |
| `downsample SESSION_DIR` | Resume calibration-video sampling; default 3 FPS | Low-FPS calibration `.avi` files |
| `calibrate SESSION_DIR` | Estimate camera geometry from ChArUco recordings | `calibration/calibration.toml` |
| `pose SESSION_DIR` | Preview GPU tracking; add `--slurm` to submit | `tracks/` HDF5 and adaptation reports |
| `convert SESSION_DIR` | Validate synchronized camera trials and normalize tracks | `pose-2d/` |
| `filter SESSION_DIR` | Convert ready tracks and apply temporal filtering | `pose-2d-filtered/` |
| `triangulate SESSION_DIR` | Convert, filter, and optimize ready 3D trials | `pose-3d/` CSV files |
| `label-3d SESSION_DIR` | Optionally render 3D skeleton animations | `videos-3d/` |

The full `run` performs the core stages through triangulation. Visualization and the following manual calibration check are optional:

```text
lb-pipeline labels SESSION_DIR --frame-index 100
lb-pipeline evaluate SESSION_DIR --labels LABEL_CSV_DIR
```

`labels` extracts frames into the configured DLC project (or `--dlc-project PATH`). Label those frames manually before running `evaluate`, which checks camera reprojection error. This differs from `model evaluate`, which compares predicted keypoints with ground truth.

### Outputs, resuming, and job status

```text
EXPERIMENT_ROOT/
  config.toml                   # generated Anipose settings
  SESSION_NAME/
    pipeline_state.json         # saved session tracking job ID
    CAGE1/
      calibration/              # calibration videos and calibration.toml
      videos-raw/               # behavioral camera videos
      tracks/                   # SuperAnimal predictions and adaptation reports
      pose-2d/                  # converted Anipose tracks
      pose-2d-filtered/          # filtered tracks
      pose-3d/                  # reconstructed 3D CSVs
      videos-3d/                # optional skeleton previews
    CAGE2/
      ...
```

Rerunning stages skips verified outputs. Conversion requires the configured synchronized camera set, matching frame counts, shared keypoints, and completed adapted tracks. A preview video alone is not a completed track. Keep `videos-raw/` available while processing or resuming the session; trial discovery uses those files.

Session-job logs are in `hosts.hpc.repository_dir/logs/`. Use the reported job ID with `squeue -j JOB_ID` or `scancel JOB_ID`. A saved tracking job ID prevents duplicate submission. If that job has ended without complete tracks, retry explicitly:

```bash
lb-pipeline pose /path/to/prepared/session --environment hpc --slurm --retry-job
```

Other session-stage submissions create new jobs. All queued stages share the configured `slurm` resources, including its GPU request even for CPU processing. Host jobs inherit the submitting environment and use its Python interpreter, unless `hosts.hpc.python_executable` overrides it.

The default settings omit/delete 2D labeled preview videos to save space. For retention settings, filtering behavior, `.stale` output handling, storage guidance, and missing-result diagnosis, see [session processing details](docs/session-processing.md).

## Develop and evaluate models

Use this workflow for experiments with stock SuperAnimal models or developing fine-tuned DLC projects. Begin with the [model-development guide](docs/model-development.md) and [example experiment definitions](config/model-development.example.json); the default setup has no development experiments enabled.

| Stage | Purpose |
| --- | --- |
| `model init EXPERIMENT` | Create a project, or explicitly adopt one with `--adopt` |
| `model match EXPERIMENT` | Propose keypoint correspondences and generate review diagnostics |
| `model prepare EXPERIMENT --conversion-table PATH` | Apply your reviewed mapping and prepare the training split |
| `model train EXPERIMENT` | Fine-tune using ordinary training or memory replay |
| `model predict EXPERIMENT` | Apply stock/project checkpoints to images or videos |
| `model evaluate EXPERIMENT` | Evaluate DLC train/test splits or an external labeled benchmark |

For a configured `top-finetune` experiment in a suitable local DLC environment:

```text
lb-pipeline model init top-finetune
lb-pipeline model match top-finetune
```

Review and correct the proposed conversion table before continuing:

```text
lb-pipeline model prepare top-finetune --conversion-table reviewed/top.csv
lb-pipeline model train top-finetune
```

Model stages accept `--environment hpc --slurm` for container jobs, or `--preview` to inspect configuration without executing. Wait for each stage to finish before starting its dependent stage. Project prediction/evaluation requires explicit pose and detector checkpoint paths. The model guide covers those commands, comparing training variants on the same split, benchmark metrics, and cache/resume behavior.

## Python interfaces

The session workflow is also available as a Python function:

```python
from pipeline import run_pose_detection_pipeline

result = run_pose_detection_pipeline(
    source_path="session.zip",
    experiment_dir=None,  # use the selected host's configured directory
    setup_json="config/setup.json",
    submit_jobs=False,
    environment="local",
)
print(result.status, result.pending_videos, result.triangulation.outputs)
```

This has the same processing/import behavior as CLI `run`. For development stages, use the [ModelDevelopment Python interface](docs/model-development.md#hpc-and-python).

## Troubleshooting and verification

| Question or problem | Guide |
| --- | --- |
| Host versus container dependencies, FFmpeg, Mayavi, headless rendering | [HPC setup](docs/hpc-setup.md) |
| Incomplete tracks, missing 3D CSVs, RANSAC errors | [Session troubleshooting](docs/session-processing.md#incomplete-tracks-and-missing-3d-results) |
| Filtering, preview retention, or limited HPC storage | [Session processing details](docs/session-processing.md) |
| Reviewed mappings, checkpoint selection, benchmark interpretation | [Model development](docs/model-development.md) |

Run the automated tests from the checkout:

```text
python -m unittest discover -s tests -v
```

Optional integration checks use real local data or a GPU environment:

```text
python tests/smoke_calibration.py
python tests/smoke_anipose.py
python tests/smoke_model_development.py --help
```

The calibration check needs an existing five-camera CAGE2 calibration-video set under the configured local experiment root. The triangulation check needs an existing CAGE2 calibration and Python 3.11+; it generates synthetic 2D poses in a temporary project. The model smoke workflow requires DLC/PyTorch with CUDA, small configured datasets, and a reviewed conversion table. Unit tests alone do not verify GPU training or inference.

For older installations: the CLI is now `lb-pipeline`; `--job` and `--submit` were replaced by `--slurm` and `--slurm-pose` respectively.
