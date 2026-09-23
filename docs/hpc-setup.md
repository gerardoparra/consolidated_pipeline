# HPC installation and optional visualization

[Back to the README](../README.md) · [Session troubleshooting](session-processing.md)

Use this guide when provisioning the cluster environments or enabling 3D preview rendering. The paths below match the checked-in MLB2 setup; replace them if your storage layout differs.

- [Host environment](#host-environment)
- [DLC GPU container](#dlc-gpu-container)
- [Optional visualization dependencies](#optional-visualization-dependencies)

## Host environment

There are two Python environments. Preparation, calibration, conversion, and
triangulation use a **host Python environment** with this package and Anipose.
The GPU Slurm worker uses Python **inside the Singularity sandbox**; installing
DeepLabCut only in the host environment will not make it available to the worker.
Without `--slurm`, `run` performs preparation and calibration in the shell
that invokes it, so use a CPU allocation for that form. With `--slurm`, submit
directly from the login node; CPU work is queued and the GPU worker is submitted
from that job when needed.

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

## DLC GPU container

The worker script loads the configured `singularity` module and executes
`hosts.hpc.sandbox` (currently `deeplabcut_sandbox`) with `--nv`. That sandbox
must have Python 3.10+, a GPU-compatible PyTorch installation, DeepLabCut 3 with
PyTorch and SuperAnimal/Model Zoo support, and the package runtime dependencies (`numpy`,
`pandas`, `tables`, and `opencv-contrib-python`). When building or updating a
**writable** sandbox, install those Python packages inside it, using the
container's Python and a PyTorch build compatible with the cluster driver:

```bash
python3 -m pip install 'deeplabcut[gui,modelzoo]' numpy pandas tables opencv-contrib-python PyYAML matplotlib
```

PyYAML and matplotlib support the model-development stages as well as their reports.
Verify that the installed DLC version exposes the PyTorch APIs used by those stages.

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

For model development, also check the APIs used by its worker in that same GPU container:

```python
from deeplabcut.pose_estimation_pytorch.apis import (
    analyze_image_folder, train_network, evaluate_network,
)
from deeplabcut.modelzoo import build_weight_init
```

Finally, confirm that `sbatch`, the configured Singularity module and sandbox,
the model directory, and writable experiment/cache/temp paths exist. The
worker writes its logs under the project directory. SuperAnimal model weights
may download on first use, so the configured cache must also be usable from a
compute node. Transfer session data to HPC storage separately; the pipeline
does not copy it between hosts.

Session-stage logs live under `hosts.hpc.repository_dir/logs/`. Model-development
submission snapshots and logs live under `hosts.hpc.development_root/.submissions/`.
The host and container must both be able to read the configured checkout and data.

## Optional visualization dependencies

The `viz` extra installs Mayavi and VTK, which are large optional dependencies.
Anipose also uses scikit-video to encode the result, so the real `ffmpeg` and
`ffprobe` executables must be on `PATH`; installing a Python package named
`ffmpeg` does not provide them. First check for an HPC module, then verify both
executables. Module names differ by cluster:

```bash
module spider ffmpeg 2>/dev/null || module avail ffmpeg
# If a module is listed, load the listed version, for example:
module load ffmpeg

command -v ffmpeg
command -v ffprobe
ffmpeg -version
ffprobe -version
```

### FFmpeg without a cluster module

If the cluster has no FFmpeg module, install the FFmpeg system binaries in a
small user-managed Micromamba prefix. Micromamba is a standalone Conda-compatible
executable and does not require an administrator or a cluster Conda module. Keep
it and its package cache on scratch rather than `/tmp`:

```bash
# This URL is for the common Intel/AMD Linux architecture.
uname -m  # should print x86_64

PIPELINE_TOOLS=/scratch/lbshks/super_animal/tools
MAMBA_ROOT_PREFIX=/scratch/lbshks/super_animal/micromamba
FFMPEG_PREFIX=/scratch/lbshks/super_animal/ffmpeg

mkdir -p "$PIPELINE_TOOLS"
cd "$PIPELINE_TOOLS"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj bin/micromamba

export MAMBA_ROOT_PREFIX
"$PIPELINE_TOOLS/bin/micromamba" create -y \
  -p "$FFMPEG_PREFIX" -c conda-forge ffmpeg
```

This prefix is only a source of executables. Do not activate it, because that
could replace the Python from the working pipeline `.venv`. Instead, activate
the existing Python environment and prepend the FFmpeg binary directory:

```bash
source /scratch/lbshks/super_animal/.venv/bin/activate
export PATH=/scratch/lbshks/super_animal/ffmpeg/bin:$PATH

command -v python
command -v anipose
command -v ffmpeg
command -v ffprobe
python -c 'import skvideo; print("scikit-video FFmpeg:", skvideo.getFFmpegPath())'
ffmpeg -version
ffprobe -version
```

The `python` and `anipose` paths should remain under
`/scratch/lbshks/super_animal/.venv`; `ffmpeg` and `ffprobe` should be under
`/scratch/lbshks/super_animal/ffmpeg/bin`. Run the activation and `PATH` export
in each new shell and in any Slurm script that renders previews. Both programs
come from the same Conda-forge FFmpeg installation.

### A dedicated rendering job

The ordinary `label-3d --slurm` command uses the shared Slurm resource settings, including its GPU request. The custom script below is an alternative when you want a separate CPU allocation.

Rendering thousands of Mayavi frames is CPU work and can take hours, so submit
it as a CPU job instead of leaving it on the login node. Save the following as
`render_3d.sbatch`; add the cluster's CPU partition or account directives if
they are required:

```bash
#!/bin/bash
#SBATCH --job-name=mlb2-render-3d
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/lbshks/super_animal/logs/render-3d-%j.out
#SBATCH --error=/scratch/lbshks/super_animal/logs/render-3d-%j.err

set -euo pipefail
source /scratch/lbshks/super_animal/.venv/bin/activate
export PATH=/scratch/lbshks/super_animal/ffmpeg/bin:$PATH
cd /scratch/lbshks/super_animal/consolidated_pipeline

python -m pipeline.main label-3d "$1"
```

Submit one session with:

```bash
mkdir -p /scratch/lbshks/super_animal/logs
sbatch render_3d.sbatch \
  /scratch/lbshks/mlb2/experiment/SESSION_NAME
```

Mayavi depends on VTK and a rendering backend; Linux compute nodes without a
display may also need the cluster's Xvfb package or module. Test the import and
display helper before starting a long render:

```bash
python -c 'from mayavi import mlab; print("Mayavi import OK")'
command -v xvfb-run
```

If Mayavi fails to build in your host environment, use a separate compatible
visualization environment. Install the pipeline and visualization dependencies
there, make its `anipose` command available, and retain access to the source
video headers, generated experiment configuration, and 3D CSVs.
