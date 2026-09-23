# Session processing: details and troubleshooting

[Back to the README](../README.md) · [HPC setup](hpc-setup.md)

For the first session, follow the [session workflow](../README.md#process-a-session).
Use this guide to interpret outputs, adjust filtering, render previews, or diagnose failures.

- [Incomplete tracks and missing 3D results](#incomplete-tracks-and-missing-3d-results)
- [Filtering and optimized triangulation](#filtering-and-optimized-triangulation)
- [Keeping or removing 2D previews](#keeping-or-removing-2d-previews)
- [Optional 3D visualization](#optional-3d-visualization)
- [Working with limited HPC storage](#working-with-limited-hpc-storage)

## Incomplete tracks and missing 3D results

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
leaves points without enough camera observations as `NaN` in the 3D CSV.
RANSAC remains disabled in the MLB2 defaults. Lowering the score threshold is
not a safe general workaround because it can admit the `-1` coordinates used
for missing detections.

```bash
# After changing config/setup.json:
python -m pipeline.main triangulate /scratch/lbshks/mlb2/experiment/SESSION_NAME
```

## Filtering and optimized triangulation

The MLB2 defaults enable Anipose's temporal 2D threshold filter and
constrained 3D optimization. The settings remain in `config/setup.json`, which
generates these sections in `config.toml`:

```json
"filter": {
  "enabled": true,
  "type": "medfilt",
  "medfilt": 5,
  "offset_threshold": 25,
  "score_threshold": 0.6,
  "spline": false
}
```

The triangulation settings retain `ransac: false` and set `optim: true`, with
`scale_smooth: 4`, `scale_length: 2`, `scale_length_weak: 0.5`, and
`n_deriv_smooth: 1`. Optimization is split into 10,000-frame chunks so long
MLB2 trials do not have to be optimized as one large problem. Linear
interpolation is used initially because it is less prone than cubic splines to
overshoot during rapid mouse movement.

To run only the temporal filtering stage and inspect its outputs:

```bash
python -m pipeline.main filter /scratch/lbshks/mlb2/experiment/SESSION_NAME
```

To perform all ready pose stages, including conversion, filtering, and
optimized triangulation, use the single command:

```bash
python -m pipeline.main triangulate /scratch/lbshks/mlb2/experiment/SESSION_NAME
```

The full `run` command follows the same sequence automatically. The pipeline
records the filter and triangulation settings used for each cage. When inputs
or relevant settings change, stale filtered 2D files are replaced. Outdated
3D CSVs and corresponding previews are archived with a `.stale` suffix before
recomputation. Run `label-3d` after triangulation to render new previews.

The median filter rejects low-confidence points and abrupt jumps, then
interpolates each camera's remaining gaps in time. Stock Anipose does not set
a maximum interpolation gap, and it only interpolates a coordinate when more
than half of that coordinate's samples are valid. The 3D optimizer then uses
the filtered camera views, neighboring frames, and the configured body-segment
constraints. These stages cannot create new camera evidence. In particular,
long `tail_end` occlusions should still be treated cautiously even if the
optimizer returns coordinates.

Anipose 1.1.24 uses an older positional form of pandas' `to_hdf` call that is
incompatible with current pandas releases. The pipeline applies a
process-local compatibility shim when it launches `anipose filter`; no pandas
downgrade or edit under `site-packages` is needed. If filtering still reports
the positional `NDFrame.to_hdf()` error, pull the current pipeline checkout
before rerunning the command.

## Keeping or removing 2D previews

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

## Optional 3D visualization

The default pipeline stops after writing `pose-3d` CSV files. The optional
pipeline `label-3d` command renders those coordinates as standalone skeleton
animations. It is not part of the default `run` and requires
Anipose's Mayavi visualization dependencies described in [HPC setup](hpc-setup.md#optional-visualization-dependencies). The configured labeling scheme connects
`left_eye` → `nose` → `right_eye` and `nose` → `tail_base` → `tail_end`, matching the
current SuperAnimal output.

`visualization.preview_fps` in `config/setup.json` sets the maximum rendered
frame rate and defaults to 10 FPS. The renderer reads the original frame rate from
one matching `.mkv` header, samples the `pose-3d` coordinate rows, and resets
their preview frame numbers before invoking Anipose. It does not decode,
downsample, or create another copy of the source video. For example, a
30-minute, 30 FPS trial has about 54,000 source frames; a 10 FPS preview renders
about 18,000 frames, while a 5 FPS preview renders about 9,000. Both retain the
30-minute playback duration. Change the setting to `5.0` if the videos are only
for quick quality control.

The worker includes a compatibility adapter for Anipose's scikit-video 1.1.11,
which still calls the removed NumPy `ndarray.tostring()` method. It converts
that call to `tobytes()` at runtime without changing packages in `.venv`.

The command writes animations under each cage's `videos-3d/` directory and
skips existing nonempty videos:

```bash
cd /scratch/lbshks/super_animal/consolidated_pipeline
python -m pipeline.main label-3d /scratch/lbshks/mlb2/experiment/SESSION_NAME
```

On a headless compute node, use a virtual framebuffer if `xvfb-run` is
available:

```bash
cd /scratch/lbshks/super_animal/consolidated_pipeline
xvfb-run -a --server-args="-screen 0 1280x1024x24" \
  python -m pipeline.main label-3d /scratch/lbshks/mlb2/experiment/SESSION_NAME
```

`anipose label-combined` additionally requires retained 2D labeled videos, so
it is not compatible with the default space-saving settings that omit or delete
those previews.

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
`lb-pipeline run SESSION_DIR --environment hpc --slurm` from the
activated HPC host environment on the login node. This queues preparation and
calibration, then submits tracking; rerun it after the GPU job finishes
to convert and triangulate ready trials. After verifying and backing up the
calibration and pose outputs, remove unneeded videos from scratch before
transferring the next session. The pipeline discovers trials from `videos-raw/`,
so deleting those files prevents a later `run` or `convert` from resuming that
session; finish and verify 3D output first.
