# SuperAnimal model development

[Back to the README](../README.md) · [Installation and configuration](../README.md#installation) · [HPC setup](hpc-setup.md)

`lb-pipeline model` is used to control DLC/Superanimal model creation, fine-tuning, and evaluation.

Follow this guide after installing the pipeline. Model development uses named
experiments and labeled datasets; the separate [session workflow](../README.md#process-a-session)
processes camera recordings into 3D poses.

- [Configure a development experiment](#configure)
- [Initialize, match, review, and train](#develop)
- [Apply models and evaluate results](#apply-and-evaluate)
- [Run on HPC or through Python](#hpc-and-python)
- [Validation and source provenance](#validation-and-provenance)

## Configure

Install reporting/configuration dependencies with `python -m pip install -e ".[model-development]"`. GPU operations require **DLC 3 with PyTorch and SuperAnimal dependencies**, normally in the configured HPC container. DLC is imported lazily; metrics work without it.

Merge the `model_development` object from [the example](../config/model-development.example.json) into your copy of `config/setup.json`. It provides complete top and sidefront experiment definitions. Edit data paths, scorer, bodyparts, and batch sizes. Set `hosts.local.development_root` and `hosts.hpc.development_root`. All relative experiment and CLI paths resolve beneath that host's development root.

```text
model-development/
  data/top/development/       # videos and labeled-data folders
  data/top/benchmark/         # independent labeled-data folders
  data/top/inspection/        # images or short videos
  data/sidefront/...
  reviewed/                  # manually reviewed gt,MasterName CSVs
  projects/top/              # normal DLC project structure
    model-development/
      split.json
      shuffle-0/             # stage records, predictions, reports
  projects/sidefront/
```

Each labeled folder must match a video stem in DLC `video_sets`, contain its images, and contain `CollectedData_<scorer>.h5` or a DLC multi-header CSV. HDF5 takes precedence over its CSV copy. Extra annotation columns (such as lockbox objects) are allowed; every configured target must be present. Initialization copies source data. The top example selects nose, ears and tail base to avoid forcing paw correspondences absent from that base; edit the target set deliberately. Benchmark datasets stay separate; matching/preparation/training reject benchmark overlap by frame identity or image content.

## Develop

```bash
lb-pipeline model init top-finetune
lb-pipeline model match top-finetune
```

For an existing project, set its path/bodyparts in the experiment and use `model init top-finetune --adopt`. Runtime config copies relocate its project path without modifying the original config. Choose an unused shuffle when preparing new training in an existing project.

Inspect `memory_replay/confusion_matrix.png`, `conversion_table.csv`, and `pseudo_predictions.json`. Copy and correct the proposed CSV into `reviewed/top.csv`. The required columns are `gt,MasterName`: project landmark and corresponding SuperAnimal landmark. Blank `gt` rows for unused SuperAnimal channels are allowed. Every target must have a reviewed assignment, with no duplicate or unknown source names. If no anatomical correspondence exists, revise the target set/initialization approach; do not assign an unrelated point. Existing questionable mappings are not adopted as defaults.

```bash
lb-pipeline model prepare top-finetune --conversion-table reviewed/top.csv
lb-pipeline model train top-finetune
```

The first prepared variant establishes the persisted DLC split. To compare memory replay, duplicate the experiment as `top-replay`, retain project/data/seed/fraction, set `mode: memory_replay`, and choose a new shuffle (e.g. `1`). Run prepare and train again. It reuses the original split through DLC's existing-split API. Ordinary `finetune` trains project targets; `memory_replay` retains SuperAnimal channels. Detector training defaults to zero epochs. Pose epochs, batch size, and save interval are explicit settings.

Completed stages reuse verified artifacts. Changed mappings/training recipes require new shuffles; changed development data/splits require a new project. Interrupted training requires `--resume-checkpoint PATH` and also `--resume-detector PATH` if detector training was enabled. There is no implicit latest checkpoint. The CLI locks each project; after a crash, verify the worker has stopped before removing a stale `.model-development.lock`.

## Apply and evaluate

```bash
lb-pipeline model predict top-finetune --source stock --inputs data/top/inspection --kind images --overlays
lb-pipeline model evaluate top-finetune --source stock --scope external

lb-pipeline model predict top-finetune --source project --inputs data/top/inspection --kind videos --checkpoint POSE.pt --detector-checkpoint DETECTOR.pt --overlays
lb-pipeline model evaluate top-finetune --source project --scope both --checkpoint POSE.pt --detector-checkpoint DETECTOR.pt
```

Replace checkpoint placeholders with actual paths printed by training. Split evaluation requires snapshots in the configured shuffle's train directory. `--scope split` uses DLC's train/test split, `external` uses the configured benchmark, and `both` runs them separately. External evaluation obtains missing predictions through the same prediction implementation. Development inference uses fixed weights without per-video adaptation.

Prediction caches include checkpoint/configuration/input contents and DLC version. Report identities also include labels and metric settings, so changing a threshold reuses predictions. Fine-tuned memory-replay output channels use the saved conversion array. Stock benchmarking uses the reviewed benchmark conversion table; unlike training, this table may omit unsupported target landmarks. Configure the same explicit `benchmark.bodyparts` across compared models. Unsupported points are reported and remain missing predictions rather than silently reducing the benchmark.

External reports contain `summary.csv` (overall, folder, bodypart, group, threshold), `points.csv`, `confidence.png`, and `report.json` (coverage, overlap flags, provenance). Thresholds include the configured default (0.5) and 0.0–0.9.

Missing ground-truth x/y follows the source scripts' **not visible** convention; this is unsuitable if missing means unannotated. Missing predictions count as undetected. Detections require finite x/y/confidence and confidence strictly above threshold. Visibility precision/recall/accuracy measure presence detection, not localization. Pixel errors average visible detected points. Localization success divides detections within the configured radius (default 5 px) by all visible points. Outliers exceed 50 px among visible detected points. Undefined ratios are NaN with counts provided. External overlap produces `held_out: false`; those scores must not be called held-out performance.

## HPC and Python

All stages accept `--config PATH`, `--environment hpc`, `--preview`, and `--slurm`:

```bash
lb-pipeline model train top-finetune --environment hpc --preview
lb-pipeline model match top-finetune --environment hpc --slurm
lb-pipeline model prepare top-finetune --conversion-table reviewed/top.csv --environment hpc --slurm
lb-pipeline model train top-finetune --environment hpc --slurm
```

Wait for each stage before submitting its dependent stage. Preview does not mutate projects or submit jobs. `.submissions/` contains frozen setup, reviewed-table copy for preparation, job script, job ID, and logs. Selected checkpoint/project config hashes are verified at worker startup. Projects/cache/temp are writable; external source datasets are read-only container mounts. Existing Slurm resource settings apply. Copy complete projects when relocating; release bundles are unnecessary.

```python
from pipeline import ModelDevelopment
from pipeline.config import Setup

dev = ModelDevelopment(Setup.load('config/setup.json').data, 'top-finetune', 'local')
with dev.lock():
    result = dev.prepare(dev.path('reviewed/top.csv'))
print(result.status, result.artifacts)
```

Methods `initialize`, `match`, `prepare`, `train`, `predict`, and `evaluate` return `ModelStageResult`. Table/metric functions live in `pipeline.model_evaluation` without DLC dependencies.

## Validation and provenance

Run `python -m unittest discover -s tests -v` for CPU regressions. In a GPU environment, run `tests/smoke_model_development.py --help` for the smoke workflow. Use small development/benchmark datasets, a reviewed table, a short inspection video, and two experiment variants with short epoch counts. It exercises both modes and stock/project evaluation; review its CSV counts and overlays before longer training.

The implementation consolidates the workflow in `mouse_lockbox/evaluate/analyze_finetuned_superanimal.ipynb` and the evaluation design from Niek Andresen's `evaluate.py` / `evaluate_superanimal.py` (August 2023). Those external files remain unchanged. Shared code corrects ambiguous file selection, dropped missing frames, CSV parsing, zero denominators, and visibility/localization naming. DLC integration follows the [official SuperAnimal tutorial](https://deeplabcut.github.io/DeepLabCut/examples/COLAB/COLAB_YOURDATA_SuperAnimal.html) and PyTorch APIs.
