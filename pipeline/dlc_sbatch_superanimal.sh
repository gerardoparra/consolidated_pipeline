#!/usr/bin/env bash
# All cluster paths and job resources come from config/setup.json via --export.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: sbatch ... dlc_sbatch_superanimal.sh VIDEO_ROOT EXTENSION RESULT_ROOT" >&2
    exit 2
fi

VIDEO_ROOT=$1
EXTENSION=$2
RESULT_ROOT=$3
: "${PIPELINE_REPOSITORY:?}"
: "${PIPELINE_SETUP:?}"
: "${PIPELINE_SANDBOX:?}"
: "${PIPELINE_CACHE:?}"
: "${PIPELINE_TMP:?}"
: "${PIPELINE_MODULE:?}"

mkdir -p "$PIPELINE_CACHE/torch_cache" "$PIPELINE_CACHE/hf_cache" "$PIPELINE_TMP"
module load "$PIPELINE_MODULE"

singularity exec --nv --containall \
    --bind "$PIPELINE_REPOSITORY:$PIPELINE_REPOSITORY:ro" \
    --bind "$VIDEO_ROOT:$VIDEO_ROOT" \
    --bind "$PIPELINE_CACHE:$PIPELINE_CACHE" \
    --bind "$PIPELINE_TMP:$PIPELINE_TMP" \
    "$PIPELINE_SANDBOX" \
    env PYTHONPATH="$PIPELINE_REPOSITORY" \
        DLC_MODELZOO_PATH="$PIPELINE_CACHE/dlc_modelzoo" \
        TORCH_HOME="$PIPELINE_CACHE/torch_cache" \
        HF_HOME="$PIPELINE_CACHE/hf_cache" \
        TRANSFORMERS_CACHE="$PIPELINE_CACHE/hf_cache" \
        TMPDIR="$PIPELINE_TMP" \
        python3 -m pipeline.main worker \
            --config "$PIPELINE_SETUP" \
            --environment hpc \
            --video-root "$VIDEO_ROOT" \
            --result-root "$RESULT_ROOT" \
            --video-extension "$EXTENSION" \
            --inference-batch-size "${PIPELINE_INFERENCE_BATCH_SIZE:-8}" \
            --detector-batch-size "${PIPELINE_DETECTOR_BATCH_SIZE:-8}" \
            --adapt-batch-size "${PIPELINE_ADAPT_BATCH_SIZE:-8}"
