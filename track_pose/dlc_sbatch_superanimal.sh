#!/bin/bash
#SBATCH -D /scratch/lbshks/super_animal
#SBATCH --job-name=superanimal-mlb2
#SBATCH --output=logs/submission-%x.%j.out
#SBATCH --error=logs/submission-%x.%j.err
#SBATCH --partition=ex_scioi_gpu,scioi_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --time=1-03:00

#Job-Status per Mail:
#SBATCH --mail-type=ALL  # will send an email when a job begins and when it ends
#SBATCH --mail-user=g.parra@campus.tu-berlin.de # please change!

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 VIDEO_ROOT EXTENSION RESULT_ROOT MAP [CAGE_MAP_FILE]" >&2
    echo "  MAP: default, rats, or an absolute/relative JSON map path" >&2
}

if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    usage
    exit 2
fi

VIDEO_ROOT=$(realpath "$1")
EXTENSION=$2
mkdir -p "$3"
RESULT_ROOT=$(realpath "$3")
MAP=$4
CAGE_MAP=${5:-}

# Slurm reads SBATCH directives before positional arguments exist. Redirect the
# actual pipeline output here so each experiment keeps its logs beside tracks.
LOG_DIR="$RESULT_ROOT/logs"
mkdir -p "$LOG_DIR"
JOB_NAME=${SLURM_JOB_NAME:-superanimal}
JOB_NAME=${JOB_NAME//\//_}
JOB_ID=${SLURM_JOB_ID:-manual}
exec >"$LOG_DIR/${JOB_NAME}.${JOB_ID}.out" 2>"$LOG_DIR/${JOB_NAME}.${JOB_ID}.err"

if [ ! -d "$VIDEO_ROOT" ]; then
    echo "VIDEO_ROOT is not a directory: $VIDEO_ROOT" >&2
    exit 2
fi
if [ -z "${EXTENSION#.}" ]; then
    echo "EXTENSION cannot be empty" >&2
    exit 2
fi

MAP_OPTION=(--perspective-preset "$MAP")
MAP_BIND=()
if [ "$MAP" != "default" ] && [ "$MAP" != "rats" ] && [ "$MAP" != "mlb2_cage1" ] && [ "$MAP" != "mlb2_cage2" ]; then
    MAP_PATH=$(realpath "$MAP")
    if [ ! -f "$MAP_PATH" ]; then
        echo "MAP must be 'default', 'rats', or a JSON file: $MAP" >&2
        exit 2
    fi
    MAP_OPTION=(--perspective-map-file "$MAP_PATH")
    MAP_PARENT=$(dirname "$MAP_PATH")
    MAP_BIND=(--bind "$MAP_PARENT:$MAP_PARENT:ro")
fi

CAGE_MAP_OPTION=()
CAGE_MAP_BIND=()
if [ -n "$CAGE_MAP" ]; then
    CAGE_MAP_PATH=$(realpath "$CAGE_MAP")
    if [ ! -f "$CAGE_MAP_PATH" ]; then
        echo "CAGE_MAP_FILE must be a JSON file: $CAGE_MAP" >&2
        exit 2
    fi
    CAGE_MAP_OPTION=(--cage-map-file "$CAGE_MAP_PATH")
    CAGE_MAP_PARENT=$(dirname "$CAGE_MAP_PATH")
    CAGE_MAP_BIND=(--bind "$CAGE_MAP_PARENT:$CAGE_MAP_PARENT:ro")
    MAP_OPTION=()
fi

WORKDIR=/scratch/lbshks
SANDBOX=$WORKDIR/super_animal/deeplabcut_sandbox
CACHES=$WORKDIR/super_animal/cache
TMP_DIR=$WORKDIR/super_animal/tmp
mkdir -p "$CACHES/torch_cache" "$CACHES/hf_cache" "$TMP_DIR"

module load singularity

if [ "$VIDEO_ROOT" = "$RESULT_ROOT" ]; then
    # Experiment mode reads videos and writes sibling tracks below one root.
    BIND_ARGS=(--bind "$WORKDIR:/wd/lbshks" --bind "$VIDEO_ROOT:$VIDEO_ROOT" --bind "$CACHES:/cache")
else
    BIND_ARGS=(--bind "$WORKDIR:/wd/lbshks" --bind "$VIDEO_ROOT:$VIDEO_ROOT:ro" --bind "$RESULT_ROOT:$RESULT_ROOT" --bind "$CACHES:/cache")
fi
PIPELINE_OPTIONS=()
if [ "${SUPERANIMAL_EXPERIMENT_LAYOUT:-0}" = "1" ]; then
    PIPELINE_OPTIONS+=(--output-layout sibling-tracks --video-directory-name videos-raw --tracks-directory-name tracks)
fi
if [ "${SUPERANIMAL_SKIP_EXISTING:-0}" = "1" ]; then
    PIPELINE_OPTIONS+=(--skip-existing)
fi
if [ -n "${SUPERANIMAL_INFERENCE_BATCH_SIZE:-}" ]; then
    PIPELINE_OPTIONS+=(--inference-batch-size "$SUPERANIMAL_INFERENCE_BATCH_SIZE")
fi
if [ -n "${SUPERANIMAL_DETECTOR_BATCH_SIZE:-}" ]; then
    PIPELINE_OPTIONS+=(--detector-batch-size "$SUPERANIMAL_DETECTOR_BATCH_SIZE")
fi
if [ -n "${SUPERANIMAL_ADAPT_BATCH_SIZE:-}" ]; then
    PIPELINE_OPTIONS+=(--adapt-batch-size "$SUPERANIMAL_ADAPT_BATCH_SIZE")
fi
singularity exec --nv --containall "${BIND_ARGS[@]}" "${MAP_BIND[@]}" "${CAGE_MAP_BIND[@]}" "$SANDBOX" \
    env DLC_MODELZOO_PATH=/cache/dlc_modelzoo \
        TORCH_HOME=/cache/torch_cache \
        HF_HOME=/cache/hf_cache \
        TRANSFORMERS_CACHE=/cache/hf_cache \
        TMPDIR=/wd/lbshks/super_animal/tmp \
        DLC_TMP_DIR=/wd/lbshks/super_animal/tmp \
        python3 /wd/lbshks/super_animal/batch_apply_superanimal.py \
            --video-path "$VIDEO_ROOT" \
            --video-extension "$EXTENSION" \
            --result-folder "$RESULT_ROOT" \
            "${MAP_OPTION[@]}" \
            "${CAGE_MAP_OPTION[@]}" \
            "${PIPELINE_OPTIONS[@]}"
