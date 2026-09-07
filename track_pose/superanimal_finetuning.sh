#!/bin/bash
#SBATCH -D /scratch/lbshks/super_animal  # the working directory from where the commands are executed. The folder /scratch/lbshks/ contains the deeplabcut.sif singularity container.
#SBATCH --job-name=sa_finetuning_test
#SBATCH --output=logs/R-%x.%j.out     # output file job-name.id.out
#SBATCH --error=logs/R-%x.%j.err	  # error file job-name.id.err
#SBATCH --partition=ex_scioi_gpu,scioi_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --time=0-04:00 # 3-04:00 means 3 days and 4 hours
# On the Tesla P100s tests have shown that 0.3 hours per video on average is enough (plus some additional slack)

#Job-Status per Mail:
#SBATCH --mail-type=ALL  # will send an email when a job begins and when it ends
#SBATCH --mail-user=g.parra@campus.tu-berlin.de # please change!

# -----------------------------
# Paths
# -----------------------------
WORKDIR=/scratch/lbshks
SANDBOX=$WORKDIR/super_animal/deeplabcut_sandbox
CACHES=$WORKDIR/super_animal/cache

# Ensure directories exist
mkdir -p $CACHES
mkdir -p $WORKDIR/super_animal/tmp

# -----------------------------
# Load modules
# -----------------------------
module load singularity/3.7.4

# -----------------------------
# Run DeepLabCut (GPU-enabled)
# -----------------------------
singularity exec --nv --containall \
    --bind $WORKDIR:/wd/lbshks \
    --bind $CACHES:/cache \
    $SANDBOX \
    bash -c "export DLC_MODELZOO_PATH=/cache/dlc_modelzoo && \
             export TORCH_HOME=/cache/torch_cache && \
             export HF_HOME=/cache/hf_cache && \
             export TRANSFORMERS_CACHE=/cache/hf_cache && \
             export TMPDIR=/wd/lbshks/super_animal/tmp && \
             export DLC_TMP_DIR=/wd/lbshks/super_animal/tmp && \
             mkdir -p \$TMPDIR /cache/torch_cache /cache/hf_cache && \
             python3 /wd/lbshks/super_animal/superanimal_naive_finetuning.py \
                 /wd/lbshks/super_animal/superanimal_top-MLB23-2025-11-26/config_hpc.yaml \
                 superanimal_topviewmouse \
                 hrnet_w32"