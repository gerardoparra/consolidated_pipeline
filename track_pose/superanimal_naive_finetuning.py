"""
    @author Gerardo Parra
    @date November 2025

    This script is used to fine-tune a SuperAnimal Model Zoo model on a new dataset with
    existing labeled data and keypoint matching in the DLC config.yaml file.

    To run this script in Singularity on the HPC one can use a script like the
    following and do 'srun the_following.sh':

    ```
   ```
"""

import deeplabcut as dlc
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis import (
    superanimal_analyze_images,
)
from deeplabcut.modelzoo import build_weight_init
import pandas as pd
import argparse
import glob
import os
from shutil import copy2

def superanimal_naive_finetuning(config_path, superanimal_name, model_name, detector_name,
                                 detector_epochs, epochs, save_epochs, batch_size, shuffle
                                ):
    """
    Fine-tune base SuperAnimal model on MLB data with existing keypoint matching in config_path.
    """
    weight_init = build_weight_init(
        cfg=auxiliaryfunctions.read_config(config_path), 
        super_animal=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        with_decoder=True,
    )

    dlc.create_training_dataset(
        config_path,
        Shuffles=[0],
        net_type=f"top_down_{model_name}",
        detector_type=detector_name,
        engine=dlc.Engine.PYTORCH,
        userfeedback=False,
        weight_init=weight_init
    )

    dlc.train_network(
        config_path,
        detector_epochs=detector_epochs,
        epochs=epochs,
        save_epochs=save_epochs,
        batch_size=batch_size,
        displayiters=10,
        shuffle=shuffle,
    )

if __name__=="__main__":
    p = argparse.ArgumentParser(description='Script to run naive fine-tuning of SuperAnimal model with labeled data in DLC project')
    p.add_argument('config_path', nargs='?', type=str, help='path to DLC config.yaml. Default: /wd/lbshks/super_animal/superanimal-MLB23-2025-11-25/config_hpc.yaml', default='/wd/lbshks/super_animal/superanimal-MLB23-2025-11-25/config_hpc.yaml')
    p.add_argument('superanimal_name', nargs='?', type=str, help='Name of Model Zoo base model to fine tune. Default: superanimal_quadruped', default='superanimal_quadruped')
    p.add_argument('model_name', nargs='?', type=str, help='Name of model architecture to use. Default: hrnet_w32', default='hrnet_w32')
    p.add_argument('detector_name', nargs='?', type=str, help='Name of detector to use for top-down SuperAnimal models. Default: fasterrcnn_resnet50_fpn_v2', default='fasterrcnn_resnet50_fpn_v2')
    p.add_argument('-de', '--detector_epochs', type=str, help='Detector epochs to run during training. Default: 1', default=1)
    p.add_argument('-e', '--epochs', type=str, help='Number of epochs to run during training. Default: 50', default=50)
    p.add_argument('-se', '--save_epochs', type=str, help='Number of epochs to save. Default: 10', default=10)
    p.add_argument('-b', '--batch_size', type=str, help='Batch size to use during training. Default: 64', default=64)
    p.add_argument('-s', '--shuffle', type=str, help='Shuffle index to use during training. Default: 0', default=0)

    args = p.parse_args()

    superanimal_naive_finetuning(
        config_path=args.config_path,
        superanimal_name=args.superanimal_name,
        model_name=args.model_name,
        detector_name=args.detector_name,
        detector_epochs=args.detector_epochs,
        epochs=args.epochs,
        save_epochs=args.save_epochs,
        batch_size=args.batch_size,
        shuffle=args.shuffle
    )