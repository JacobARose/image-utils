"""

imutils/ml/run_main.py

python run_main.py train.pl_trainer.fast_dev_run=True

python run_main.py \
    +train.pl_trainer.limit_train_batches=1 \
    +train.pl_trainer.limit_val_batches=1 \
    +train.pl_trainer.limit_test_batches=1 \
    +train.pl_trainer.max_epochs=2 \
    data.datamodule.batch_size=16 \
    train.callbacks.class_counts_stats=null \
    train.callbacks.image_stats_accumulator=null
    
python run_main.py \
    +train.pl_trainer.limit_train_batches=5 \
    +train.pl_trainer.limit_val_batches=5 \
    +train.pl_trainer.limit_test_batches=5 \
    +train.pl_trainer.max_epochs=3 \
    data.datamodule.batch_size=16 \
    train.callbacks.class_counts_stats=null \
    train.callbacks.image_stats_accumulator=null


python run_main.py \
    train.pl_trainer.gpus=2 \
    +train.pl_trainer.accelerator="ddp" \
    +train.pl_trainer.max_epochs=30 \
    data.datamodule.batch_size=64 \
    train.callbacks.class_counts_stats=null \
    train.callbacks.image_stats_accumulator=null

(6 AM 2022-03-21)
- Fixing accelerator, strategy, and devices arguments
python run_main.py \
    train.pl_trainer.devices=2 \
    train.pl_trainer.accelerator="gpu" \
    +train.pl_trainer.strategy="ddp" \
    +train.pl_trainer.max_epochs=30 \
    data.datamodule.batch_size=64 \
    train.callbacks.class_counts_stats=null \
    train.callbacks.image_stats_accumulator=null
    

(6:20 AM 2022-03-21)
- Scaling down to 1 gpu to debug weird miscalculation of num batches in an epoch
- Updated config to remove the need to nullify extraneous callbacks w/ every cmdline execution

python run_main.py \
    train.pl_trainer.devices=1 \
    train.pl_trainer.accelerator="gpu" \
    +train.pl_trainer.max_epochs=30 \
    data.datamodule.batch_size=64
    
############################

(8:30 AM 2022-03-21)
- Staying on 1 gpu to see if speed improves at all with higher batch size, since current GPU memory use is low (<10%)
    - [x] Increasing batch_size from 64->128
        - Appears to have taken ~5 mins to actually begin fitting.
        - p3 is being used a bit more than before, but still should not be enough to cause any conflicts here.
        - However, as training is still beginning (batches 0-30) - loading speed appears to be ~3 s/batch
- Removing lr_scheduler to reduce complexity while debugging other things & developing baselines,
    - [] I will have to test it out at some point down the line.
- [x] Adding more logging guarantees to validate # of trainable parameters.
    - This took me like an hour, considering the issues with hydra resolving the wrong data type.
    - Set to log a textual model summary in the model hook `on_fit_start`
        - This confirmed that the backbone is indeed failing to properly freeze, currently trainable_params=57 million=100%.
- [x] Adding simple pl profiler.

export CUDA_VISIBLE_DEVICES=6; python run_main.py \
    train.pl_trainer.devices=1 \
    train.pl_trainer.accelerator="gpu" \
    +train.pl_trainer.max_epochs=30 \
    +train.pl_trainer.profiler="simple" \
    optim.use_lr_scheduler=False \
    data.datamodule.batch_size=128


#############################


(9:40 AM AM 2022-03-21)
- Fixing the freezing/unfreezing issue, on 1-gpu until then. Next, back to debugging multi-gpu.
- Took ~5 mins to start training again w/ bsz=128.

export CUDA_VISIBLE_DEVICES=7; python run_main.py \
    train.pl_trainer.devices=1 \
    train.pl_trainer.accelerator="gpu" \
    +train.pl_trainer.max_epochs=30 \
    +train.pl_trainer.profiler="simple" \
    optim.use_lr_scheduler=False \
    data.datamodule.batch_size=128

(10:10 AM AM 2022-03-21)
- Model.on_train_start didnt get called for some reason in the 9:40 AM run, plus it was going really slow, so I restarted it & lowered the batch size back down to 64.
- Took ~5 mins to start training again w/ bsz=64.
- Observation: It appears my freezing function was accidentally using layer.enable_grad rather than the correct layer.requires_grad attribute

- [SOLVED] - at 11:00 AM 2022-03-21

export CUDA_VISIBLE_DEVICES=7; python run_main.py \
    train.pl_trainer.devices=1 \
    train.pl_trainer.accelerator="gpu" \
    +train.pl_trainer.max_epochs=30 \
    +train.pl_trainer.profiler="simple" \
    optim.use_lr_scheduler=False \
    data.datamodule.batch_size=64



(11:15 AM 2022-03-21)
- Setting random__flips=False to reduce compute
- Moving back to 2-gpus and strategy=DDP, batch_size=128
- freezing full backbone now


- Result (11:57 AM): [OOM ERROR] -- Crashed in the middle of Fossil Sync meeting
    - wandb shows process gpu memory increasing linearly until the crash, indicating a likely MEMORY LEAK.
    - Proposed solution: Refactor the steps to not return full image vectors at every batch.

export CUDA_VISIBLE_DEVICES='5,7'; python run_main.py \
    train.pl_trainer.devices=2 \
    train.pl_trainer.accelerator="gpu" \
    +train.pl_trainer.strategy="ddp" \
    +train.pl_trainer.max_epochs=30 \
    +train.pl_trainer.profiler="simple" \
    optim.use_lr_scheduler=False \
    data.datamodule.batch_size=128

################


(12:00 PM AM 2022-03-21)
Goals:
    - 1. Trying to identify and remove the possible memory leak
    - 2. Trying to debug why DDP isnt working on both GPUs
Actions:
    - 1. Refactoring step to not return x at every batch.
- Keeping to 2-gpus and strategy=DDP, batch_size=128
- freezing full backbone

- (12:15 PM) -- Removed x from step returns to see if the memory leak disappears
    - Also printing available GPU devices between model creations & DDP.


export CUDA_VISIBLE_DEVICES='0,4'; python run_main.py \
        train.pl_trainer.devices=2 \
        train.pl_trainer.accelerator="gpu" \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=64


#################################################


(1:10 PM AM 2022-03-21)
Goals:
    - 1. Trying to specify GPUs without CUDA_VISIBLE_DEVICES
    - 2. Trying to debug why DDP isnt working on both GPUs
Actions:
    - 1. specifying devices=[0,4]
    
- Had to enter this command first:
unset CUDA_VISIBLE_DEVICES

python run_main.py \
        train.pl_trainer.devices=[0,4] \
        train.pl_trainer.accelerator="gpu" \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=128

##################################


python run_main.py \
        train.pl_trainer.devices=[0,4] \
        train.pl_trainer.accelerator="gpu" \
        data.datamodule.num_workers=1 \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=128


python run_main.py \
        train.pl_trainer.devices=[0,4] \
        train.pl_trainer.accelerator="gpu" \
        data.datamodule.num_workers=0 \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=128
        
python run_main.py \
        train.pl_trainer.devices=[0,4] \
        train.pl_trainer.accelerator="gpu" \
        data.datamodule.num_workers=8 \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=128
        
################################


(~2:40 PM)

python run_main.py \
        train.pl_trainer.devices=[0,4] \
        train.pl_trainer.accelerator="gpu" \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=128 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf


####################################


(~3 PM)

(~2:40 PM)

python run_main.py \
        train.pl_trainer.devices=[0,4,5,6] \
        train.pl_trainer.accelerator="gpu" \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=32 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf

################

(~3:40 PM) -- Removed gradient_clip_val = 10,000,000 from default trainer cfg.

python run_main.py \
        train.pl_trainer.devices=[0,4,5,6] \
        train.pl_trainer.accelerator="gpu" \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=32 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf
        
        
python run_main.py \
        train.pl_trainer.devices=[0,4,5,6] \
        train.pl_trainer.accelerator="gpu" \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=24 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf


#############################


python run_main.py \
        train.pl_trainer.devices=[0,4,5,6] \
        train.pl_trainer.accelerator="gpu" \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=24 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf


###################

python run_main.py \
        train.pl_trainer.devices=[0,4] \
        train.pl_trainer.accelerator="gpu" \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.strategy="ddp" \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=32 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf \
        +overfit_batches=10


#############################################

(11:30 PM 2022-03-22)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=32 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf

#############################################

(12:40 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        +train.pl_trainer.limit_train_batches=0.5 \
        +train.pl_trainer.limit_val_batches=0.5 \
        +train.pl_trainer.limit_test_batches=0.5 \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=48 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(1:25 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        +train.pl_trainer.limit_train_batches=0.5 \
        +train.pl_trainer.limit_val_batches=0.5 \
        +train.pl_trainer.limit_test_batches=0.5 \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=64 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(2:45 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        +train.pl_trainer.limit_train_batches=0.5 \
        +train.pl_trainer.limit_val_batches=0.5 \
        +train.pl_trainer.limit_test_batches=0.5 \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=96 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf


###############################################

(3:15 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        +train.pl_trainer.limit_train_batches=0.5 \
        +train.pl_trainer.limit_val_batches=0.5 \
        +train.pl_trainer.limit_test_batches=0.5 \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=128 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(4:30 AM 2022-03-23)
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        +train.pl_trainer.limit_train_batches=0.5 \
        +train.pl_trainer.limit_val_batches=0.5 \
        +train.pl_trainer.limit_test_batches=0.5 \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=144 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(5:45 AM 2022-03-23) Experiment #7
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        +train.pl_trainer.limit_train_batches=0.5 \
        +train.pl_trainer.limit_val_batches=0.5 \
        +train.pl_trainer.limit_test_batches=0.5 \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="simple" \
        train.pl_trainer.accumulate_grad_batches=2 \
        optim.use_lr_scheduler=False \
        data.datamodule.batch_size=144 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(8:50 AM 2022-03-23) Experiment #8
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        +train.pl_trainer.limit_train_batches=0.1 \
        +train.pl_trainer.limit_val_batches=0 \
        +train.pl_trainer.limit_test_batches=0 \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="advanced" \
        train.pl_trainer.accumulate_grad_batches=2 \
        optim.use_lr_scheduler=False \
        optim.optimizer.lr=1e-2 \
        data.datamodule.batch_size=128 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf


###############################################

(11:20 AM 2022-03-23) Experiment #9
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        +train.pl_trainer.limit_train_batches=0.01 \
        +train.pl_trainer.limit_val_batches=0 \
        +train.pl_trainer.limit_test_batches=0 \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        train.freeze_backbone_up_to=-3 \
        +train.pl_trainer.profiler="advanced" \
        train.pl_trainer.accumulate_grad_batches=1 \
        optim.use_lr_scheduler=False \
        optim.optimizer.lr=1e-2 \
        data.datamodule.batch_size=128 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf

###############################################

(11:45 AM 2022-03-23) Experiment #10
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        +train.pl_trainer.limit_train_batches=0.05 \
        +train.pl_trainer.limit_val_batches=0 \
        +train.pl_trainer.limit_test_batches=0 \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        train.freeze_backbone_up_to=-3 \
        +train.pl_trainer.overfit_batches=0.05 \
        +train.pl_trainer.track_grad_norm=2 \
        +train.pl_trainer.profiler="advanced" \
        train.pl_trainer.accumulate_grad_batches=1 \
        optim.use_lr_scheduler=False \
        optim.optimizer.lr=1e-2 \
        data.datamodule.batch_size=128 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf
        
###############################################

(12:25 PM 2022-03-23) Experiment #11
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        train.freeze_backbone_up_to=-4 \
        +train.pl_trainer.profiler="advanced" \
        train.pl_trainer.accumulate_grad_batches=1 \
        optim.use_lr_scheduler=False \
        optim.optimizer.lr=0.5e-3 \
        data.datamodule.batch_size=128 \
        hp.resolution=448 \
        aug@data.datamodule.transform_cfg=medium_image_aug_conf


###############################################

(3:00 PM 2022-03-23) Experiment #12
export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
        train.pl_trainer.devices=2 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        train.freeze_backbone_up_to=-4 \
        +train.pl_trainer.profiler="advanced" \
        train.pl_trainer.accumulate_grad_batches=1 \
        optim.use_lr_scheduler=False \
        optim.optimizer.lr=1e-2 \
        data.datamodule.batch_size=128 \
        hp.preprocess_size=256 \
        hp.resolution=224 \
        aug@data.datamodule.transform_cfg=default_image_aug_conf


###############################################

(3:00 PM 2022-03-23) Experiment #13
export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
        train.pl_trainer.devices=4 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        train.freeze_backbone_up_to=-4 \
        +train.pl_trainer.profiler="advanced" \
        train.pl_trainer.accumulate_grad_batches=1 \
        optim.use_lr_scheduler=False \
        optim.optimizer.lr=2e-2 \
        data.datamodule.batch_size=128 \
        hp.preprocess_size=256 \
        hp.resolution=224 \
        aug@data.datamodule.transform_cfg=default_image_aug_conf



###############################################
- OOM crash after bsz=128 and accumulate_grad_batches=1, so trying 1/2 the bsz and 2x the accumulate
(5:00 PM 2022-03-23) Experiment #14
export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
        train.pl_trainer.devices=4 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        train.freeze_backbone_up_to=-4 \
        +train.pl_trainer.profiler="advanced" \
        train.pl_trainer.accumulate_grad_batches=2 \
        optim.use_lr_scheduler=False \
        optim.optimizer.lr=1e-3 \
        optim.optimizer.weight_decay=1e-6 \
        data.datamodule.batch_size=64 \
        hp.preprocess_size=256 \
        hp.resolution=224 \
        aug@data.datamodule.transform_cfg=default_image_aug_conf \
        model_cfg.backbone.name=resnext50_32x4d \
        model_cfg.backbone.pretrained=false \
        model_cfg.backbone.freeze_backbone=false \
        train.freeze_backbone_up_to=0 \
        train.freeze_backbone=false



###############################################
(4:20 AM 2022-03-24) Experiment #15
- Restored model checkpointing to code after #14's success.

export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
        train.pl_trainer.devices=4 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="advanced" \
        train.pl_trainer.accumulate_grad_batches=2 \
        optim.use_lr_scheduler=False \
        optim.optimizer.lr=5e-3 \
        optim.optimizer.weight_decay=1e-5 \
        data.datamodule.batch_size=64 \
        aug@data.datamodule.transform_cfg=default_image_aug_conf \
        hp.preprocess_size=256 \
        hp.resolution=224 \
        model_cfg.backbone.name=resnext50_32x4d \
        model_cfg.backbone.pretrained=false \
        model_cfg.backbone.freeze_backbone=false \
        train.freeze_backbone_up_to=0 \
        train.freeze_backbone=false


###############################################
(7:35 AM 2022-03-24) Experiment #15b
- Performance is slightly worse in Exp #15 compared to Experiment #14 within the first 2 epochs -> Attempting to reduce lr down from 5e-3 (Exp#15) -> 2e-3, vs. 1e-3 (Exp#14)

export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
        train.pl_trainer.devices=4 \
        data.datamodule.num_workers=4 \
        +train.pl_trainer.max_epochs=30 \
        +train.pl_trainer.profiler="advanced" \
        train.pl_trainer.accumulate_grad_batches=2 \
        optim.use_lr_scheduler=False \
        optim.optimizer.lr=2e-3 \
        optim.optimizer.weight_decay=1e-5 \
        data.datamodule.batch_size=64 \
        aug@data.datamodule.transform_cfg=default_image_aug_conf \
        hp.preprocess_size=256 \
        hp.resolution=224 \
        model_cfg.backbone.name=resnext50_32x4d \
        model_cfg.backbone.pretrained=false \
        model_cfg.backbone.freeze_backbone=false \
        train.freeze_backbone_up_to=0 \
        train.freeze_backbone=false





"""


import logging
import os
import shutil
from pathlib import Path
from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
import omegaconf
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    # ProgressBar,
    # TQDMProgressBar
)
from pytorch_lightning.loggers import WandbLogger
import torch

from imutils.ml.utils.common import load_envs
from imutils.ml.utils import template_utils
from imutils.ml.utils.experiment_utils import configure_callbacks, configure_loggers, configure_trainer
import imutils.ml.models.pl.classifier

torch.backends.cudnn.benchmark = True

logging = template_utils.get_logger(__file__)

# Set the cwd to the project root
# os.chdir(Path(__file__).parent.parent)

# Load environment variables
load_envs()





def train(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    from icecream import ic
    
    OmegaConf.register_new_resolver("int", int)
    
    if cfg.train.deterministic:
        pl.seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers = 0
        # cfg.data.datamodule.num_workers.train = 0
        # cfg.data.datamodule.num_workers.val = 0
        # cfg.data.datamodule.num_workers.test = 0

    # Hydra run directory
    try:
        hydra_dir = Path(HydraConfig.get().run.dir)
    except Exception as e:
        print(e)
        hydra_dir = os.getcwd()

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup()

    logging.warning("1. Before model, before trainer")
    ic(torch.cuda.device_count())
    ic(torch.cuda.is_available())
    ic(torch.cuda.current_device())    
    ic(torch.cuda.get_device_name(0))

    
    hydra.utils.log.info(f"Instantiating <{cfg.model_cfg._target_}>")
    # model: pl.LightningModule = hydra.utils.instantiate(cfg.model, cfg=cfg, _recursive_=False)
    model = imutils.ml.models.pl.classifier.LitClassifier(cfg=cfg, #model_cfg=cfg.model_cfg,
                                                          loss=cfg.model_cfg.loss)
    

    logging.warning("2. After model, before trainer")
    ic(torch.cuda.device_count())
    ic(torch.cuda.is_available())
    ic(torch.cuda.current_device())    
    ic(torch.cuda.get_device_name(0))
    
    wandb_logger = configure_loggers(cfg=cfg, model=model)
    # Instantiate the callbacks
    callbacks: List[pl.Callback] = configure_callbacks(cfg=cfg.train)



    from rich import print as pp
    
    
#     trainer_cfg = 
    
#     from pytorch_lightning.plugins import DDPPlugin
#     if cfg.train.pl_trainer.get("strategy").lower()=="ddp":
#         cfg.train.pl_trainer.strategy=DDPPlugin(find_unused_parameters=False)
    
    hydra.utils.log.info(f"Instantiating the Trainer")
    pp(OmegaConf.to_container(cfg.train.pl_trainer))
    
    trainer = configure_trainer(cfg,
                                callbacks=callbacks,
                                logger=wandb_logger)
    
    # trainer = pl.Trainer(
    #     default_root_dir=cfg.run_output_dir, #hydra.run.dir,
    #     logger=wandb_logger,
    #     # strategy=strategy,
    #     callbacks=callbacks,
    #     deterministic=cfg.train.deterministic,
    #     val_check_interval=cfg.logging.val_check_interval,
    #     # log_every_n_steps=10,
    #     #auto_select_gpus=True,
    #     # benchmark=True,
    #     # accelerator=None,  # 'dp', "ddp" if args.gpus > 1 else None,
    #     #plugins=[DDPPlugin(find_unused_parameters=True)],
    #     **cfg.train.pl_trainer,
    # )
    
    # trainer = configure_trainer(cfg,
    #                             callbacks=callbacks,
    #                             logger=wandb_logger,
    #                             **kwargs
    

    logging.warning("3. After model, after trainer, before fit")
    ic(torch.cuda.current_device())

    ic(os.environ.get("WORLD_SIZE"))
    ic(os.environ.get("NODE_RANK"))
    
    # num_samples = len(datamodule.train_dataset)
    num_classes = cfg.model_cfg.head.num_classes
    batch_size = datamodule.batch_size #["train"]

    hydra.utils.log.info("Starting training with {} classes and batches of {} images".format(
        num_classes,
        batch_size))

    trainer.fit(model=model, datamodule=datamodule)

    print(f"Skipping testing for now, must run predict on unlabeled test set")
    # hydra.utils.log.info(f"Starting testing!")
    # trainer.test(model=model, datamodule=datamodule)

    shutil.copytree(".hydra", Path(wandb_logger.experiment.dir) / "hydra")

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


# dotenv.load_dotenv(override=True)

# @hydra.main(config_path="configs/", config_name="multi-gpu")
@hydra.main(config_path="conf", config_name="base_conf")
def main(cfg: omegaconf.DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    # from lightning_hydra_classifiers.train_multi_gpu import train
    # from lightning_hydra_classifiers.utils import template_utils
    
    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    # template_utils.extras(cfg)
    omegaconf.OmegaConf.set_struct(cfg, False)

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        template_utils.print_config(cfg, resolve=True)

    return train(cfg)
        
        
        
        

# @hydra.main(config_path="conf", config_name="base_conf")
# def main(cfg: omegaconf.DictConfig):
#     run(cfg)


if __name__ == "__main__":
    main()