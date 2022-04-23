


# import logging
# import os
# import shutil
# from pathlib import Path
# from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
from icecream import ic
# import omegaconf
import os
from omegaconf import DictConfig, OmegaConf
from rich import print as pp
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from imutils.ml.utils.common import load_envs
from imutils.ml.utils import template_utils
logging = template_utils.get_logger(__file__)

# Set the cwd to the project root
# os.chdir(Path(__file__).parent.parent)

# Load environment variables
load_envs()



def init_cfg(cfg: DictConfig):
	
	if cfg.train.deterministic:
		pl.seed_everything(cfg.train.random_seed)
		
	if cfg.train.pl_trainer.devices==1:
		cfg.train.pl_trainer.strategy=None
		
		

	if cfg.train.pl_trainer.fast_dev_run:
		hydra.utils.log.info(
			f"Debug mode <{cfg.train.pl_trainer.fast_dev_run}>. "
			f"Forcing debugger friendly configuration!"
		)
		# Debuggers don't like GPUs nor multiprocessing
		if cfg.train.callbacks.get('watch_model_with_wandb') is not None:
			del cfg.train.callbacks.watch_model_with_wandb
		if cfg.train.callbacks.get('uploadcheckpointsasartifact') is not None:
			del cfg.train.callbacks.uploadcheckpointsasartifact
		if cfg.train.callbacks.get('model_checkpoint') is not None:
			del cfg.train.callbacks.model_checkpoint
		# cfg.train.pl_trainer.gpus = 0
		cfg.data.datamodule.num_workers = 0

	cfg.run_output_dir = os.path.abspath(cfg.run_output_dir)
	
	return cfg




def run_pretrain(cfg: DictConfig) -> None:
	"""
	Generic pretrain loop

	:param cfg: run configuration, defined by Hydra in /conf
	"""
	
	import os
	
	cfg = init_cfg(cfg)

	hydra_dir = os.path.abspath(os.getcwd())
	print(f"Using hydra_dir: {hydra_dir}")
	# hydra.utils.log.info(f"Before pretrain.lr_tuner value of lr: {cfg.optim.optimizer.lr}")	
	if cfg.execution_list.auto_lr_tune:
		hydra.utils.log.info(f"Executing pretrain stage: auto_lr_tune")
		from imutils.ml import pretrain
		cfg = pretrain.lr_tuner.run(cfg=cfg)
						   # datamodule=datamodule)
						   # model=model)

	else:
		hydra.utils.log.info(f"Skipping pretrain stage: auto_lr_tune")
	hydra.utils.log.info(f"Proceeding with cfg.optim.lr_scheduler.warmup_start_lr={cfg.optim.lr_scheduler.warmup_start_lr}")
	return cfg



def train(cfg: DictConfig) -> None:
	"""
	Generic train loop

	:param cfg: run configuration, defined by Hydra in /conf
	"""
	
	from imutils.ml.utils.experiment_utils import (configure_model,
												   configure_callbacks,
												   configure_loggers,
												   configure_trainer,
												   configure_loss_func)
	import imutils.ml.models.pl.classifier
	
	cfg = init_cfg(cfg)
	hydra_dir = os.path.abspath(os.getcwd())	
	print(f"Using hydra_dir: {hydra_dir}")
	
	if cfg.execution_list.model_fit:
		
		hydra.utils.log.info(f"Executing train stage: model_fit")
		hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
		datamodule: pl.LightningDataModule = hydra.utils.instantiate(
			cfg.data.datamodule, _recursive_=False)
		datamodule.setup()
		
		loss_func = configure_loss_func(cfg, targets=datamodule.train_dataset.df.y)
		# model: pl.LightningModule = hydra.utils.instantiate(cfg.model, cfg=cfg, _recursive_=False)
		
		# model = imutils.ml.models.pl.classifier.LitClassifier(cfg=cfg,
		# 													  loss_func=loss_func)
		
		model = configure_model(cfg=cfg,
								loss_func=loss_func)
		
		ic(model.lr, cfg.hp.lr, cfg.optim.optimizer.lr)
		
		loggers = configure_loggers(cfg=cfg, model=model)

		callbacks: List[pl.Callback] = configure_callbacks(cfg=cfg.train)	
		hydra.utils.log.info(f"Instantiating the Trainer")
		pp(OmegaConf.to_container(cfg.train.pl_trainer))
		trainer = configure_trainer(cfg,
									callbacks=callbacks,
									logger=loggers)

		num_samples_train = len(datamodule.train_dataset)
		num_samples_val = len(datamodule.val_dataset)
		# num_classes = cfg.model_cfg.head.num_classes
		num_classes = cfg.hp.num_classes
		batch_size = datamodule.batch_size #["train"]
		rank_zero_only(hydra.utils.log.info)(
			"Starting training: \n" \
			+ f"train_size: {num_samples_train} images" + "\n" \
			+ f"val_size: {num_samples_val} images" + "\n" \
			+ f"num_classes: {num_classes}" + "\n" \
			+ f"batch_size: {batch_size}" + "\n"
		)
		trainer.fit(model=model, datamodule=datamodule)


	template_utils.finish(
		config=cfg,
		logger=loggers,
		callbacks=callbacks)

	# if args.train:
	#	 trainer.fit(model, dm)
	# if args.test:
	#	 ckpt_path = (
	#		 checkpoint_callback.best_model_path if args.train else cfg.model.checkpoint
	#	 )
	#	 trainer.test(model=model, datamodule=dm)

	# print(f"Skipping testing for now, must run predict on unlabeled test set")
	# hydra.utils.log.info(f"Starting testing!")
	# trainer.test(model=model, datamodule=datamodule)
	# print(f"SUCCESS: Made it to the other side of experiment finished.", f"device:{torch.cuda.current_device()}")





# @hydra.main(config_path="configs/", config_name="multi-gpu")
@hydra.main(config_path="conf", config_name="base_conf")
def main(cfg: DictConfig):

	# Imports should be nested inside @hydra.main to optimize tab completion
	# Read more here: https://github.com/facebookresearch/hydra/issues/934
	

	# template_utils.extras(cfg)
	template_utils.initialize_config(cfg)
	
	print(f"CUBLAS_WORKSPACE_CONFIG = {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")

	# Pretty print config using Rich library
	if cfg.get("print_config_only"):
		template_utils.print_config(cfg, resolve=True)
		return
	if cfg.execution_list.get("print_cfg"):
		template_utils.print_config(cfg, resolve=True)

	cfg = run_pretrain(cfg=cfg)
	
	return train(cfg)

# def initialize_config(cfg: DictConfig):
# 	OmegaConf.set_struct(cfg, False)
# 	OmegaConf.register_new_resolver("int", int)
# 	return cfg
		

# @hydra.main(config_path="conf", config_name="base_conf")
# def main(cfg: omegaconf.DictConfig):
#	 run(cfg)


if __name__ == "__main__":
	main()
	




########################################################
########################################################
"""
Sunday April 10th, 2022

Experiment #22 (Previous efforts for class balanced cross entropy on hold)

- Introducing our in house datasets with Extant_Leaves
- Implemented actual custom dataset-specific means & stds, instead of using imagenet stats for everything
- Refactored image augmentations to use albumentations instead of torchvision.transforms
- Reimplemented render_image_predictions model hook for sanity checking the augmentations.

Observations:
* Initial run went fast but wasnt learning much, when I realized I hadnt adapted the config to use different values for num_classes when switching datasets yet. This resulted in accidentally building a model with output size 15,501 for an extant leaves dataset with 94 families as unique classes.
	* (~4:30 AM Monday April 11th, 2022) -- Now Im rerunning after fixing the model output size.


-- 2-gpus

export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
'core.name="[DEV] - extant_leaves_family_10_512 - Experiment #22 (2022-04-10)"' \
+train.pl_trainer.limit_train_batches=8 \
+train.pl_trainer.limit_val_batches=8 \
train.pl_trainer.log_every_n_steps=2 \
execution_list.auto_lr_tune=false \
hp.warmup_epochs=5 \
hp.batch_size=24 \
hp.lr=1e-2 \
aug@data.datamodule.transform_cfg=medium_image_aug_conf \
hp.preprocess_size=512 \
hp.resolution=448 \
data/datamodule@data=extant_leaves_family_10_512_datamodule \
optim.optimizer.weight_decay=5e-6 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=false \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=2 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1

######################

-- 4-gpus
* (Launched ~4:30 AM Monday April 11th, 2022)

export CUDA_VISIBLE_DEVICES=4,5,6,7; python run_main.py \
'core.name="[EXP] - extant_leaves_family_10_512 - Experiment #22 (2022-04-10)"' \
train.pl_trainer.log_every_n_steps=10 \
execution_list.auto_lr_tune=true \
hp.warmup_epochs=5 \
hp.batch_size=24 \
hp.lr=1e-2 \
aug@data.datamodule.transform_cfg=medium_image_aug_conf \
hp.preprocess_size=512 \
hp.resolution=448 \
data/datamodule@data=extant_leaves_family_10_512_datamodule \
optim.optimizer.weight_decay=5e-6 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=2


######################

-- 4-gpus
* (Launched ~4:30 AM Monday April 11th, 2022)

export CUDA_VISIBLE_DEVICES=4,5,6,7; python run_main.py \
'core.name="[EXP] - extant_leaves_family_10_512 - Experiment #22 (2022-04-11)"' \
train.pl_trainer.log_every_n_steps=10 \
execution_list.auto_lr_tune=false \
hp.warmup_epochs=7 \
hp.batch_size=24 \
hp.lr=2e-2 \
aug@data.datamodule.transform_cfg=medium_image_aug_conf \
hp.preprocess_size=512 \
hp.resolution=448 \
data/datamodule@data=extant_leaves_family_10_512_datamodule \
optim.optimizer.weight_decay=5e-6 \
optim.lr_scheduler.warmup_start_lr=1e-4 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=75 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=4



######################
## Experiment #23
-- RGB Extant_Leaves_family_10_512

* (Launched ~4:55 PM Friday April 15th, 2022)
* (Finished ~8:35 PM Friday April 15th, 2022)

export CUDA_VISIBLE_DEVICES=4,5,6,7; python run_main.py \
'core.name="[EXP] - extant_leaves_family_10_512 - Experiment #23 (2022-04-15)"' \
train.pl_trainer.log_every_n_steps=5 \
execution_list.auto_lr_tune=true \
hp.warmup_epochs=7 \
hp.batch_size=24 \
hp.lr=1e-2 \
hp.preprocess_size=512 \
hp.resolution=448 \
data/datamodule@data=extant_leaves_family_10_512_datamodule \
optim.optimizer.weight_decay=5e-6 \
optim.lr_scheduler.warmup_start_lr=1e-4 \
model_cfg.backbone.name=hrnet_w32 \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=75 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=4


######################
## Experiment #24
-- Grayscale Extant_Leaves_family_10_512

* (Launched 8:40 PM Friday April 15th, 2022)
* (Finished ~12:15 AM Saturday April 16th, 2022)

export CUDA_VISIBLE_DEVICES=4,5,6,7; python run_main.py \
'core.name="[EXP] - extant_leaves_family_10_512 - Experiment #24 (2022-04-15)"' \
execution_list.auto_lr_tune=true \
experiments=grayscale_3-channel \
hp.warmup_epochs=7 \
hp.batch_size=24 \
hp.lr=1e-2 \
hp.preprocess_size=512 \
hp.resolution=448 \
data/datamodule@data=extant_leaves_family_10_512_datamodule \
optim.optimizer.weight_decay=5e-6 \
optim.lr_scheduler.warmup_start_lr=1e-4 \
model_cfg.backbone.name=hrnet_w32 \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=75 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=4

#############################

######################
## Experiment #25
-- Grayscale herbarium2022-res_512_datamodule
-- Using AdamW instead of Adam
    - Changing weight_decay from 5e-6 -> 1e-2
-- Created new transform_cfg: medium_image_aug & renamed previous one from auto_image_aug->light_image_aug
    - Increased shift & scale limits of shift_scale_rotate from 0.05->0.15
    - Increased the probability p of shift_scale_rotate from 0.5->0.6
    - Increased probability p of random_brightness_contrast from 0.5->0.6
    - Added horizontal flip w/ p=0.5
    - Added vertical flip w/ p=0.5


    ### Attempt #1:
    * (Launched 4:27 AM Saturday April 16th, 2022)
    * I'm seeing NaN train Loss within epoch 0, trying to lower weight_decay from 1e-2 -> 1e-4

    ### Attempt #2
    * (Launched 4:48 AM Saturday April 16th, 2022)

* (Launched 4:27 AM Saturday April 16th, 2022)
* (Finished xx:xx AM Saturday April 16th, 2022)




export CUDA_VISIBLE_DEVICES=3,4,5,7; python run_main.py \
'core.name="[EXP] - Grayscale herbarium2022-res_512_datamodule - Experiment #25 (2022-04-16)"' \
execution_list.auto_lr_tune=true \
+data/datamodule@data=herbarium2022-res_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug \
experiments=grayscale_3-channel \
hp.warmup_epochs=7 \
hp.batch_size=24 \
hp.lr=1e-2 \
hp.preprocess_size=512 \
hp.resolution=448 \
optim/optimizer=AdamW \
optim.optimizer.weight_decay=1e-4 \
optim.lr_scheduler.warmup_start_lr=1e-4 \
model_cfg.backbone.name=hrnet_w32 \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=75 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=4


##################
#################

######################
## Experiment #26
-- Loading from best pretrained weights -- Experiment #21 -- RGB Herbarium2022 512
-- Finetuning on Grayscale Extant_Leaves_family_10_512


* (Launched 9:54 AM Monday April 18th, 2022)
* (Finished 1:11 PM Monday April 18th, 2022)




export CUDA_VISIBLE_DEVICES=0,1,2,6; python run_main.py \
'core.name="[EXP] - Herbarium2022-pretrained-weights -> Finetuning on Extant_Leaves_family_10_512 - Experiment #26 (2022-04-18)"' \
execution_list.auto_lr_tune=true \
data/datamodule@data=extant_leaves_family_10_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug \
experiments=grayscale_3-channel \
hp.warmup_epochs=7 \
hp.batch_size=32 \
hp.lr=5e-3 \
hp.preprocess_size=512 \
hp.resolution=448 \
optim/optimizer=AdamW \
optim.optimizer.weight_decay=1e-5 \
optim.lr_scheduler.warmup_start_lr=1e-5 \
model_cfg.backbone.name=resnext50_32x4d \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=75 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=4 \
hp.load_from_checkpoint=true \
'ckpt_path="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/2022/herbarium2022/hydra_experiments/2022-04-01/21-13-25/ckpts/epoch=22-val_loss=1.316-val_macro_F1=0.720/model_weights.ckpt"'





######################
## Experiment #27
-- Loading from best pretrained weights -- Experiment #21 -- RGB Herbarium2022 512
-- Finetuning on Grayscale Extant_Leaves_family_10_512

Goal: Manually adjusting some hyperparameters from Experiment #26 in order to reduce overfitting


Compared to Experiment #26
-- Increasing weight decay from 1e-5 to 1e-4
-- Increasing max_epochs from 75 to 100
-- Increasing warmup_epochs from 7 to 10
-- Increasing early_stopping.patience from 15 to 20
-- Decreasing early_stopping.min_delta from 0.05 to 0.02

* (Launched 1:25 PM Monday April 18th, 2022)
* (Finished 5:20 PM Monday April 18th, 2022) -- Crashed in the 43rd epoch due to a mysterious CUDA OOM error




export CUDA_VISIBLE_DEVICES=0,1,2,6; python run_main.py \
'core.name="[EXP] - Herbarium2022-pretrained-weights -> Finetuning on Extant_Leaves_family_10_512 - Experiment #27 (2022-04-18)"' \
execution_list.auto_lr_tune=false \
data/datamodule@data=extant_leaves_family_10_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug \
experiments=grayscale_3-channel \
hp.warmup_epochs=10 \
hp.batch_size=32 \
hp.lr=5e-3 \
hp.preprocess_size=512 \
hp.resolution=448 \
optim/optimizer=AdamW \
optim.optimizer.weight_decay=1e-4 \
optim.lr_scheduler.warmup_start_lr=2.7673e-6 \
model_cfg.backbone.name=resnext50_32x4d \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=100 \
train.callbacks.early_stopping.patience=20 \
train.callbacks.early_stopping.min_delta=0.02 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=4 \
hp.load_from_checkpoint=true \
'ckpt_path="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/2022/herbarium2022/hydra_experiments/2022-04-01/21-13-25/ckpts/epoch=22-val_loss=1.316-val_macro_F1=0.720/model_weights.ckpt"'


##################################################
##################################################

######################
## Experiment #28
-- Loading from best pretrained weights -- Experiment #21 -- RGB Herbarium2022 512
-- Finetuning on Grayscale Extant_Leaves_family_10_512

Goal: Adding label smoothing CE Loss in order to reduce overfitting observed in Experiments #26 & #27


Compared to Experiment #27
-- Added Label Smoothing = 0.1
-- Decreasing weight decay from 1e-4 to 5e-5


* (Launched 5:40 PM Monday April 18th, 2022)
* (Finished x:xx PM Monday April 18th, 2022) -- Crashed in the 43rd epoch due to a mysterious CUDA OOM error




export CUDA_VISIBLE_DEVICES=0,1,2,6; python run_main.py \
'core.name="[EXP] - Herbarium2022-pretrained-weights -> Finetuning on Extant_Leaves_family_10_512 w ls-ce loss - Experiment #28 (2022-04-18)"' \
execution_list.auto_lr_tune=false \
data/datamodule@data=extant_leaves_family_10_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug \
experiments=grayscale_3-channel \
hp.warmup_epochs=10 \
hp.batch_size=32 \
hp.lr=5e-3 \
hp.preprocess_size=512 \
hp.resolution=448 \
optim/optimizer=AdamW \
optim.optimizer.weight_decay=5e-5 \
optim.lr_scheduler.warmup_start_lr=2.7673e-6 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg/loss=label-smoothing_ce-loss \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=100 \
train.callbacks.early_stopping.patience=20 \
train.callbacks.early_stopping.min_delta=0.02 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=4 \
hp.load_from_checkpoint=true \
'ckpt_path="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/2022/herbarium2022/hydra_experiments/2022-04-01/21-13-25/ckpts/epoch=22-val_loss=1.316-val_macro_F1=0.720/model_weights.ckpt"'


########################################################
########################################################



######################
## Experiment #29
-- Grayscale Herbarium2022 512 -- Using m=272 **family** labels

Goal: Train model until saturation on family labels, the lowest cardinality level in the hierarchy. Then, downstream, finetune it on more fine-grained labels.



-- Changed primary class label level from scientificName (M=15501) to family (M=272)
-- Increased Label Smoothing from 0.1 to 0.2
-- Increasing hp.lr from 5e-3 to 7e-3


* (Launched 6:30 AM Tuesday April 19th, 2022)
(Stalled overnight for some reason, we have a zoom seder in 15 mins so I couldnt really diagnose)

* (Relaunched 5:45 PM Tuesday April 19th, 2022)


* (Relaunched 12:05 PM Tuesday April 21st, 2022)


* (Finished x:xx PM Monday April 19th, 2022) -- Crashed in the 43rd epoch due to a mysterious CUDA OOM error




export CUDA_VISIBLE_DEVICES=0,1,2,6; python run_main.py \
'core.name="[EXP] - Grayscale Herbarium2022 pretrain on family - Experiment #29 (2022-04-21)"' \
execution_list.auto_lr_tune=false \
optim.lr_scheduler.warmup_start_lr=1.746e-3 \
data/datamodule@data=herbarium2022-res_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug \
'data.datamodule.label_col="family"' \
experiments=grayscale_3-channel \
hp.warmup_epochs=10 \
hp.batch_size=32 \
hp.lr=7e-3 \
hp.preprocess_size=512 \
hp.resolution=448 \
optim/optimizer=AdamW \
optim.optimizer.weight_decay=5e-5 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg/loss=label-smoothing_ce-loss \
model_cfg.loss.label_smoothing=0.1 \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=100 \
train.callbacks.early_stopping.patience=20 \
train.callbacks.early_stopping.min_delta=0.02 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=4


########################################################
########################################################










"""


"""

## Temporary Experiment Launch command references.


Experiment #21:

- Extending warmup epochs from 3 to 5
- Setting hp.preprocess_size=None

Observations:

 - Note: It looks like the lr_tuner did not result in an identical warmup_start_lr between Exp #20 and #21, indicating a potential replicability problem.
	 #20: warmup_start_lr = 1.208e-5 (About 437% the magnitude of #21)
	 #21: warmup_start_lr = 2.767e-6 (Only about 23% the magnitude of #20)


export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
'core.name="Experiment #21 (2022-04-01)"' \
hp.warmup_epochs=5 \
optim.optimizer.weight_decay=5e-6 \
hp.batch_size=24 \
hp.lr=2e-3 \
data/datamodule@data=herbarium2022-res_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug_conf \
hp.preprocess_size=None \
hp.resolution=448 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=2




###################
Experiment #22:

- Class-Balanced Cross Entropy Loss --> Key aspect of this experiment is the introduction of our first attempt at changing the loss from basic cross-entropy to use the following:
	* model_cfg/loss=class-balanced-ce-loss

- Lots of refactoring to be compatible with running on serrep2

Observations:



export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_main.py \
'core.name="Experiment #22 (2022-04-03)"' \
'model_cfg/loss=class-balanced-ce-loss' \
model_cfg.loss.beta=0.5 \
hp.warmup_epochs=5 \
hp.batch_size=24 \
hp.lr=0.1 \
hp.preprocess_size=None \
hp.resolution=448 \
data/datamodule@data=herbarium2022-res_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug_conf \
optim.optimizer.weight_decay=5e-6 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=4 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=2



#####################

Experiment #22 [Dev] 
-- 2-gpus

export CUDA_VISIBLE_DEVICES=6,7; python run_main.py \
'core.name="[DEV] Experiment #22 (2022-04-03)"' \
+train.pl_trainer.overfit_batches=10 \
execution_list.auto_lr_tune=false \
'model_cfg/loss=class-balanced-ce-loss' \
model_cfg.loss.beta=0.5 \
hp.warmup_epochs=5 \
hp.batch_size=24 \
hp.lr=0.1 \
hp.preprocess_size=None \
hp.resolution=448 \
data/datamodule@data=herbarium2022-res_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug_conf \
optim.optimizer.weight_decay=5e-6 \
model_cfg.backbone.name=resnext50_32x4d \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=2 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=4 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1


#####

Experiment #22 [Dev] 
-- 1-gpu

export CUDA_VISIBLE_DEVICES=7; python run_main.py \
'core.name="[DEV][1-GPU] Experiment #22 (2022-04-03)"' \
+train.pl_trainer.limit_train_batches=8 \
+train.pl_trainer.limit_val_batches=8 \
train.pl_trainer.log_every_n_steps=2 \
execution_list.auto_lr_tune=false \
'model_cfg/loss=ce-loss' \
hp.batch_size=16 \
hp.lr=2e-3 \
hp.preprocess_size=None \
hp.resolution=448 \
data/datamodule@data=herbarium2022-res_512_datamodule \
aug@data.datamodule.transform_cfg=medium_image_aug_conf \
optim.optimizer.weight_decay=1e-9 \
optim.use_lr_scheduler=false \
model_cfg.backbone.name=resnet18 \
model_cfg.backbone.pretrained=true \
hp.freeze_backbone_up_to=0 \
hp.freeze_backbone=false \
train.pl_trainer.devices=1 \
train.pl_trainer.accelerator="gpu" \
data.datamodule.num_workers=0 \
train.pl_trainer.max_epochs=50 \
+train.pl_trainer.profiler="simple" \
train.pl_trainer.accumulate_grad_batches=1 \
+train.pl_trainer.detect_anomaly=true \
data.datamodule.transform_cfg.skip_augmentations=true



train.pl_trainer.precision=32 \
'model_cfg/loss=class-balanced-ce-loss' \


"""





















