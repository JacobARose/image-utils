


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


@rank_zero_only
def ddp_print(*args, rich: bool=True):
	if rich:
		pp(*args)
		return
	print(*args)
	



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
			ddp_print(f"Removing cfg.train.callbacks.watch_model_with_wandb")
			del cfg.train.callbacks.watch_model_with_wandb
		if cfg.train.callbacks.get('uploadcheckpointsasartifact') is not None:
			ddp_print(f"Removing cfg.train.callbacks.uploadcheckpointsasartifact")
			del cfg.train.callbacks.uploadcheckpointsasartifact
		if cfg.train.callbacks.get('model_checkpoint') is not None:
			ddp_print(f"Removing cfg.train.callbacks.model_checkpoint")
			del cfg.train.callbacks.model_checkpoint
		# cfg.train.pl_trainer.gpus = 0
		# cfg.data.datamodule.num_workers = 0

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
	ddp_print(f"Using hydra_dir: {hydra_dir}")
	# hydra.utils.log.info(f"Before pretrain.lr_tuner value of lr: {cfg.optim.optimizer.lr}")	
	if cfg.execution_list.auto_lr_tune:
		ddp_print(f"Executing pretrain stage: auto_lr_tune")
		from imutils.ml import pretrain
		cfg = pretrain.lr_tuner.run(cfg=cfg)
						   # datamodule=datamodule)
						   # model=model)

	else:
		ddp_print(f"[SKIPPING PRETRAIN STAGE]: auto_lr_tune")
		# hydra.utils.log.info(f"Skipping pretrain stage: auto_lr_tune")
	ddp_print(f"[PROCEEDING] with value cfg.optim.lr_scheduler.warmup_start_lr={cfg.optim.lr_scheduler.warmup_start_lr}")
	# hydra.utils.log.info(f"Proceeding with cfg.optim.lr_scheduler.warmup_start_lr={cfg.optim.lr_scheduler.warmup_start_lr}")
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
	ddp_print(f"Using hydra_dir: {hydra_dir}")
	
	if cfg.execution_list.model_fit:
		
		# hydra.utils.log.info(f"Executing train stage: model_fit")
		# hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
		ddp_print(f"Executing train stage: model_fit")
		ddp_print(f"Instantiating {cfg.data.datamodule._target_}")
		datamodule: pl.LightningDataModule = hydra.utils.instantiate(
			cfg.data.datamodule, _recursive_=False)
		datamodule.setup()
		
		loss_func = configure_loss_func(cfg, targets=datamodule.train_dataset.df.y)
		
		model = configure_model(cfg=cfg,
								loss_func=loss_func)
		ddp_print(f"{cfg.optim.lr_scheduler.warmup_start_lr=}")
		ddp_print(f"{model.lr=}, {cfg.hp.lr=}, {cfg.optim.optimizer.lr}")
		
		loggers = configure_loggers(cfg=cfg, model=model)

		callbacks: List[pl.Callback] = configure_callbacks(cfg=cfg.train)	
		ddp_print(f"Instantiating the Trainer")
		ddp_print(OmegaConf.to_container(cfg.train.pl_trainer, resolve=True))
		trainer = configure_trainer(cfg,
									callbacks=callbacks,
									logger=loggers)

		num_samples_train = len(datamodule.train_dataset)
		num_samples_val = len(datamodule.val_dataset)
		# num_classes = cfg.model_cfg.head.num_classes
		num_classes = cfg.hp.num_classes
		batch_size = datamodule.batch_size #["train"]
		ddp_print(
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


# @hydra.main(config_path="configs/", config_name="multi-gpu")
@hydra.main(config_path="conf", config_name="base_conf")
def main(cfg: DictConfig):

	# Imports should be nested inside @hydra.main to optimize tab completion
	# Read more here: https://github.com/facebookresearch/hydra/issues/934
	

	# template_utils.extras(cfg)
	template_utils.initialize_config(cfg)
	
	ddp_print(f"CUBLAS_WORKSPACE_CONFIG = {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")

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


















