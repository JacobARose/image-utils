"""

image-utils/imutils/ml/utils/template_utils.py



"""

import logging
import warnings
from typing import List, Sequence, Union
import os
import pytorch_lightning as pl
import rich
import wandb
from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import WandbLogger, LightningLoggerBase

# pl.loggers.LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree
import hydra

def get_logger(name=__name__, level=logging.INFO):
	"""Initializes python logger."""

	logger = logging.getLogger(name)
	
	handler = logging.StreamHandler()
	formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	
	logger.setLevel(level)

	# this ensures all logging levels get marked with the rank zero decorator
	# otherwise logs would get multiplied for each GPU process in multi-GPU setup
	for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
		setattr(logger, level, rank_zero_only(getattr(logger, level)))

	return logger


log = get_logger()

def get_wandb_logger(loggers: Union[LightningLoggerBase, List[LightningLoggerBase]]):
	"""
	Looks through either a single or list of loggers, if it finds a WandbLogger, it returns only that. If it doesn't, it returns only None.
	"""
	if isinstance(loggers, Sequence):
		for l in loggers:
			if isinstance(l, WandbLogger):
				return l
	elif isinstance(loggers, WandbLogger):
		return loggers
	else:
		return None




def extras(config: DictConfig) -> None:
	"""A couple of optional utilities, controlled by main config file.
		- disabling warnings
		- easier access to debug mode
		- forcing debug friendly configuration
		- forcing multi-gpu friendly configuration
	Args:
		config (DictConfig): [description]
	"""

	# enable adding new keys to config
	OmegaConf.set_struct(config, False)

	# disable python warnings if <config.disable_warnings=True>
	if config.get("disable_warnings"):
		log.info(f"Disabling python warnings! <config.disable_warnings=True>")
		warnings.filterwarnings("ignore")

	# set <config.trainer.fast_dev_run=True> if <config.debug=True>
	if config.get("debug"):
		log.info("Running in debug mode! <config.debug=True>")
		config.train.pl_trainer.fast_dev_run = True

	# force debugger friendly configuration if <config.trainer.fast_dev_run=True>
	if config.train.pl_trainer.get("fast_dev_run"):
		log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
		# Debuggers don't like GPUs or multiprocessing
		if config.train.pl_trainer.get("gpus"):
			config.train.pl_trainer.gpus = 0
		if config.data.datamodule.get("num_workers"):
			config.data.datamodule.num_workers = 0

	# force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
	if config.train.pl_trainer.get("accelerator") in ["ddp", "ddp_spawn", "dp", "ddp2"]:
		log.info("Forcing ddp friendly configuration! <config.trainer.accelerator=ddp>")
		# ddp doesn't like num_workers>0 or pin_memory=True
		if config.data.datamodule.get("num_workers"):
#			 config.datamodule.num_workers = 0
			gpus = config.train.pl_trainer.get("gpus", 0)
			gpus = gpus or config.train.pl_trainer.get("devices", 0)
			print(f"GPUs: {gpus}")
			if isinstance(gpus, list):
				config.data.datamodule.num_workers = 4 * len(gpus)
			elif isinstance(gpus, int):
				config.data.datamodule.num_workers = 4
		if config.data.datamodule.get("pin_memory"):
			config.data.datamodule.pin_memory = False

	# disable adding new keys to config
	OmegaConf.set_struct(config, True)
	
def initialize_config(cfg: DictConfig):
	OmegaConf.set_struct(cfg, False)
	OmegaConf.register_new_resolver("int", int)
	return cfg



@rank_zero_only
def print_config(
	config: DictConfig,
	fields: Sequence[str] = (
        "pretrain",
		"train",
		"model_cfg",
		"optim",
		"data",
		"callbacks",
		"logging",
		"hp",
		"seed"
	),
	resolve: bool = True,
	file: str = None
) -> None:
	"""Prints content of DictConfig using Rich library and its tree structure.

	Args:
		config (DictConfig): Config.
		fields (Sequence[str], optional): Determines which main fields from config will be printed
		and in what order.
		resolve (bool, optional): Whether to resolve reference fields of DictConfig.
	"""

	style = "dim"
	tree = Tree(f":gear: CONFIG", style=style, guide_style=style)

	for field in fields:
		branch = tree.add(field, style=style, guide_style=style)

		config_section = config.get(field)
		branch_content = str(config_section)
		if isinstance(config_section, DictConfig):
			branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

		branch.add(Syntax(branch_content, "yaml"))

	rich.print(tree, file=file)


def empty(*args, **kwargs):
	pass


@rank_zero_only
def log_hyperparameters(
	config: DictConfig,
	model: pl.LightningModule,
	datamodule: pl.LightningDataModule,
	trainer: pl.Trainer,
	callbacks: List[pl.Callback],
	logger: List[pl.loggers.LightningLoggerBase],
) -> None:
	"""This method controls which parameters from Hydra config are saved by Lightning loggers.

	Additionaly saves:
		- sizes of train, val, test dataset
		- number of trainable model parameters
	"""

	hparams = {}

	# choose which parts of hydra config will be saved to loggers
	hparams["train"] = config["train"]
	hparams["model"] = config["model"]
	hparams["data"] = config["data"]
	if "optim" in config:
		hparams["optim"] = config["optim"]
	if "callbacks" in config:
		hparams["callbacks"] = config["callbacks"]
	if "hp" in config:
		hparams["hp"] = config["hp"]

	# save sizes of each dataset
	# (requires calling `datamodule.setup()` first to initialize datasets)
	datamodule.setup()
	if hasattr(datamodule, "data_train") and datamodule.data_train:
		hparams["datamodule/train_size"] = len(datamodule.data_train)
	if hasattr(datamodule, "data_val") and datamodule.data_val:
		hparams["datamodule/val_size"] = len(datamodule.data_val)
	if hasattr(datamodule, "data_test") and datamodule.data_test:
		hparams["datamodule/test_size"] = len(datamodule.data_test)

	# save number of model parameters
	hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
	hparams["model/params_trainable"] = sum(
		p.numel() for p in model.parameters() if p.requires_grad
	)
	hparams["model/params_not_trainable"] = sum(
		p.numel() for p in model.parameters() if not p.requires_grad
	)
	
	for logger in trainer.logger:
		if hasattr(logger, "log_hyperparams"):
			trainer.logger.log_hyperparams(hparams)
		if hasattr(logger, 'save_dir'):
			os.makedirs(logger.save_dir, exist_ok=True)
	
	if 'wandb' in config.logger.keys():
		wandb.watch(model.classifier, criterion=model.criterion, log='all')
		log.info(f'Logging classifier gradients to wandb using wandb.watch()')

	# disable logging any more hyperparameters for all loggers
	# (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
	trainer.logger.log_hyperparams = empty


from torch import distributed as dist
	
# @rank_zero_only
def init(config: DictConfig):
#	 wandb.init(..., reinit=dist.is_available() and dist.is_initialized() and dist.get_rank() == 0)
	if config.trainer.accelerator == "ddp":
#	 if wandb.run is None:
		config.wandb.init.reinit = True# = dist.is_available() and dist.is_initialized() and (dist.get_rank() == 0)
#		 print(f"dist.is_available()={dist.is_available()}")
#		 print(f"dist.is_initialized()={dist.is_initialized()}")
#		 print(f"dist.get_rank() == 0)={(dist.get_rank() == 0)}")
	
		logging.info(f"Since trainer.accelerator={config.trainer.accelerator}, setting config.wandb.init.reinit to: {config.wandb.init.reinit}")
		
#			 logging.info(f"torch.distributed.get_rank() = {dist.get_rank()}")
		
		local_rank = os.environ.get("LOCAL_RANK", 0)
		print(f'local_rank={local_rank}')
		if str(local_rank)=="0":
			hydra.utils.instantiate(config.wandb.init)
			print(f'Just may have successfully initiated wandb')
		else:
			print(f'Skipping wandb.init b/c local_rank={local_rank}')
		
	
# def init_ddp_connection(self, *args, **kwargs):
#	 super().init_ddp_connection(*args, **kwargs)

#	 if torch.distributed.get_rank() == 0:
#		 import wandb
#		 wandb.run = None


def find_ckpt_callback(callbacks: List[pl.Callback]=None):
	for cb in callbacks:
		if isinstance(cb, pl.callbacks.ModelCheckpoint):
			return cb
	return None



def write_ckpts_info2yaml(cfg: DictConfig,
						  callbacks: List[pl.Callback]=None):
	# save top weights paths to yaml
	if "model_checkpoint" in cfg.train.callbacks:
		ckpts_info_path = os.path.join(
			cfg.train.callbacks.model_checkpoint.dirpath, "best_ckpts_meta.yaml"
		)
		cb = find_ckpt_callback(callbacks)
		if isinstance(cb, pl.callbacks.ModelCheckpoint):
			os.makedirs(os.path.dirname(ckpts_info_path), exist_ok=True)
			cb.to_yaml(filepath=ckpts_info_path)
		if os.path.isfile(ckpts_info_path):
			print(f"Find a listing of the path(s) best ckpt(s) located at: {ckpts_info_path} ")
		else:
			print(f"Warning: problem encountered trying to save ckpts info to yaml file at path: {ckpts_info_path}. Continuing without error.")





from pathlib import Path
import shutil

@rank_zero_only
def finish(
	config: DictConfig,
	logger: List[pl.loggers.LightningLoggerBase],
	# model: pl.LightningModule=None,
	# datamodule: pl.LightningDataModule=None,
	# trainer: pl.Trainer=None,
	callbacks: List[pl.Callback]=None
) -> None:
	"""Perform some simple reporting hooks for the end of common workflows + Makes sure everything closed properly."""
	
	write_ckpts_info2yaml(cfg=config,
						  callbacks=callbacks)
	
	wandb_logger = get_wandb_logger(logger)
	if wandb_logger is None:
		return
	
	try:
		log_dir = Path(config.run_output_dir, "hydra_cfg")
		shutil.copytree(".hydra", log_dir)
	except Exception as e:
		print(f"WARNING: Error while copying .hydra files to hydra_cfg directory. Attempting to finish up anyway.")
		print(f"The exception: {e}")

	
	if isinstance(wandb_logger, WandbLogger):
		wandb.finish()
