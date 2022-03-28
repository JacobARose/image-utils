"""

image-utils/imutils/ml/pretrain/lr_tuner.py

originally located at:
	lightning_hydra_classifiers/scripts/pretrain/lr_tuner.py


Created on: Friday Sept 3rd, 2021
Moved on: Monday March 28th, 2022
Author: Jacob A Rose


"""


import pytorch_lightning as pl
import argparse
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from rich import print as pp
from typing import *


from imutils.ml.utils.etl_utils import ETL
from imutils.ml.utils import template_utils
############################################
logger = template_utils.get_logger(name=__name__)


__all__ = ["run"]



from dataclasses import dataclass, asdict
import hydra
from pytorch_lightning.utilities import rank_zero_only

@dataclass
class LRTunerConfig:

	min_lr: float = 1e-08
	max_lr: float = 1.0
	num_training: int = 100
	mode: str = 'exponential'
	early_stop_threshold: float = 4.0

DEFAULT_CONFIG = OmegaConf.structured(LRTunerConfig())


# @rank_zero_only
def run(cfg: DictConfig,
		datamodule=None,
		model=None) -> DictConfig:
	"""
	WIP implementation that encloses trainer instantiation within the pretrain stage to ensure proper release of memory prior to multi-GPU training.
	
	- TODO: Figure out heuristic for scaling lr found using lr_tuner from 1-GPU to multi-GPUs
	
	
	ToDO: Consider how to override the optimizer scheduler temporarily.
	
	"""
	import pytorch_lightning as pl
	import imutils
	import imutils.ml.models.pl.classifier
	use_lr_scheduler = cfg.optim.use_lr_scheduler
	trainer_args = argparse.Namespace(**cfg.pretrain.lr_tuner.pl_trainer)
	if trainer_args.auto_lr_find is False:
		hydra.utils.log.info("Skipping pretrain.lr_tuner stage b/c trainer_args.auto_lr_find==False")
		return cfg

	import torch

	hparams_path = cfg.pretrain.lr_tuner.get("hparams_path", "lr_tuner_hparams.yaml")
	if os.path.isfile(hparams_path):
		best_hparams = ETL.config_from_yaml(hparams_path)
		best_lr = best_hparams['lr']
		
		hydra.utils.log.info(f"device:{torch.cuda.current_device()}")
		hydra.utils.log.info(f"Loaded best lr_tuner hparams: {best_hparams} from file: {hparams_path}")
		cfg = _update_cfg_lr(cfg,
							 lr=best_lr,
							 use_lr_scheduler=use_lr_scheduler)
		return cfg

	if datamodule is None:
		hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
		datamodule: pl.LightningDataModule = hydra.utils.instantiate(
			cfg.data.datamodule, _recursive_=False
		)
		datamodule.setup()

	if model is None:
		hydra.utils.log.info(f"Instantiating <{cfg.model_cfg._target_}> prior to pretrain.lr_tuner")
		# model: pl.LightningModule = hydra.utils.instantiate(cfg.model, cfg=cfg, _recursive_=False)
		model = imutils.ml.models.pl.classifier.LitClassifier(cfg=cfg, #model_cfg=cfg.model_cfg,
															  loss=cfg.model_cfg.loss)

	hydra.utils.log.info("INITIATING STAGE: pretrain.lr_finder using cfg settings:")
	template_utils.print_config(cfg, fields=["pretrain", "model_cfg", "optim"])
	trainer = pl.Trainer.from_argparse_args(args=trainer_args)
	
	## lr_tuner execution
	tuner_args = OmegaConf.to_container(cfg.pretrain.lr_tuner.tuner, resolve=True)

	lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule, **tuner_args)
	new_lr = lr_finder.suggestion()
	
	if cfg.train.pl_trainer.devices > 1:
		num_gpus = cfg.train.pl_trainer.devices
		hydra.utils.log.info(f"NOTICE: Since devices={num_gpus}, scaling new lr by the same amount [EXPERIMENTAL].")
		hydra.utils.log.info(f"BEFORE: {new_lr}")
		new_lr = new_lr*num_gpus
		hydra.utils.log.info(f"AFTER: {new_lr}")
	
	cfg = _update_cfg_lr(cfg,
						 lr=new_lr,
						 use_lr_scheduler=use_lr_scheduler)
	best_hparams = {"lr":new_lr}
	os.makedirs(os.path.dirname(hparams_path), exist_ok=True)
	ETL.config2yaml(best_hparams, hparams_path)
	
	hydra.utils.log.info(f"device:{torch.cuda.current_device()}")
	hydra.utils.log.info(f"Saved best lr_tuner hparams: {best_hparams} to file: {hparams_path}")
	print(f"SUCCESS: Updated config:")
	print("\n".join(["=" * 80, f"Learning rate updated to {new_lr}","=" * 80]))
	template_utils.print_config(cfg, fields=["pretrain", "model_cfg", "optim"])
	

	return cfg


def _update_cfg_lr(cfg,
				   lr: float,
				   use_lr_scheduler: bool=False):
	cfg.optim.use_lr_scheduler = use_lr_scheduler
	# if cfg.optim.lr_scheduler is not None:
	if cfg.optim.use_lr_scheduler:
		# cfg.optim.use_lr_scheduler = True
		cfg.optim.lr_scheduler.warmup_start_lr = lr
		hydra.utils.log.info("[MODE 1] UPDATING WARMUP STARTING LEARNING RATE W/ SCHEDULE")
		hydra.utils.log.info("RESULTS of lr_tuner stage -- Using an lr_scheduler -->\n" \
							 + f" Updated cfg.optim.lr_scheduler.warmup_start_lr={cfg.optim.lr_scheduler.warmup_start_lr}")
	else:
		cfg.hp.lr = lr
		cfg.optim.optimizer.lr = lr
		# model.lr = new_lr
		hydra.utils.log.info("[MODE 2] UPDATING BASE LEARNING RATE W/O SCHEDULE")
		hydra.utils.log.info("RESULTS of lr_tuner stage -- Not using an lr_scheduler -->\n" \
							 + f" Updated cfg.optim.optimizer.lr={cfg.optim.optimizer.lr}")

	return cfg

# def run_lr_tuner(trainer: pl.Trainer,
#				  model: pl.LightningModule,
#				  datamodule: pl.LightningDataModule,
#				  config: argparse.Namespace,
#				  results_dir: str,
#				  group: str=None,
#				  run: Optional=None):
#				  # strict_resume: bool=False):
# #				  run=None):
#	 """
#	 Learning rate tuner
	
#	 Adapted and refactored from "lightning-hydra-classifiers/lightning_hydra_classifiers/scripts/train_basic.py"
#	 """
#	 tuner_config = OmegaConf.create(DEFAULT_CONFIG)

#	 try:
#		 cfg = asdict(config)
#	 except TypeError:
#		 cfg = OmegaConf.to_container(config, resolve=True)
#	 finally:
#		 cfg = dict(config)
	
#	 if "pretrain" in cfg:
#		 logger.info(f"Proceeding with overrides merged with default parameters")
# #		 logger.info(f"overrides: {config.lr_tuner}")
# #		 logger.info(f"defaults: {tuner_config}")
#		 tuner_config = OmegaConf.merge(DEFAULT_CONFIG, cfg["pretrain"])
#	 else:
#		 for k, v in DEFAULT_CONFIG.items():
#			 if k in cfg:
#				 tuner_config.update({k:config[k]})

#		 config.pretrain = OmegaConf.create(tuner_config)


#	 results_path = str(Path(results_dir, "results.csv"))
#	 hparams_path = str(Path(results_dir, "hparams.yaml"))
#	 if os.path.isfile(hparams_path):
		
#		 best_hparams = ETL.config_from_yaml(hparams_path)
#		 results = None
#		 if os.path.isfile(results_path):
#			 results = ETL.df_from_csv(results_path)
		
#		 best_lr = best_hparams['lr']
#		 if hasattr(model, "config"):
#			 model.config.lr = best_lr


#		 model.hparams.lr = best_lr
#		 config.model.lr = best_lr
# #		 config.model.optimizer.lr = model.config.lr
		
#		 assert config.model.lr == best_lr

#		 logger.info(f'[FOUND] Previously completed trial. Results located in file:\n`{results_path}`')
#		 logger.info(f'[LOADING] Previous results + avoiding repetition of tuning procedure.')
#		 logger.info(f'Proceeding with learning rate, lr = {config.model.lr:.3e}')
#		 logger.info('Model hparams =')
#		 pp(best_hparams)
#		 suggestion = {"lr": config.model.lr,
#					   "loss": None}
#		 return suggestion, results, config
	
#	 if run is None:
#		 run = wandb.init(job_type = "lr_tune",
#						  config=cfg,
#						  group=group,
#						  reinit=True)
#		 logger.info(f"[Initiating Stage] lr_tuner")
#		 lr_tuner = trainer.tuner.lr_find(model,
#										  datamodule,
#										  **cfg.get("pretrain", {}))
#		 lr_tuner_results = lr_tuner.results
#		 best_lr = lr_tuner.suggestion()
#		 suggestion = {"lr": best_lr,
#					   "loss":lr_tuner_results['loss'][lr_tuner._optimal_idx]}
		
#		 if hasattr(model, "config"):
#			 model.config.lr = suggestion['lr']
#		 model.hparams.lr = suggestion['lr']
#		 config.model.lr = model.hparams.lr
# #		 config.model.optimizer.lr = model.hparams.lr
#		 model.hparams.update(config.model)
#		 best_hparams = OmegaConf.create({"optimized_hparam_key": "lr",
#										  "lr":best_lr,
#										  "batch_size":config.data.batch_size,
#										  "image_size":config.data.image_size,
#										  "lr_tuner_config":config.pretrain}) #.lr_tuner})
#		 results_dir = Path(results_path).parent
#		 os.makedirs(results_dir, exist_ok=True)
#		 ETL.config2yaml(best_hparams, hparams_path)
#		 logger.info(f'Saved best lr value (along w/ batch_size, image_size) to file located at: {str(hparams_path)}') # {str(results_dir / "hparams.yaml")}')
#		 logger.info(f'File contents expected to contain: \n{dict(best_hparams)}')	

#		 fig = lr_tuner.plot(suggest=True)
#		 plot_fname = 'lr_tuner_results_loss-vs-lr.png'
#		 plot_path = results_dir / plot_fname

#		 plt.suptitle(f"Suggested lr={best_lr:.4e} |\n| Searched {lr_tuner.num_training} lr values $\in$ [{lr_tuner.lr_min},{lr_tuner.lr_max}] |\n| bsz = {config.data.batch_size}", fontsize='small')
#		 plt.tight_layout()
#		 plt.subplots_adjust(bottom=0.2, top=0.8)
#		 plt.savefig(plot_path)
		
#		 if run is not None:
#	 #		 run.summary['lr_finder/plot'] = wandb.Image(fig, caption=plot_fname)
#			 run.log({'lr_finder/plot': wandb.Image(str(plot_path), caption=plot_fname)})
#			 run.log({'lr_finder/best/loss': suggestion["loss"]})
#			 run.log({'lr_finder/best/lr': suggestion["lr"]})
#			 run.log({'lr_finder/batch_size': config.data.batch_size})
#			 run.log({'image_size': config.data.image_size})
#			 run.log({'lr_finder/hparams': OmegaConf.to_container(best_hparams)})
			
#			 df = pd.DataFrame(lr_tuner.results)
#			 try:
#				 ETL.df2csv(df, results_path)
#				 run.log({"lr_finder/results":wandb.Table(dataframe=df)})
#			 except Exception as e:
#				 if hasattr(df, "to_pandas"):
#					 run.log({"lr_finder/results":wandb.Table(dataframe=df.to_pandas())})

#	 logger.info(f'FINISHED: `run_lr_tuner(config)`')
#	 logger.info(f'Proceeding with:\n')
#	 logger.info(f'Learning rate = {config.model.lr:.3e}')
#	 logger.info(f'Batch size = {config.data.batch_size}')
	
#	 return suggestion, lr_tuner_results, config



