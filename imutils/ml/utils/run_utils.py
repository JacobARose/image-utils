"""

imutils/ml/utils/run_utils.py


Created on: Saturday March 19th, 2022  
Created by: Jacob Alexander Rose  

"""

import hydra
import logging
import numpy as np
import os
import pytorch_lightning as pl
import random
import torch
from typing import List



def seed_fix(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def configure_callbacks(config) -> List[pl.Callback]:
    callbacks: List[pl.Callback] = []
    for k, cb_conf in config.callbacks.items():
        if "_target_" in cb_conf:
            logging.info(f"Instantiating callback <{cb_conf._target_}>")
#             if k == "image_prediction_logger":
#                 callbacks.append(hydra.utils.instantiate(cb_conf, datamodule=datamodule))
#             else:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


# def default_callbacks(cfg: DictConfig) -> List[pl.Callback]:
#     callbacks: List[pl.Callback] = []

#     if "lr_monitor" in cfg.logging:
#         hydra.utils.log.info(f"Adding callback <LearningRateMonitor>")
#         callbacks.append(
#             LearningRateMonitor(
#                 logging_interval=cfg.logging.lr_monitor.logging_interval,
#                 log_momentum=cfg.logging.lr_monitor.log_momentum,
#             )
#         )

#     if "early_stopping" in cfg.train:
#         hydra.utils.log.info(f"Adding callback <EarlyStopping>")
#         callbacks.append(
#             EarlyStopping(
#                 monitor=cfg.train.monitor_metric,
#                 mode=cfg.train.monitor_metric_mode,
#                 patience=cfg.train.early_stopping.patience,
#                 verbose=cfg.train.early_stopping.verbose,
#             )
#         )

#     if "model_checkpoints" in cfg.train.model_checkpoints:
#         hydra.utils.log.info(f"Adding callback <ModelCheckpoint>")
#         callbacks.append(
#             ModelCheckpoint(
#                 monitor=cfg.train.monitor_metric,
#                 mode=cfg.train.monitor_metric_mode,
#                 save_top_k=cfg.train.model_checkpoints.save_top_k,
#                 verbose=cfg.train.model_checkpoints.verbose,
#             )
#         )

#     callbacks.append(
#         TQDMProgressBar(
#             refresh_rate=cfg.logging.progress_bar_refresh_rate
#             )
#         )
#     return callbacks











def configure_loggers(config) -> List[pl.loggers.LightningLoggerBase]:
    logger: List[pl.loggers.LightningLoggerBase] = []
    for _, lg_conf in config["logger"].items():
        if "_target_" in lg_conf:
            logging.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
