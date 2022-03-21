"""

lightning_hydra_classifiers/callbacks/image_stats_callbacks.py


Pytorch Lightning Callbacks for analyzing image statistics

Created on: Wednesday September 8th, 2021
Author: Jacob A Rose


"""


# from time import time
import time
from tqdm import tqdm
from torch import nn
import torch
from pathlib import Path
import os
from typing import *
import pytorch_lightning as pl
# from pytorch_lightning.utilities.distributed import rank_zero_only
from imutils.ml.callbacks.wandb_callbacks import get_wandb_logger
from imutils.ml.utils.template_utils import get_logger
logger = get_logger(name=__name__)


__all__ = ["ImageStatsAccumulatorCallback", "ImageStatsAccumulator", "PersistentModule"]


class PersistentModule(nn.Module):
    """Base nn.Module class containing custom state management with the following methods:
        ::self.cache
        ::self.load_from_cache
    """
    
#     @rank_zero_only
    def cache(self):
        if not isinstance(self.cache_dir, (str, Path)):
            return
#         self.cache_path = os.path.join(self.cache_dir, self.name + ".pth")
        torch.save(self.state_dict(), self.cache_path)
        logger.info(f"Updated Image Stats cache located at: {self.cache_path}")

#     @rank_zero_only
    def load_from_cache(self):
        if not isinstance(self.cache_dir, (str, Path)):
            logger.warning(f"No cache detected. No-op.")
            return
#         self.cache_path = os.path.join(self.cache_dir, self.name + ".pth")
        if os.path.isfile(self.cache_path):            
            self.load_state_dict(torch.load(self.cache_path))
            logger.info(f"Loaded Image Stats from cache located at: {self.cache_path}")
        else:
            logger.warning(f"No cache detected. No-op.")

    def state_dict(self):
        return self.__dict__
    
    def load_state_dict(self, state_dict: Dict):
        for k, v in state_dict.items():
            if k in self.__dict__:
                self.__dict__[k] = v
        
    def finalize(self, **kwargs):
        raise NotImplementedError
        
        
        

class ImageStatsAccumulator(PersistentModule):
    """
    Calculates a dataset-wide set of image statistics.
    Currently:
        - per-channel pixel mean
        - per-channel pixel std
        - per-channel pixel min
        - per-channel pixel max
    
    """
    
    def __init__(self, name: str="image_stats", cache_dir: str=None, clear_cache: bool=True):
        self.pixel_count = torch.zeros(1)
        self.image_count = torch.zeros(1)
        self.sum_of_pixels = torch.zeros(3)
        self.sum_of_square_pixels = torch.zeros(3)
        self.resolution = torch.zeros(1)
#         self.channels = torch.Tensor(3)
        
        self.global_max_pixel = torch.Tensor([-float("inf")]*3)
        self.global_min_pixel = torch.Tensor([float("inf")]*3)
        
        self.name = name
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(self.cache_dir, self.name + ".pth")
#         self.clear_cache = clear_cache
        if os.path.isfile(self.cache_path):
            if clear_cache:
                logger.warning(f"Removing prior cache prior to stats update.")
                os.remove(self.cache_path)
            else:
                self.load_from_cache()
    
    
    def update_step(self, batch):
        images = batch[0]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        if self.resolution == 0:
            self.resolution = h
        assert self.resolution == h == w
        
        batch_max = torch.amax(images, dim=(0,2,3))
        batch_min = torch.amin(images, dim=(0,2,3))
        
        self.global_max_pixel = torch.amax(torch.stack([self.global_max_pixel, batch_max]), dim=0)
        self.global_min_pixel = torch.amin(torch.stack([self.global_min_pixel, batch_min]), dim=0)
        
        
        self.sum_of_pixels        += torch.sum(images,    dim=[0, 2, 3])
        self.sum_of_square_pixels += torch.sum(images**2, dim=[0, 2, 3])
        self.pixel_count          += nb_pixels
        self.image_count          += b
        
        
    def update(self, loader: torch.utils.data.DataLoader):
        for batch in tqdm(loader):
            self.update_step(batch)
        global_mean, global_std, global_min, global_max = self.compute()
        logger.info(f"[FINISHED] Calculating full dataset stats from a collection of {self.image_count} images at resolution {self.resolution}")
        logger.info(f"Global channel-wise mean = {global_mean}")
        logger.info(f"Global channel-wise std = {global_std}")
        logger.info(f"Global channel-wise pixel min = {global_min}")
        logger.info(f"Global channel-wise pixel max = {global_max}")

        return global_mean, global_std, global_min, global_max

    
    def compute(self):        
        global_mean = self.sum_of_pixels / self.pixel_count
        global_var = (self.sum_of_square_pixels / self.pixel_count) - global_mean**2
        global_std = torch.sqrt(global_var)
        return global_mean, global_std, self.global_min_pixel, self.global_max_pixel
    
    
    # TODO Move the state dict stuff to a structured config dataclass
    def state_dict(self):
        global_mean, global_std, _, _ = self.compute()
        return {"global_mean": global_mean,
                "global_std": global_std,
                "pixel_count": self.pixel_count,
                "image_count": self.image_count,
                "sum_of_pixels": self.sum_of_pixels,
                "sum_of_square_pixels": self.sum_of_square_pixels,
                "resolution": self.resolution,
                "global_max_pixel": self.global_max_pixel,
                "global_min_pixel": self.global_min_pixel}
    
    def load_state_dict(self, state_dict):
        self.pixel_count = state_dict["pixel_count"]
        self.image_count = state_dict["image_count"]
        self.sum_of_pixels = state_dict["sum_of_pixels"]
        self.sum_of_square_pixels = state_dict["sum_of_square_pixels"]
        self.resolution = state_dict["resolution"]
        self.global_max_pixel = state_dict["global_max_pixel"]
        self.global_min_pixel = state_dict["global_min_pixel"]
        
        global_mean, global_std, _, _ = self.compute()
        assert torch.all(global_mean == state_dict["global_mean"])
        assert torch.all(global_std == state_dict["global_std"])    
    

    def __repr__(self):
        return "ImageStatsAccumulator:\n" + str('\n'.join([f"{k}: {v}" for k,v in self.state_dict().items()]))
    
    
    
    
    

class ImageStatsAccumulatorCallback(pl.Callback):
    
    # TODO: Consider calculating per-class image statistics and logging as a bar plot
    
    def __init__(self,
                 name: str="image_stats",
                 cache_dir: str=None,
                 clear_cache: bool=True):
        super().__init__()
        
        self.accumulator = ImageStatsAccumulator(name=name,
                                                 cache_dir=cache_dir,
                                                 clear_cache=clear_cache)
        
    
    def on_pretrain_routine_start(self, trainer, pl_module):
        self._start_time = time.time()
#         dataloader = trainer.test_dataloader()
        dataloader = trainer.datamodule.train_dataloader()
        self.accumulator.update(dataloader)

        
    def on_pretrain_routine_end(self, trainer, pl_module):
        logs = {}
        
        summary_stats = self.accumulator.compute()
        logger.info(str(self.accumulator))
        logger.info("Summary Stats:\n" + f"Global Mean: {summary_stats[0]}, Global Std: {summary_stats[1]}")
        trainer.datamodule.update_stats(mean=summary_stats[0], std=summary_stats[1])
        total_time = time.time() - self._start_time
        
        logger = get_wandb_logger(trainer)
        if isinstance(logger, pl.loggers.wandb.WandbLogger):
            logger.experiment.summary["mean"] = summary_stats[0]
            logger.experiment.summary["std"] = summary_stats[1]
            
            
            
            
#         trainer.logger.log_metrics(logs, step=trainer.current_epoch)

        

    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
    #     if (batch_idx + 1) % trainer.log_every_n_steps == 0:
    #         x, y = batch[:1]
            #TBD: Add logic for multiple loggers
            # logger = trainer.logger
            # logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            # logger.experiment.add_histogram("target", y, global_step=trainer.global_step)
            
            
            
            
        

#     def on_train_epoch_end(self, trainer, pl_module) -> None:
#         logs = {}
#         epoch_time = time.time() - self._start_time
#         trainer.logger.log_metrics(logs, step=trainer.current_epoch)

#         if self._verbose:
#             rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
#             rank_zero_info(f"Average Peak memory: {peak_memory:.2f} MB")
#             rank_zero_info(f"Average Free memory: {free_memory:.2f} MB")