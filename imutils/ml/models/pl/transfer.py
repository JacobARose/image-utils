"""

lightning_hydra_classifiers/models/transfer.py

Author: Jacob A Rose
Created: Thursday Oct 14th, 2021

Plugins:
    LayerSelectPlugin
    LayerFreezeLightningPlugin
    LightningMetricsPlugin
Super classes:
    BaseLightningModule
Usable classes:
    LightningClassifier


--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
The previous script with this name (from Wednesday June 23rd, 2021) has been deprecated, and might be found in:
lightning_hydra_classifiers/models/_DEPRECATED/transfer.py
----------------------------------------------------------
"""

import logging
import os
from rich import print as pp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
import glob
import hydra
from collections import OrderedDict
from typing import *

# from lightning_hydra_classifiers.utils.model_utils import count_parameters, collect_results
# from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics
# from lightning_hydra_classifiers.models.backbones.backbone import build_model
# # from lightning_hydra_classifiers.experiments.multitask import schedulers
# from lightning_hydra_classifiers.experiments.multitask.schedulers import configure_schedulers


logger = logging.Logger(__name__)
logger.setLevel('INFO')

from tqdm.auto import tqdm, trange
from prettytable import PrettyTable



__all__ = ["BaseLightningModule", "LightningClassifier"]


BN_TYPE = (torch.nn.modules.batchnorm._BatchNorm,)

def is_bn(layer: nn.Module) -> bool:
    """ Return True if layer's type is one of the batch norms."""
    return isinstance(layer, BN_TYPE)

def grad_check(tensor: torch.Tensor) -> bool:
    """ Returns True if tensor.requires_grad==True, else False."""
    return tensor.requires_grad == True



class LayerSelectPlugin:
    """
    LayerSelectPlugin

    To be subclassed by main Pytorch Lightning Module
    
    Available methods:
    * classmethods:
        - count_parameters
    * instance methods:
        - get_batchnorm_modules
        - get_conv_modules
        - get_linear_modules
        - get_named_parameters
        - get_named_modules
        - get_trainable_parameters
        - get_nontrainable_parameters
        - count_trainable_batchnorm_layers

    
    
    """
    
    # TODO: Switch to using get_modules_by_type instead of get_named_modules
    @classmethod
    def get_batchnorm_modules(cls,
                              model: Optional=None):
#         model = model or self
#         return self.get_named_modules("bn", model=model)
        return cls.get_modules_by_type(include_type=BN_TYPE,
                                        model=model)
    
    def get_conv_modules(self,
                         model: Optional=None):
        model = model or self
        return self.get_named_modules("conv", model=model)
    
    def get_linear_modules(self,
                           model: Optional=None):
        model = model or self
        return ((n,m) for n,m in self.named_modules(model=model) if isinstance(m, nn.modules.Linear))
    
    
    def get_named_parameters(self,
                             filter_pattern: Optional[str]=None,
                             model: Optional=None,
                             trainable: bool=False,
                             nontrainable: bool=False):
        
#         if isinstance(model, (Sequence,Generator)):
#             named_params = ()
        model = model or self
        named_params = model.named_parameters()
        if isinstance(filter_pattern, str):
            named_params = ((n,p) for n,p in named_params if filter_pattern in n)
        if trainable and nontrainable:
            logger.warning('Returning all parameters regardless of the values of requires_grad.')
            return named_params
        if trainable:
            return ((n, p) for n, p in named_params if p.requires_grad)
        if nontrainable:
            return ((n, p) for n, p in named_params if not p.requires_grad)
        return named_params
    
    ## TODO: Create a filter_modules() helper method that takes a filter function to be re-used in each of the various get_modules_by* methods.

    def get_named_modules(self,
                          filter_pattern: Optional[str]=None,
                          model: Optional=None):
        model = model or self
        if isinstance(filter_pattern, str):
            return ((n,l) for n,l in model.named_modules() if filter_pattern in n)
        return model.named_modules()
    
    @classmethod
    def get_modules_by_type(cls,
                            include_type: Any=None,
                            model: Optional=None):
#         model = model or self
        if include_type is not None:
            return ((n,l) for n,l in model.named_modules() if isinstance(l, include_type))
        return model.named_modules()
    
    
            
    def get_trainable_parameters(self, 
                                 model: Optional=None,
                                 count_params: bool=False,
                                 count_layers: bool=False):
        model = model or self
        out = (p for _, p in self.get_named_parameters(model=model,
                                                       trainable=True))
        if count_params:
            out = sum((p.numel() for p in out))
        elif count_layers:
            out = len(list(out))
        return out
#         return (p for p in self.parameters() if p.requires_grad)

    def get_nontrainable_parameters(self,
                                    model: Optional=None,
                                    count_params: bool=False,
                                    count_layers: bool=False):
        model = model or self
        out = (p for _, p in self.get_named_parameters(model=model,
                                                       nontrainable=True))
        if count_params:
            out = sum((p.numel() for p in out))
        elif count_layers:
            out = len(list(out))
        return out
#         return (p for p in self.parameters() if not p.requires_grad)

    def count_trainable_batchnorm_layers(self,
                                         model: Optional=None) -> Tuple[Dict[str, int]]:
        model = model or self
        is_training = np.array([m.training for _, m in self.get_batchnorm_modules(model=model)])
        training = {"True":np.sum(is_training),
                    "False":np.sum(~is_training),
                    "Total": len(is_training)}
#         print(f"trainable batchnorm modules:{}")
#         print(f"nontrainable batchnorm modules:{np.sum(~is_training)}")


        modules = list(tuple(p.parameters()) for n, p in self.get_batchnorm_modules(model=model))
        is_training = np.array([(grad_check(m[0]) and grad_check(m[1])) for m in modules])

#         is_training = np.array([p.requires_grad for _, p in self.get_named_parameters("bn", model=model)])
        requires_grad = {"True":np.sum(is_training),
                         "False":np.sum(~is_training),
                         "Total": len(is_training)}
        if getattr(self, "_verbose", False):
            print("layer.training-> Use batch statistics")
            print("layer.training-> Use running statistics")
            print("is_training (layer.training==True):"); pp(training)
            print("requires_grad (layer.requires_grad==True):"); pp(requires_grad)
        
        return training, requires_grad

#         print(f"batchnorm params with requires_grad=True: :{np.sum(is_training)}")
#         print(f"batchnorm params with requires_grad=False:{np.sum(~is_training)}")
        
    @classmethod
    def count_parameters(cls, model, verbose: bool=True) -> PrettyTable:
        return count_parameters(model, verbose=verbose)


class LayerFreezeLightningPlugin:
    """
    LayerFreezeLightningPlugin
    
    To be subclassed by main Pytorch Lightning Module
    
    Available methods:
    * classmethods:
        - freeze_up_to
        - freeze
        - unfreeze
        - freeze_bn
        - set_bn_eval
    * instance methods:
        - freeze_backbone
        - unfreeze_backbone_top_layers
    """
    valid_strategies : Tuple[str] = ("feature_extractor",
                                     "feature_extractor_+_bn.eval()",
                                     "feature_extractor_+_except_bn")
    _verbose: bool = True
    def set_strategy(self,
                     feature_extractor_strategy: str=None,
                     finetuning_strategy: str=None):
        if feature_extractor_strategy == "feature_extractor":
            self.feature_extractor_strategy(freeze_bn=True,
                                            eval_bn=False)
        elif feature_extractor_strategy == "feature_extractor_+_bn.eval()":
            self.feature_extractor_strategy(freeze_bn=True,
                                            eval_bn=True)
        elif feature_extractor_strategy == "feature_extractor_+_except_bn":
            self.feature_extractor_strategy(freeze_bn=False,
                                            eval_bn=False)
        elif feature_extractor_strategy is None:
            print(f"Initializing model from scratch due to feature_extractor_strategy=None")
            self.freeze_bn = False
            self.eval_bn = False

            self.unfreeze(self.model)
        else:
            raise NotImplementedError(f"{feature_extractor_strategy} is not a valid feature_extractor_strategy. Please select from 1 of the following {len(self.valid_strategies)} strategies: {self.valid_strategies}")
        self.hparams.feature_extractor_strategy = feature_extractor_strategy
        if self._verbose: print(f"Set current model training strategy to: {self.hparams.feature_extractor_strategy}")
            
        
        if finetuning_strategy == "finetuning_unfreeze_layers_on_plateau":
            if "resnet" in self.hparams.backbone_name:
                self.finetuning_milestones = ["layer4", "layer3", "layer2", "layer1"]
            elif "efficient" in self.hparams.backbone_name:
                self.finetuning_milestones = ['blocks.6', 'blocks.5', 'blocks.4', 'blocks.3', 'blocks.2', 'blocks.1', 'blocks.0']
            else:
                self.finetuning_milestones = None
            if self._verbose: print(f"Set current model finetuning strategy to: {self.finetuning_milestones}")
            
#             self.feature_extractor_strategy(freeze_bn=False,
#                                             eval_bn=False)

        
        

    
    def feature_extractor_strategy(self,
                                   freeze_bn: bool=True,
                                   eval_bn: bool=False):
        """
        Defaults to PyTorch default, which is to freeze the gradients for batchnorm layers, but not necessarily apply eval() to them upon freezing, thus allowing the running mean & std to continue training on each incoming batch.
        
        Allows the option of both freezing and setting to eval mode all batch norm layers, thus fully removing the possibility of data leakage or accidental injection of noise.

        Arguments:
           freeze_bn: bool, default=True
               If True, set bn layers' attribute requires_grad to False.
           eval_bn: bool, default=False
               If True, apply layer.eval() to bn layers. If False, apply layer.train() to bn layers.
        
        """
        self.freeze_backbone(freeze_bn=freeze_bn)
        self.freeze_bn = freeze_bn
        self.eval_bn = eval_bn
        
        if not freeze_bn:
            self.unfreeze_bn(self.model,
                             unfreeze_bn=True)
#             self.unfreeze(self.model,
#                           filter_pattern="bn")
        if self.eval_bn:
            self.set_bn_eval(self.model)

            
#     def on_validation_model_eval(self) -> None:
#         """
#         Sets the model to eval during the val loop
#         """
#         self.eval()

    # def on_validation_model_train(self) -> None:
    #     """
    #     Sets the model to train during the val loop
    #     """
    #     self.train()
    #     self.set_strategy(feature_extractor_strategy=self.hparams.feature_extractor_strategy)

    
    @classmethod
    def freeze(cls,
               module,
               freeze_bn: bool=True,
               filter_pattern: Optional[str]=None):
        modules = list(module.named_modules())
        
        for n, m in modules:
#             if filter_pattern not in n:
            if isinstance(filter_pattern, str) and (filter_pattern not in n):
                continue
#             m.eval()
            for p_name, p in m.named_parameters():
                if isinstance(filter_pattern, str) and (filter_pattern not in n):
                    continue
#                 if freeze_bn or not isinstance(m, nn.BatchNorm2d):
                if freeze_bn or not is_bn(m):
                    p.requires_grad=False
            cls.freeze_bn(m, freeze_bn)      

            
    @classmethod
    def unfreeze(cls,
                 module,
                 unfreeze_bn: bool=True,
                 filter_pattern: Optional[str]=None):

#         out = (p for _, p in cls.get_named_parameters(model=module))
        
        if isinstance(module, (Generator, Sequence)):
            modules = module
        else:
            modules = list(module.named_modules())
        for n, m in modules:
            if isinstance(filter_pattern, str) and (filter_pattern not in n):
                continue
            if is_bn(m) and not unfreeze_bn:
                continue
            cls.unfreeze_bn(m, unfreeze_bn)
            for p_name, p in m.named_parameters():
                p.requires_grad=True
            m.train()

    @classmethod
    def freeze_bn(cls, module: nn.Module, freeze_bn: bool=True):
        for n, m in cls.get_batchnorm_modules(model=module):
            if freeze_bn:
                for p in m.parameters():
                    p.requires_grad = False
                    if cls._verbose: logger.debug(f"[freeze_bn][Layer={n}] Set requires_grad=False")

    @classmethod
    def unfreeze_bn(cls, module: nn.Module, unfreeze_bn: bool=True):
        for n, m in cls.get_batchnorm_modules(model=module):
            if unfreeze_bn:
                for p in m.parameters():
                    p.requires_grad = True
                    if cls._verbose: logger.debug(f"[unfreeze_bn][Layer={n}] Set requires_grad=True")

                    
    @classmethod
    def set_bn_eval(cls, module: nn.Module)->None:
        "Set bn layers in eval mode for all recursive children of `m`."
        for n, l in module.named_children():
#             if isinstance(l, nn.BatchNorm2d) and not next(l.parameters()).requires_grad:
            if is_bn(l) and not next(l.parameters()).requires_grad:
                l.eval()
                if cls._verbose: logger.debug(f"[set_bn_eval][Layer={n}] Called layer.eval()")

#                 continue
            cls.set_bn_eval(l)


    def freeze_backbone(self, freeze_bn: bool=True):
        
        self.freeze(self.model.backbone,
                    freeze_bn=freeze_bn)
        self.unfreeze(self.model.head)

        
        
    def unfreeze_backbone_top_layers(self,
                                     unfreeze_down_to: Union[str,int]=-1):
        if isinstance(unfreeze_down_to, str):
            layers = list(reversed(list(self.model.backbone.named_children())))
        if isinstance(unfreeze_down_to, int):
            if unfreeze_down_to == 0:
                print(f"Pass non-zero integer or str label name to unfreeze layers. Returning without change.")
                return
            layers = list(reversed(list(enumerate(self.model.backbone.children()))))
            if unfreeze_down_to < 0:
                unfreeze_down_to = len(layers) + unfreeze_down_to
        
        for layer_id, l in layers:
            self.unfreeze(l)
            if layer_id == unfreeze_down_to:
                break

#     @classmethod
#     def freeze_up_to(cls, 
#                      module, 
#                      stop_layer: Union[int, str]=None,
#                      freeze_bn: bool=True):

#         cls.unfreeze(module, freeze_bn=freeze_bn)

#         if isinstance(stop_layer, str):
#             modules = list(module.named_modules())
#         elif isinstance(stop_layer, int) or (stop_layer is None):
#             modules = list(enumerate(module.modules()))

#         for module_id, m in modules:
#             if stop_layer == module_id:
#                 logger.warning(f"Stopping at layer: {n}")
#                 break
#             cls.freeze(m, freeze_bn=freeze_bn)
#             logger.warning(f"Layer {module_id}: type={type(m)}|training={m.training}")
#             logger.warning(f"requires_grad={np.all([p.requires_grad for p in m.parameters()])}")



class LightningMetricsPlugin:
    """
    LightningMetricsPlugin
    
    To be subclassed by main Pytorch Lightning Module
    
    Available methods:
    * instance methods:
        - log_metric_step
        - init_metrics
    """
    
    
    def log_metric_step(self,
                        stage: str='train',
                        omit_metric_types: Optional[List[str]]=None,
                        omit_metric_keys: Optional[List[str]]=None):
        omit_metric_types = omit_metric_types or []
        omit_metric_keys = omit_metric_keys or []
        
        for metric_type, metric_collection in self.all_metrics[stage].items():
            if metric_type in omit_metric_types:
                continue
            if metric_type == "scalar":
                self.log_dict({k:v for k,v in metric_collection.items() if k not in omit_metric_keys},
                               on_step=False, on_epoch=True, prog_bar=True, logger=True)

            elif metric_type == "per_class":
                for k,v in metric_collection.items():
                    if k in omit_metric_keys:
                        continue
                    results = v.compute()
                    for class_idx, result in enumerate(results): #range(len(results)):
                        self.log(f"{k}_class_{class_idx}", result,
                                 on_step=False, on_epoch=True, prog_bar=False, logger=True)
            else:
                logger.warning(f"[Warning] {metric_type} requires specialized handling in lightningmodule.log_metric_step().")

    def init_metrics(self,
                     stage: str='train',
                     tag: Optional[str]=None):
        tag = tag or ""
        if not hasattr(self, "all_metrics"):
            self.all_metrics = {}
        
        if not hasattr(self,"num_classes") and hasattr(self.hparams, "num_classes"):
            self.num_classes = self.hparams.num_classes
        
        print(f"self.num_classes={self.num_classes}")
        if stage in ['train', 'all']:
            prefix=f'{tag}_train'.strip("_")
            self.metrics_train = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
            self.metrics_train_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='train')
            self.all_metrics['train'] = {"scalar":self.metrics_train,
                                         "per_class":self.metrics_train_per_class}
            
        if stage in ['val', 'all']:
            prefix=f'{tag}_val'.strip("_")
            self.metrics_val = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
            self.metrics_val_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='val')
            self.all_metrics['val'] = {"scalar":self.metrics_val,
                                       "per_class":self.metrics_val_per_class}
            
        if stage in ['test', 'all']:
            if isinstance(tag, str):
                prefix=tag
            else:
                prefix = "test"
#             prefix=f'{tag}_test'.strip("_")
            self.metrics_test = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
            self.metrics_test_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix=prefix)
            self.all_metrics['test'] = {"scalar":self.metrics_test,
                                        "per_class":self.metrics_test_per_class}
            
            


class BaseLightningModule(LightningMetricsPlugin,
                          LayerFreezeLightningPlugin,
                          LayerSelectPlugin,
                          pl.LightningModule):
    """
    BaseLightningModule
    
    Additional methods made available through subclassed plugins.
    
        Available methods:
    * instance methods:
        - step
        - update_metric_step
        - training_step
        - validation_step
        - test_step
    
    Plugins:
    
        -- LightningMetricsPlugin
            * instance methods:
                - log_metric_step
                - init_metrics
        -- LayerFreezeLightningPlugin
            * classmethods:
                - freeze_up_to
                - freeze
                - unfreeze
                - freeze_bn
                - set_bn_eval
            * instance methods:
                - freeze_backbone
                - unfreeze_backbone_top_layers        
        -- LayerSelectPlugin
            * classmethods:
                - count_parameters
            * instance methods:
                - get_batchnorm_modules
                - get_conv_modules
                - get_linear_modules
                - get_named_parameters
                - get_named_modules
                - get_trainable_parameters
                - get_nontrainable_parameters
                - count_trainable_batchnorm_layers
        
    
    
    """
    
    def __init__(self, seed: Optional[int]=None):
        super().__init__()
        self.seed = seed
        pl.seed_everything(seed)

    def step(self, batch, batch_idx):
        image, y_true = batch[0], batch[1]
        y_logit = self(image)
        y_pred = torch.argmax(y_logit, dim=-1)
        return y_logit, y_true, y_pred    
    
    def update_metric_step(self,
                           y_logit,
                           y_true,
                           stage: str='train'):
        out = {}
        for metric_type, metric_collection in self.all_metrics[stage].items():
            out[metric_type] = metric_collection(y_logit, y_true)
        return out
    
    
    def training_step(self, batch, batch_idx):
        if self.eval_bn:
            if self._verbose: logger.debug(f"[training_step] Calling self.set_bn_eval(self.model)")
            self.set_bn_eval(self.model)
            
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
        self.update_metric_step(y_logit,
                                y_true,
                                stage="train")
        self.log_dict({"train_acc": self.metrics_train["train/acc_top1"],
                       "train_loss": loss},
                      on_step=True, on_epoch=True,
                      prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
        self.update_metric_step(y_logit,
                                y_true,
                                stage="val")
        self.log_dict({"val_acc": self.metrics_val["val/acc_top1"],
                       "val_loss": loss},
                      on_step=True, on_epoch=True,
                      prog_bar=True, logger=True)
        

        return {"loss":loss,
                "y_logit":y_logit,
                "y_pred":y_pred,
                "y_true":y_true}
    
    def test_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
        self.update_metric_step(y_logit,
                                y_true,
                                stage="test")
        self.log_dict({"test_acc": self.metrics_test["test/acc_top1"],
                       "test_loss": loss},
                      on_step=False, on_epoch=True,
                      prog_bar=True, logger=True)
        self.log_metric_step(stage="test")
        return {"loss":loss,
                "y_logit":y_logit,
                "y_pred":y_pred,
                "y_true":y_true}
    
    def predict_step(self, batch, batch_idx=None):
        out = self.step(batch, batch_idx)
        if hasattr(batch, "metadata"):
            if "path" in batch.metadata:
                out = [*out, batch.metadata["path"]]
        return out
    

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        """
        Called at the end of predicting.
        """
        
#         y_logit, y_true, y_pred, paths = collect_results(results)
        return collect_results(results)




class LightningClassifier(BaseLightningModule):
    def __init__(self,
                 backbone_name='gluon_seresnext50_32x4d',
                 pretrained: Union[bool, str]=True,
                 num_classes: int=1000,
                 pool_size: int=1,
                 pool_type: str='avg',
                 head_type: str='linear',
                 hidden_size: Optional[int]=512,
                 dropout_p: Optional[float]=0.3,
                 lr: float=2e-03,
                 backbone_lr_mult: float=0.1,
                 feature_extractor_strategy: str="feature_extractor",
                 finetuning_strategy: str=None,
                 weight_decay: float=0.01,
                 scheduler_config: Dict[str,Any]=None,
                 seed: int=None,
                 **kwargs):
        super().__init__(seed=seed)
        self.save_hyperparameters()
        
        self.model = build_model(backbone_name=backbone_name,
                                      pretrained=pretrained,
                                      num_classes=num_classes,
                                      pool_size=pool_size,
                                      pool_type=pool_type,
                                      head_type=head_type,
                                      hidden_size=hidden_size,
                                      dropout_p=dropout_p)
        print(f"self.hparams: {self.hparams}")
        print(f"feature_extractor_strategy: {feature_extractor_strategy}")
        self.set_strategy(feature_extractor_strategy=feature_extractor_strategy,
                          finetuning_strategy=finetuning_strategy)
    
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = self.init_metrics(stage='all')
    
    
    def forward(self,x):
        return self.model(x)
    
    
    def get_lr(self, group: str=None):
        if group is None:
            return self.hparams.lr
        if group == "backbone":
            return self.hparams.lr * self.hparams.backbone_lr_mult
        if group == "head":
            return self.hparams.lr
    
    def configure_optimizers(self):
        print(f"self.hparams={self.hparams}")
        self.optimizers = [torch.optim.AdamW([
			{
				"params":self.model.backbone.parameters(),
				"lr":self.get_lr("backbone"),
				"weight_decay": self.hparams.weight_decay
			},
			{
				"params":self.model.head.parameters(),
				"lr":self.get_lr("head"),
				"weight_decay": self.hparams.weight_decay
			}
		])]
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers, T_max=self.config.t_max, eta_min=self.config.min_lr)
        self.schedulers = configure_schedulers(optimizer=self.optimizers[0],
                                               config=self.hparams.get("scheduler",{}))
    
        return self.optimizers, self.schedulers
    
#         return {'optimizer': self.optimizer,
#                 'scheduler': self.scheduler}

    @classmethod
    def init_pretrained_backbone_w_new_classifier(cls,
                                                  ckpt_path: str,
                                                  new_num_classes: Optional[int]=None,
                                                  **kwargs):
        """
        Create a new instance of this LightningClassifier with:
            - backbone weights pretrained on a custom dataset (like Extant_Leaves)
            - classifier weights randomly initialized
        """
        if isinstance(new_num_classes, int):
            kwargs["num_classes"] = new_num_classes
        model = cls(**kwargs)
        ckpt = torch.load(ckpt_path)
        state_dict = {}
#         if "metadata" in ckpt:
#             state_dict["metadata"] = ckpt["metadata"]
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
            
        backbone_state_dict = OrderedDict({})
        for k,v in state_dict.items():
            if k.startswith("model."):
                k = k.split("model.")[-1]
            if k.startswith("backbone."):
                k = k.split("backbone.")[-1]
            backbone_state_dict[k] = v

        missed_keys = model.model.backbone.load_state_dict(backbone_state_dict, strict=False)
        print(f"missed_keys: {missed_keys}")

        return model
    
    
    def save_backbone_weights(self,
                              ckpt_dir: str,
                              ckpt_filename: str="backbone.ckpt",
                              metadata: Optional[Dict[str, Any]]=None,
                              verbose: bool=True):
        """
        Save the weights from this model's backbone to ${ckpt_dir}/${ckpt_filename}
        """
        
        state_dict = {"state_dict": self.model.model.backbone.state_dict()}
        if isinstance(metadata, dict):
            state_dict["metadata"] = metadata
        
        ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
        torch.save(state_dict, ckpt_path)
        
        if verbose and os.path.isfile(ckpt_path):
            print(f"Saved backbone state_dict to disk at: {ckpt_path}")
            
        if not os.path.isfile(ckpt_path):
            print(f"[WARNING] Error saving model backbone to {ckpt_path} ")
            
        return ckpt_path

    
    def save_model_checkpoint(ckpt_dir: str,
                              ckpt_filename: str="model.ckpt",
                              metadata: Optional[Dict[str, Any]]=None,
                              verbose: bool=True):
        """
        Save the weights from this model's backbone & classifier head to ${ckpt_dir}/${ckpt_filename}
        
        state_dict should have top level keys: 
        - model.backbone
        - model.classifier
        
        """
        
        state_dict = {"state_dict": self.model.model.state_dict()}
        if isinstance(metadata, dict):
            state_dict["metadata"] = metadata
        if hasattr(self, "label_encoder"):
            state_dict["label_encoder"] = self.label_encoder
        ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
        torch.save(state_dict, ckpt_path)
        
        if verbose and os.path.isfile(ckpt_path):
            print(f"Saved backbone state_dict to disk at: {ckpt_path}")
            
        if not os.path.isfile(ckpt_path):
            print(f"[WARNING] Error saving model backbone to {ckpt_path} ")
            
        return ckpt_path
    
    
    @classmethod
    def load_model_from_checkpoint(cls,
                                   ckpt_path: str,
                                   **kwargs):    
        """
        Create a new instance of this LightningClassifier with:
            - backbone & classifier weights pretrained on a custom dataset (like Extant_Leaves)

        """
        model = cls(**kwargs)
        ckpt = torch.load(ckpt_path)
        state_dict = {}
        
        if "label_encoder" in ckpt:
            self.label_encoder = ckpt["label_encoder"]
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
            
        backbone_state_dict = OrderedDict({})
        for k,v in state_dict.items():
            if k.startswith("model.model."):
                k = k.split("model.")[-1]
            backbone_state_dict[k] = v

        missed_keys = model.model.load_state_dict(backbone_state_dict, strict=False)
        print(f"missed_keys: {missed_keys}")

        return model ##, missed_keys
    
