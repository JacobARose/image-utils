"""
image-utils/imutils/ml/models/backbone.py

Author: Jacob A Rose
Created: Saturday May 29th, 2021

"""

from typing import Any, List, Optional, Dict, Tuple, Union
from omegaconf import DictConfig
import torch
from torch import nn, functional as F
import torchvision
from pytorch_lightning import LightningModule

import torchmetrics as metrics
import timm
from torchsummary import summary
import pandas as pd
from pathlib import Path
from stuf import stuf
import wandb
import numpy as np
from rich import print as pp
# from . import resnet, senet, efficientnet

from imutils.ml.models.layers.pool_layers import build_global_pool, Flatten

__all__ = ["build_model", "build_model_head", "build_model_backbone", "load_model_checkpoint"]

# AVAILABLE_MODELS = {"resnet":resnet.AVAILABLE_MODELS,
#                     "senet":senet.AVAILABLE_MODELS,
#                     "efficientnet":efficientnet.AVAILABLE_MODELS}


def load_model_checkpoint(model, ckpt_path: str):
    ckpt_path = glob.glob(hydra.utils.to_absolute_path(ckpt_path))
    model = model.load_state_dict(torch.load(ckpt_path[0]))
    return model

from collections import OrderedDict
from imutils.ml.models.base import BaseModule


def build_timm_backbone(backbone_name='gluon_seresnext50_32x4d',
                         pretrained: Union[bool, str]=True,
                         num_classes: int=1000,
                         freeze_backbone: bool=False,
                         feature_layer: int=-2):
    if pretrained == "imagenet":
        num_classes = 1000

    model = timm.create_model(model_name=backbone_name, num_classes=num_classes, pretrained=pretrained)
    if isinstance(pretrained, str) and pretrained != "imagenet":
        model = load_model_checkpoint(model, ckpt_path=pretrained)
        
#     body = nn.Sequential(*list(model.children())[:feature_layer])

    body = nn.Sequential(OrderedDict(
        list(
            model.named_children()
        )[:feature_layer]
    ))
    
    if freeze_backbone:
        print(f"Freezing all layers in backbone of model.")
        BaseModule.freeze(body)
    
    return body



# def build_generic_backbone(backbone_name='gluon_seresnext50_32x4d',
#                             pretrained: Union[bool, str]=True,
#                             num_classes: int=1000,
#                             feature_layer: int=-2,
#                             model_repo: str= "timm",
#                             _target_=None):
#     cfg = DictConf(dict(
#         backbone_name=backbone_name,
#         pretrained=pretrained,
#         num_classes=1000,
#         feature_layer=feature_layer,
#         model_repo=model_repo
#     ))
#     hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
#     model: pl.LightningModule = hydra.utils.instantiate(cfg, _recursive_=False)
#     return model                    




def build_model_backbone(backbone_name='gluon_seresnext50_32x4d',
                         pretrained: Union[bool, str]=True,
                         num_classes: int=1000,
                         feature_layer: int=-2,
                         model_repo: str= "timm",
                        _target_=None,
                        **kwargs):

    if model_repo == "timm":
        return build_timm_backbone(backbone_name=backbone_name,
                                   pretrained=pretrained,
                                   num_classes=num_classes,
                                   feature_layer=feature_layer)
    else:
        try:
            print("Trying non-timm based model loading")
            cfg = DictConf(dict(
                backbone_name=backbone_name,
                pretrained=pretrained,
                num_classes=1000,
                feature_layer=feature_layer,
                model_repo=model_repo
            ))
            # return hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
            # model: pl.LightningModule = hydra.utils.instantiate(cfg.model, cfg=cfg, _recursive_=False)
        except:
            # TBD Add other pretrained model backends
            raise NotImplementedError(f"Invalid model_repo={model_repo}")

def build_model_head(num_classes: int=1000,
                     pool_size: int=1,
                     pool_type: str='avg',
                     head_type: str='linear',
                     feature_size: int=512,
                     hidden_size: Optional[int]=512,
                     dropout_p: Optional[float]=0.3,
                    **kwargs):
    """
    
    Returns a nn.Sequential model containing 3 children:
        global_pool -> flatten -> classifier
        
    Available pool_types:
        - "avg"
            global_avg_pool
        - "avgdrop"
            global_avg_pool -> dropout
        - "avgmax"
            [global_avg_pool | global_max_pool]
        "max"
            global_max_pool
            
            
    pool_types to be explored:
        - "maxdrop"
            global_max_pool -> dropout
        - "avgmaxdrop"
            [global_avg_pool | global_max_pool] -> dropout

        
    Available head_types:
        - linear
        
        - custom
    
    """

    head = OrderedDict()
    global_pool, feature_size = build_global_pool(pool_type=pool_type,
                                                  pool_size=pool_size,
                                                  feature_size=feature_size,
                                                  dropout_p=dropout_p)
    head["global_pool"] = global_pool
    head["flatten"] = Flatten()
    
    classifier_input_feature_size = feature_size*(pool_size**2)
    if head_type=='linear':
        hidden_size = 0
        head["classifier"] = nn.Linear(classifier_input_feature_size, num_classes)
    elif head_type=='custom':
        head["classifier"] = nn.Sequential(nn.Linear(classifier_input_feature_size, hidden_size),
                                nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False),
                                nn.BatchNorm1d(hidden_size),
                                nn.Linear(hidden_size, num_classes))
    head = nn.Sequential(head)
    print(f"Initializing weights of the model head.")
    BaseModule.initialize_weights(head)
    return head



def build_model(backbone_cfg: Optional[DictConfig]=None,
                head_cfg: Optional[DictConfig]=None,
                backbone_name='gluon_seresnext50_32x4d',
                pretrained: Union[bool, str]=True,
                num_classes: int=1000,
                freeze_backbone: bool=False,
                pool_size: int=1,
                pool_type: str='avg',
                head_type: str='linear',
                hidden_size: Optional[int]=512,
                dropout_p: Optional[float]=0.3):
    """
    
    Creates a classifier lightning module from a backbone model + a classication head.
    
    
    Return:
    ```
            model = nn.Sequential(OrderedDict({
                "backbone":backbone,
                "head":head
            }))
    ```

    
    """
    if backbone_cfg is None:
        backbone = build_model_backbone(backbone_name=backbone_name,
                                        pretrained=pretrained,
                                        num_classes=num_classes,
                                        freeze_backbone=freeze_backbone,
                                        feature_layer=-2)
    else:
        backbone = build_model_backbone(**backbone_cfg)
    
    feature_size = list(backbone.parameters())[-1].shape[0]
    
    if head_cfg is None:
        head = build_model_head(num_classes=num_classes,
                                pool_size=pool_size,
                                pool_type=pool_type,
                                head_type=head_type,
                                feature_size=feature_size,
                                hidden_size=hidden_size,
                                dropout_p=dropout_p)
    else:
        head = build_model_head(feature_size=feature_size,
                                **head_cfg)
    
    model = nn.Sequential(OrderedDict({
        "backbone":backbone,
        "head":head
    }))

    
    
    return model


from imutils.ml.utils.model_utils import log_model_summary


        



## Save for later (commented out on 10-17-21)
# def build_model(model_name: str,
#                 pretrained: bool=False,
#                 progress: bool=True,
#                 num_classes: int=1000,
#                 global_pool_type: str='avg',
#                 drop_rate: float=0.0,
#                 **kwargs) -> nn.Module:

    
#     if model_name in resnet.AVAILABLE_MODELS:
#         ModelFactory = resnet.build_model
        
# #     elif 'senet' in model_name:
#     elif model_name in senet.AVAILABLE_MODELS:
#         ModelFactory = senet.build_model

# #     elif 'efficientnet' in model_name:
#     elif model_name in efficientnet.AVAILABLE_MODELS:
#         ModelFactory = efficientnet.build_model
#     else:
#         print(f"model with name {model_name} has not be implemented yet.")
#         print("Available Models:")
#         pp(AVAILABLE_MODELS)
#         return None
    
#     model = ModelFactory(model_name=model_name,
#                          pretrained=pretrained,
#                          progress=progress,
#                          num_classes=num_classes,
#                          global_pool_type=global_pool_type,
#                          drop_rate=drop_rate,
#                          **kwargs)

#     print(f"[BUILDING MODEL] build_model({model_name}, pretrained={pretrained})")
    
#     return model
    
#     model.name = model_name
#     model.pretrained = pretrained
#     return model










# 3.a Optional: Register a custom backbone
# This is useful to create new backbone and make them accessible from `ImageClassifier`
# @ImageClassifier.backbones(name="resnet18")
# def fn_resnet(pretrained: bool = True):
#     model = torchvision.models.resnet18(pretrained)
#     # remove the last two layers & turn it into a Sequential model
#     backbone = nn.Sequential(*list(model.children())[:-2])
#     num_features = model.fc.in_features
#     # backbones need to return the num_features to build the head
#     return backbone, num_features


# def create_classifier(num_features: int, num_classes: int, pool_type='avg', bias: bool=True):
#     global_pool = nn.AdaptiveAvgPool2d(1)
#     flatten_layer = nn.Flatten()
#     linear_layer = nn.Linear(num_features, num_classes, bias=bias)
#     return [global_pool, flatten_layer, linear_layer]

