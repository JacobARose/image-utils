"""
Created on: Tuesday, April 20th, 2021
Author: Jacob A Rose


"""

import torchmetrics as metrics
from torch import nn
import collections



from torchinfo import summary
import os
from typing import Tuple, Optional
from torch import nn

from torchinfo import summary


__all__ = ["count_parameters", "collect_results", "log_model_summary"]

from more_itertools import unzip
from toolz.itertoolz import concat
from prettytable import PrettyTable



def count_parameters(model, verbose: bool=True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        
        total_params += param
        if not parameter.requires_grad:
            continue
        table.add_row([name, param])
        trainable_params+=param
    if verbose:
        print(table)
    print(f"Total Trainable Params: {trainable_params:,}")
    print(f"Total non-Trainable Params: {total_params-trainable_params:,}")
    return table


def collect_results(results):
    """
    Converts a list of flat records/rows to a list of tall, row-wise concatenated columns
    
    Useful for collecting a set of batches or individual samples of 0- and 1-dimensional tensors returned by each model step, collected in a list in model.on_epoch_end.
    
    in:
        [(item_00, item_01,... item_0C),
         (item_10, item_11,... item_1C),
         ...
         (item_N0, item_N1,... item_NC)]
     out:
         [[item_00, item_10,... item_N0],
         [item_01, item_11,... item_N1],
         ...
         [item_0C, item_1C,... item_NC]]
    
    """

    rows = [list(concat(r)) for r in unzip(results)]
    cols = []*len(rows)
    print(cols)
    for i, row in enumerate(rows):
        if isinstance(row[0], torch.Tensor):
            if len(row[0].shape) <= 1:
                cols.append(torch.stack(row, dim=0).cpu().numpy())
            else:
                cols.append(torch.cat(row).cpu().numpy())
        elif isinstance(row[0], list):
            cols.extend(list(concat(row)))
        elif isinstance(row[0], (str, int)):
            cols.append(list(row))

    np.all([len(c)==len(cols[0]) for c in cols])
    
    return cols





def log_model_summary(model: nn.Module,
                      input_size: Tuple[int],
                      full_summary: bool=True,
                      working_dir: str=".",
                      model_name: Optional[str]=None,
                      verbose: bool=1):
    """
    produce a text file with the model summary
    
    TODO: Add this to Eval Plugins
    
    log_model_summary(model=model,
                  working_dir=working_dir,
                  input_size=(1, data_config.channels, *data_config.image_size),
                  full_summary=True)

    """

    if full_summary:
        col_names=("kernel_size", "input_size","output_size", "num_params", "mult_adds")
    else:
        col_names=("input_size","output_size", "num_params")

    model_summary = summary(model.cuda(),
                            input_size=input_size,
                            row_settings=('depth', 'var_names'),
                            col_names=col_names,
                            verbose=verbose)

    if (model_name is None) and (hasattr(model, "name")):
        model_name = model.name
    if (model_name is None):
        summary_path = os.path.join(working_dir, f'model_summary.txt')
    else:
        summary_path = os.path.join(working_dir, f'{model_name}_model_summary.txt')
    
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w") as f:
        f.write(str(model_summary))
        
    return model_summary







# source: https://github.com/BloodAxe/pytorch-toolbelt/blob/master/pytorch_toolbelt/utils/torch_utils.py
def transfer_weights(model: nn.Module, model_state_dict: collections.OrderedDict):
    """
    Copy weights from state dict to model, skipping layers that are incompatible.
    This method is helpful if you are doing some model surgery and want to load
    part of the model weights into different model.
    :param model: Model to load weights into
    :param model_state_dict: Model state dict to load weights from
    :return: None
    """
    for name, value in model_state_dict.items():
        try:
            model.load_state_dict(collections.OrderedDict([(name, value)]), strict=False)
        except Exception as e:
            print(e)
