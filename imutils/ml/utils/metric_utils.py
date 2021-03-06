"""
image-utils/imutils/ml/utils/metric_utils.py

[TODO] - refactor metric configuration to be more general & use Hydra config-style configuration.


Author: Jacob A Rose
Created: Thursday June 11th, 2021

"""



import torchmetrics as metrics
# from typing import List


__all__ = ["get_scalar_metrics", "get_per_class_metrics"]


def get_scalar_metrics(num_classes: int,
                       average: str='macro', 
                       prefix: str='',
					   delimiter: str="_"
                      ) -> metrics.MetricCollection:
    default = {f'{average}_acc': metrics.Accuracy(top_k=1, num_classes=num_classes, average=average),
               f'{average}_acc_top3': metrics.Accuracy(top_k=3, num_classes=num_classes, average=average),
               f'{average}_F1':  metrics.F1(top_k=1, num_classes=num_classes, average=average),
               f'{average}_F1_top3':  metrics.F1(top_k=3, num_classes=num_classes, average=average),
               f'{average}_precision': metrics.Precision(top_k=1, num_classes=num_classes, average=average),
               f'{average}_recall': metrics.Recall(top_k=1, num_classes=num_classes, average=average)}
    if len(prefix)>0:
        for k in list(default.keys()):
            default[prefix + delimiter + k] = default[k]
            del default[k]
    
    return metrics.MetricCollection(default)


def get_per_class_metrics(num_classes: int,
                          normalize: str='true',
                          prefix: str=''
                         ) -> metrics.MetricCollection:
    """
    Contents:
        * Per-class F1 metric
        * Confusion Matrix
        
    These metrics return non-scalar results, requiring more careful handling.
    
    Arguments:
        num_classes (int)
        average (str): default='true'.
            The average mode to be applied to the confusion matrix. Options include:
                None or 'none': no normalization (default)
                'true': normalization over the targets (most commonly used)
                'pred': normalization over the predictions
                'all': normalization over the whole matrix
    """
    
    default = {'F1': metrics.F1(num_classes=num_classes, average=None)}#,
#                'ConfusionMatrix': metrics.ConfusionMatrix(num_classes=num_classes, normalize=normalize)}
    
    
    if len(prefix)>0:
        for k in list(default.keys()):
            default[prefix + r'/per_class/' + k] = default[k]
            del default[k]
    
    
    
    
    return metrics.MetricCollection(default)