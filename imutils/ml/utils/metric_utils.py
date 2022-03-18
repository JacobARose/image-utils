"""
lightning_hydra_classifiers/utils/metric_utils.py

Author: Jacob A Rose
Created: Thursday Junee 11th, 2021

"""



import torchmetrics as metrics
# from typing import List


__all__ = ["get_scalar_metrics", "get_per_class_metrics"]


def get_scalar_metrics(num_classes: int,
                       average: str='macro', 
                       prefix: str=''
                      ) -> metrics.MetricCollection:
    default = {'acc_top1': metrics.Accuracy(top_k=1, num_classes=num_classes, average=average),
               'acc_top3': metrics.Accuracy(top_k=3, num_classes=num_classes, average=average),
               'F1_top1':  metrics.F1(top_k=1, num_classes=num_classes, average=average),
               'precision_top1': metrics.Precision(top_k=1, num_classes=num_classes, average=average),
               'recall_top1': metrics.Recall(top_k=1, num_classes=num_classes, average=average)}
    if len(prefix)>0:
        for k in list(default.keys()):
            default[prefix + r'/' + k] = default[k]
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