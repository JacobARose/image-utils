import glob
import os
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb
import pandas as pd
# from torchmetrics import metrics
from torch.utils.data import DataLoader #,Dataset, Subset, random_split
from pytorch_lightning import Callback, Trainer, LightningDataModule
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from pathlib import Path
# from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


from imutils.ml.utils.plot_utils import plot_confusion_matrix

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score



# def get_labels_from_filepath(path: str, fix_catalog_number: bool = False):
#     return None

# def get_labels_from_filepath(path: str, fix_catalog_number: bool = False) -> Dict[str,str]:
#     """
#     Splits a precisely-formatted filename with the expectation that it is constructed with the following fields separated by '_':
#     1. family
#     2. genus
#     3. species
#     4. collection
#     5. catalog_number
    
#     If fix_catalog_number is True, assume that the collection is not included and must separately be extracted from the first part of the catalog number.
    
#     """
#     family, genus, species, collection, catalog_number = Path(path).stem.split("_", maxsplit=4)
#     if fix_catalog_number:
#         catalog_number = '_'.join([collection, catalog_number])
#     return {"family":family,
#             "genus":genus,
#             "species":species,
#             "collection":collection,
#             "catalog_number":catalog_number}





class ImagePredictionLogger(Callback):
    def __init__(self, 
                 datamodule: LightningDataModule,
                 log_every_n_epochs: int=1,
                 subset: str='val',
                 max_samples_per_epoch: int=64,
                 fix_catalog_number: bool=False):
        super().__init__()
        self.max_samples_per_epoch = max_samples_per_epoch
        self.log_every = log_every_n_epochs
        self.subset = subset
        self.datamodule = datamodule
        self.classes = self.datamodule.classes
        self.fix_catalog_number = fix_catalog_number
        self.reset_iterator()
        
#     def on_validation_epoch_end(self, trainer, model):
#         print('Inside hook: on_validation_epoch_end(self, trainer, model)')
        
    def reset_iterator(self):
        self.datamodule.return_paths = True
        stage = 'test' if self.subset=='test' else 'fit'
        self.datamodule.setup(stage=stage)
        self.data_iterator = iter(self.datamodule.get_dataloader(self.subset))
        
    def on_validation_epoch_end(self, trainer, model):
#         self.datamodule.batch_size = self.max_samples_per_epoch
#         self.datamodule.return_paths = True
#         stage = 'test' if self.subset=='test' else 'fit'
#         self.datamodule.setup(stage=stage)
#         x, y, paths = next(iter(self.datamodule.get_dataloader(self.subset)))
        x, y, paths = [], [], []
        for idx, batch in enumerate(self.data_iterator):
            x_batch, y_batch = batch[:2]
            x.append(x_batch.detach().cpu().numpy())
            y.append(y_batch.detach().cpu().numpy())

            if idx*x_batch.shape[0] >= self.max_samples_per_epoch:
                break
        if len(x)==0:
            self.reset_iterator()
            return
        x = torch.cat(x, 0)
        y = torch.cat(y, 0)

        self.current_epoch = trainer.current_epoch
        skip_epoch = self.current_epoch % self.log_every > 0
        if skip_epoch:
            print(f'Current epoch: {self.current_epoch}. Skipping Image Prediction logging.')
            return
        print(f'Current epoch: {self.current_epoch}.\nInitiating Image Prediction Artifact creation.')
        # Get model prediction
        if model.training:
            training = True
            subset='train'
        elif self.subset=="test":
            training = False
            subset='test'
        else:
            training = False
            subset='val'

        
        model.eval()
        logits = model.cpu()(x)
        preds = torch.argmax(logits, -1).detach().cpu().numpy()
        scores = logits.softmax(1).detach().cpu().numpy()
        
        columns = ['catalog_number',
                   'image',
                   'guess',
                   'truth']
        for j, class_name in enumerate(self.classes):
            columns.append(f'score_{class_name}')

        x = x.permute(0,2,3,1)
        prediction_rows = []
        x = (255 * (x - x.min()) / (x.max() - x.min())).numpy().astype(np.uint8)
        for i in range(len(preds)):
            labels = get_labels_from_filepath(path=paths[i],
                                        fix_catalog_number=self.fix_catalog_number)        
            row = [
                    labels['catalog_number'],
                    wandb.Image(x[i,...]),
                    self.classes[preds[i]],
                    self.classes[y[i]]
            ]
            
            for j, score_j in enumerate(scores[i,:].tolist()):
                row.append(np.round(score_j, 4))
            prediction_rows.append(row)
            
        prediction_table = wandb.Table(data=prediction_rows, columns=columns)
        prediction_artifact = wandb.Artifact(f"{self.subset}_predictions" + wandb.run.id, 
                                             type="predictions")
        prediction_artifact.add(prediction_table, f"{self.subset}_predictions")
        wandb.run.log_artifact(prediction_artifact)
        
        if training:
            model.train()
        model.cuda()

        
###################################
###################################




def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    print(type(trainer.logger))
    
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)



from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import os

CHECKPOINT_FOLDER = 'checkpoints'

class WandbModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        if not wandb.run:
            raise Exception('Wandb has not been initialized. Please call wandb.init first.')
        wandb_dir = os.path.abspath(wandb.run.dir)
        super().__init__(dirpath=os.path.join(wandb_dir, CHECKPOINT_FOLDER), *args, **kwargs)








class UploadCodeToWandbAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadCheckpointsToWandbAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class LogPerClassMetricsToWandb(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self,
                 class_names: List[str] = None,
                 logits_key: str="logits",
                 true_labels_key: str="y_true"):
        self.class_names = class_names
        self.logits_key = logits_key
        self.true_labels_key = true_labels_key
        self.preds = []
        self.targets = []
        self.path = []
        self.catalog_number = []
        self.ready = True
        
        self.annotation_class_name_max_len = 125

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if not isinstance(outputs, dict):
            return
        if 'log' in outputs.keys():
            outputs = outputs['log']
        elif 'hidden' in outputs.keys():
            outputs = outputs['hidden']
        if self.ready:
            self.path.extend(getattr(batch, "path", []))
            self.catalog_number.extend(getattr(batch, "catalog_number", []))
            self.preds.append(outputs[self.logits_key].detach().cpu())
            self.targets.append(outputs[self.true_labels_key].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap,
        then generate confusion matrix."""
        if self.ready and len(self.preds) and len(self.targets):
            wandb_logger = get_wandb_logger(trainer=trainer)
            rank = pl_module.global_rank
#             print(f'Rank: {rank}')
#             print(f"wandb_logger.experiment={wandb_logger.experiment}")
#             print(f"dir(wandb_logger.experiment)={dir(wandb_logger.experiment)}")
#             print(f"wandb_logger.experiment.disabled={wandb_logger.experiment.disabled}")
#             print(f"wandb_logger.experiment.id={wandb_logger.experiment.id}")
#             print(f"wandb_logger.experiment.name={wandb_logger.experiment.name}")
#             if
            if (rank > 0) or (wandb_logger is None) or (wandb_logger.experiment.id is None):
                print(f"(wandb_logger.experiment.id is None) = {(wandb_logger.experiment.id is None)}")
                print(f"Rank>0, skipping per class metrics\n", "="*20)
                return


            preds = torch.cat(self.preds).numpy() #.cpu().numpy()
            targets = torch.cat(self.targets).numpy() #.cpu().numpy()
            
#             print(f"preds.shape={preds.shape}")
#             print(f"targets.shape={targets.shape}")
            
            # self._log_classification_report(preds, targets, logger=wandb_logger)
        
#             self._log_per_class_scores(preds, targets, logger=wandb_logger)
            self._log_confusion_matrix(preds, targets, logger=wandb_logger, epoch=trainer.current_epoch)
            wandb_logger.experiment.log({f"current_epoch": trainer.current_epoch})
            
#             print("num_validation_samples: ", len(preds))
            self.preds.clear()
            self.targets.clear()
            self.preds = []
            self.targets = []
            self.path = []
            self.catalog_number = []
            
    def _log_confusion_matrix(self, preds: np.ndarray, targets: np.ndarray, logger, epoch: Optional[int]=0):
        """
        Generate confusion_matrix heatmap
        """
        
        sns_context = "poster"
        sns_style = "seaborn-bright"
        cmap="YlGnBu_r"
        sns.set_context(context=sns_context, font_scale=0.7)
        plt.style.use(sns_style)
        
        # if targets.ndim >= 2:
        #     targets = np.argmax(targets, axis=1)
        if preds.ndim >= 2:
            preds = np.argmax(preds, axis=1)
        
        confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_true=targets, y_pred=preds))
        confusion_matrix.index.name = "True"
        confusion_matrix = confusion_matrix.T
        confusion_matrix.index.name = "Predicted"
        confusion_matrix = confusion_matrix.T
        
        try:
            plot_confusion_matrix(cm=confusion_matrix, title=None)
            print(f"val/confusion_matrix/{logger.experiment.name}")
            # current_epoch = self.current_epoch
            logger.experiment.log({
                f"val/confusion_matrix/{logger.experiment.name}": wandb.Image(plt),
                "current_epoch":epoch}, commit=False)
#             logger.experiment.log({f"val/confusion_matrix_png/{logger.experiment.name}": wandb.Image()}, commit=False)

        except Exception as e:
            print(e)
            print('continuing anyway')

        plt.clf()
        plt.close(plt.gcf())
            
            
            
            
    def _log_classification_report(self, preds: np.ndarray, targets: np.ndarray, logger):
        

#         true = np.random.randint(0, num_classes, size=num_samples)
#         pred = np.random.randint(0, num_classes, size=num_samples)
#         labels = np.arange(num_classes)
#         target_names = [r.get_random_word() for _ in range(num_classes)]
        # target_names = list("ABCDEFGHI")

        clf_report = classification_report(targets,
                                           preds,
                                           zero_division=0,
                                           output_dict=True)
        
        
        # table = wandb.Table(data=[(label,

        sns.set_theme(context="talk")
        fig, ax = plt.subplots(1,2, figsize=(18,30))

        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, annot_kws={"fontsize":6}, ax=ax[0])
        sns.heatmap(pd.DataFrame(clf_report).iloc[-1:, :-3].T, annot=True, annot_kws={"fontsize":6}, ax=ax[1])

#         fig = plt.figure(figsize=(15,29))
#         sns.heatmap(pd.DataFrame(clf_report).iloc[:, :].T, annot=True, annot_kws={"fontsize":7})
        logger.experiment.log({f"val/classification_report": wandb.Image(plt)}, commit=False)
#         logger.experiment.log({f"val/classification_report": wandb.Image(fig)}, commit=False)
        plt.clf()
        plt.close(fig)
            
            
            
    def _log_per_class_scores(self, preds: np.ndarray, targets: np.ndarray, logger):
        """
        Generate f1, precision and recall heatmap
        """
        sns_context = "poster"
        sns_style = "seaborn-bright"
        cmap="YlGnBu_r"
        sns.set_context(context=sns_context, font_scale=0.7)
        plt.style.use(sns_style)
        
        from icecream import ic
        ic(preds.ndim, targets.ndim)
        # if targets.ndim >= 2:
        #     targets = np.argmax(targets, axis=1)
        if preds.ndim >= 2:
            preds = np.argmax(preds, axis=1)

        f1 = f1_score(preds, targets, average=None, zero_division=0)
        r = recall_score(preds, targets, average=None, zero_division=0)
        p = precision_score(preds, targets, average=None, zero_division=0)

        
        class_indices, support = np.unique(targets, return_counts=True)
        
        num_classes = len(self.class_names)
#         print(f1.shape, r.shape, p.shape, support.shape)
        
        if len(support) < num_classes:
            f1_full = np.zeros(num_classes)
            r_full = np.zeros_like(f1_full)
            p_full = np.zeros_like(f1_full)
            support_full = np.zeros_like(f1_full)
            
#             for i, support_class_i in zip(class_indices, support):
            for i, class_i in enumerate(class_indices):

#                 if class_i >= len(support)-1:
#                     break
                f1_full[class_i] = f1[i]
                r_full[class_i] = r[i]
                p_full[class_i] = p[i]
                support_full[class_i] = support[i]
            f1 = f1_full
            r = r_full
            p = p_full
            support = support_full
            
            
        data = [f1, p, r, support]
        
#         for d in data:
#             print(f"len(d)={len(d)}")
        w = int(len(self.class_names)//10) + 10
        h = 10
        plt.figure(figsize=(w, h))
        annot = bool(len(self.class_names) < self.annotation_class_name_max_len)
        xticklabels = self.class_names if annot else []
        yticklabels=["F1", "Precision", "Recall", "Support"]

        g = sns.heatmap(data.T,
                        annot=annot,
                        vmin=0.0, vmax=1.0,
                        linewidths=1, annot_kws={"size": 8}, fmt=".2f", cmap=cmap,
                        yticklabels=yticklabels,
                        xticklabels=xticklabels)
        plt.suptitle(f"(M={num_classes}) per-class F1_Precision_Recall -- heatmap")
        g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize='medium', fontweight='light')
        g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right', fontsize='medium', fontweight='light')
        plt.subplots_adjust(bottom=0.2, top=0.95, wspace=None, hspace=0.07)
        try:
            logger.experiment.log({f"val/per_class/f1_p_r_heatmap": wandb.Image(plt.gcf())}, commit=False)
        except Exception as e:
            print(e)
            logger.experiment.log({f"val/per_class/f1_p_r_heatmap": wandb.Image(plt)}, commit=False)
            print('retry successful')
        plt.clf()
        plt.close(fig)
        
#         import pdb; pdb.set_trace()
        
        logger.experiment.log({f"val/per_class/f1_p_r_table": 
                               wandb.Table(dataframe=pd.DataFrame(np.stack(data).T,
                                                                  index=xticklabels,
                                                                  columns=yticklabels).T
                                          )
                              }, commit=False)
        
        num_samples = len(preds)
        idx = list(range(len(self.class_names)))
        
        for j, metric in enumerate(["F1", "Precision", "Recall"]):
#             print(f"metric={metric},", f"len(data[j])={len(data[j])}")
            metric_data = pd.DataFrame(data[j]).T.to_records()
#             print(f"val/per_class/{metric}_distributions: metric_data[0].shape={metric_data.shape}")
            logger.experiment.log({f"val/per_class/{metric}_distributions" : wandb.plot.line_series(xs=idx,
                                                                                                  ys=metric_data,
                                                                                                  keys=self.class_names,
                                                                                                  title=f"per-class {metric} -- time series",
                                                                                                  xname="family")},
                                                                                                  commit=False)

        
        
        
        
###################################################


# source: https://colab.research.google.com/drive/1k89TDv8ybckgfVByUIhY6peBjtNGBH-k?usp=sharing#scrollTo=RO1MSGLeAzWp
class WandbClassificationCallback(Callback):

    def __init__(self, monitor='val_loss', verbose=0, mode='auto',
                 save_weights_only=False, log_weights=False, log_gradients=False,
                 save_model=True, training_data=None, validation_data=None,
                 labels=[], data_type=None, predictions=1, generator=None,
                 input_type=None, output_type=None, log_evaluation=False,
                 validation_steps=None, class_colors=None, log_batch_frequency=None,
                 log_best_prefix="best_", 
                 log_confusion_matrix=False,
                 confusion_examples=0, confusion_classes=5):
        
        super().__init__(monitor=monitor,
                        verbose=verbose, 
                        mode=mode,
                        save_weights_only=save_weights_only,
                        log_weights=log_weights,
                        log_gradients=log_gradients,
                        save_model=save_model,
                        training_data=training_data,
                        validation_data=validation_data,
                        labels=labels,
                        data_type=data_type,
                        predictions=predictions,
                        generator=generator,
                        input_type=input_type,
                        output_type=output_type,
                        log_evaluation=log_evaluation,
                        validation_steps=validation_steps,
                        class_colors=class_colors,
                        log_batch_frequency=log_batch_frequency,
                        log_best_prefix=log_best_prefix)
                        
        self.log_confusion_matrix = log_confusion_matrix
        self.confusion_examples = confusion_examples
        self.confusion_classes = confusion_classes
               
    def on_epoch_end(self, epoch, logs={}):
        if self.generator:
            self.validation_data = next(self.generator)

        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)
        
        if self.log_confusion_matrix:
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                wandb.log(self._log_confusion_matrix(), commit=False)                    

        if self.input_type in ("image", "images", "segmentation_mask") or self.output_type in ("image", "images", "segmentation_mask"):
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                if self.confusion_examples > 0:
                    wandb.log({'confusion_examples': self._log_confusion_examples(
                                                    confusion_classes=self.confusion_classes,
                                                    max_confused_examples=self.confusion_examples)}, commit=False)
                if self.predictions > 0:
                    wandb.log({"examples": self._log_images(
                        num_images=self.predictions)}, commit=False)

        wandb.log({'epoch': epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary["%s%s" % (self.log_best_prefix, self.monitor)] = self.current
                wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
                if self.verbose and not self.save_model:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f' % (
                        epoch, self.monitor, self.best, self.current))
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current
        
    def _log_confusion_matrix(self):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_val = np.argmax(y_val, axis=1)
        y_pred = np.argmax(self.model.predict(x_val), axis=1)

        confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
        confdiag = np.eye(len(confmatrix)) * confmatrix
        np.fill_diagonal(confmatrix, 0)

        confmatrix = confmatrix.astype('float')
        n_confused = np.sum(confmatrix)
        confmatrix[confmatrix == 0] = np.nan
        confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': self.labels, 'y': self.labels, 'z': confmatrix,
                                 'hoverongaps':False, 'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})

        confdiag = confdiag.astype('float')
        n_right = np.sum(confdiag)
        confdiag[confdiag == 0] = np.nan
        confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': self.labels, 'y': self.labels, 'z': confdiag,
                               'hoverongaps':False, 'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})

        fig = go.Figure((confdiag, confmatrix))
        transparent = 'rgba(0, 0, 0, 0)'
        n_total = n_right + n_confused
        fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [1, f'rgba(180, 0, 0, {max(0.2, (n_confused/n_total) ** 0.5)})']], 'showscale': False}})
        fig.update_layout({'coloraxis2': {'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, {min(0.8, (n_right/n_total) ** 2)})'], [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})

        xaxis = {'title':{'text':'y_true'}, 'showticklabels':False}
        yaxis = {'title':{'text':'y_pred'}, 'showticklabels':False}

        fig.update_layout(title={'text':'Confusion matrix', 'x':0.5}, paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)
        
        return {'confusion_matrix': wandb.data_types.Plotly(fig)}

    def _log_confusion_examples(self, rescale=255, confusion_classes=5, max_confused_examples=3):
            x_val = self.validation_data[0]
            y_val = self.validation_data[1]
            y_val = np.argmax(y_val, axis=1)
            y_pred = np.argmax(self.model.predict(x_val), axis=1)

            # Grayscale to rgb
            if x_val.shape[-1] == 1:
                x_val = np.concatenate((x_val, x_val, x_val), axis=-1)

            confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
            np.fill_diagonal(confmatrix, 0)

            def example_image(class_index, x_val=x_val, y_pred=y_pred, y_val=y_val, labels=self.labels, rescale=rescale):
                image = None
                title_text = 'No example found'
                color = 'red'

                right_predicted_images = x_val[np.logical_and(y_pred==class_index, y_val==class_index)]
                if len(right_predicted_images) > 0:
                    image = rescale * right_predicted_images[0]
                    title_text = 'Predicted right'
                    color = 'rgb(46, 184, 46)'
                else:
                    ground_truth_images = x_val[y_val==class_index]
                    if len(ground_truth_images) > 0:
                        image = rescale * ground_truth_images[0]
                        title_text = 'Example'
                        color = 'rgb(255, 204, 0)'

                return image, title_text, color

            n_cols = max_confused_examples + 2
            subplot_titles = [""] * n_cols
            subplot_titles[-2:] = ["y_true", "y_pred"]
            subplot_titles[max_confused_examples//2] = "confused_predictions"
            
            n_rows = min(len(confmatrix[confmatrix > 0]), confusion_classes)
            fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
            for class_rank in range(1, n_rows+1):
                indx = np.argmax(confmatrix)
                indx = np.unravel_index(indx, shape=confmatrix.shape)
                if confmatrix[indx] == 0:
                    break
                confmatrix[indx] = 0

                class_pred, class_true = indx[0], indx[1]
                mask = np.logical_and(y_pred==class_pred, y_val==class_true)
                confused_images = x_val[mask]

                # Confused images
                n_images_confused = min(max_confused_examples, len(confused_images))
                for j in range(n_images_confused):
                    fig.add_trace(go.Image(z=rescale*confused_images[j],
                                        name=f'Predicted: {self.labels[class_pred]} | Instead of: {self.labels[class_true]}',
                                        hoverinfo='name', hoverlabel={'namelength' :-1}),
                                row=class_rank, col=j+1)
                    fig.update_xaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j+1, mirror=True)
                    fig.update_yaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j+1, mirror=True)

                # Comparaison images
                for i, class_index in enumerate((class_true, class_pred)):
                    col = n_images_confused+i+1
                    image, title_text, color = example_image(class_index)
                    fig.add_trace(go.Image(z=image, name=self.labels[class_index], hoverinfo='name', hoverlabel={'namelength' :-1}), row=class_rank, col=col)    
                    fig.update_xaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True, title_text=title_text)
                    fig.update_yaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True, title_text=self.labels[class_index])

            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            
            return wandb.data_types.Plotly(fig)