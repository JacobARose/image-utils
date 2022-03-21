"""
imutils/ml/utils/plot_utils.py


Created on: Tuesday, July 27th, 2021
Author: Jacob A Rose


"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import List, Optional
from PIL import Image

__all__ = ["plot_confusion_matrix", "colorbar", "display_images"]






    
def plot_confusion_matrix(cm: pd.DataFrame,
                          labels: List[str]=None,
                          robust: bool=False,
                          title: str="Confusion matrix of val predictions for model",
                          save_path: Optional[str]=None,
                          backend="seaborn"):
    """
    Helper function with good defaults for quickly plotting confusion matrices, whether num_classes=10 or 200.
    
    Optionally plot as an interactive plot using plotly, defaults to seaborn.
    
    Caution: confusion matrices with 50 or more classes can cause major lag issues with plotly.
    
    """
    num_classes = cm.shape[0]
    
    linewidths = 0.01
    if num_classes < 15:
        figsize = (9,8)
    elif num_classes < 60:
        figsize = (18,16)
    elif num_classes < 150:
        figsize = (18,16)
        linewidths = 0.0
    elif num_classes >= 150:
        figsize = (27, 24)
        linewidths = 0.0

    annot = True if num_classes<=35 else False

    if backend=="plotly":
        import plotly.figure_factory as ff
        
        size = 20*num_classes
        fig = ff.create_annotated_heatmap(z=cm,
                                          x=labels,
                                          y=labels)
        fig.update_layout(height=size,
                          width=size+40)
#         fig = px.imshow(cm,
#                         labels=dict(x="Predicted Labels", y="True Labels", color="Count"),
#                         title=title)
        ax = plt.gca()
        if isinstance(save_path, str):
            fig.write_html(f"{save_path}.html")
            
    elif backend=="seaborn":
        fig, ax = plt.subplots(1,1, figsize=figsize)

        sns.heatmap(cm,
                    linewidths=linewidths,
                    robust=robust,
                    square=True,
                    annot=annot,
                    fmt="d",
                    cmap="BuGn",
                    cbar_kws={"shrink": .9})
        plt.title(title)
        if isinstance(save_path, str):
            plt.savefig(f"{save_path}.png")
    
    return fig, ax





def colorbar(mappable):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


######################################################



def display_images(
                   images: List[Image.Image],
                   labels: Optional[List[str]]=None,
                   max_images: int=32,
                   columns: int=5,
                   width: int=20,
                   height: int=12,
                   label_wrap_length: int=50,
                   label_font_size: int="medium") -> None:

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    rows = int(len(images)/columns)
#     height = max(height, rows * height)
    plt.subplots(rows, columns, figsize=(width, height), sharex=True, sharey=True)
    for i, image in enumerate(images):

        plt.subplot(rows + 1, columns, i + 1)
        plt.imshow(image)
        ax = plt.gca()
        for label in ax.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in ax.yaxis.get_ticklabels():
            label.set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        title=None
        if isinstance(labels, list):
            title = labels[i]
        elif hasattr(image, 'filename'):
            title=image.filename
            
        if title:
#             if title.endswith("/"): title = title[:-1]
            title=Path(title).stem
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size); 

    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.9, wspace=0.0, hspace=0.2*rows) #wspace=0.05, hspace=0.1)
