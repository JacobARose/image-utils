"""


"""

import os
from pathlib import Path
from PIL import Image
import multiprocessing as mproc
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import *

__all__ = ["Herbarium2022DataModule", "Herbarium2022Dataset"]


def get_metadata(label_col="scientificName",
                 train_size=0.7,
                 seed=14):
    
    from imutils.big.make_train_val_splits import main as make_train_val_splits

    DATA_DIR = "/media/data/jacob/GitHub/image-utils/notebooks/herbarium_2022/data"
    split_dir = Path(DATA_DIR, "splits", f"train_size-{train_size}")

    encoder, train_data, val_data, test_data = make_train_val_splits(source_dir=DATA_DIR,
                                                                     split_dir=split_dir,
                                                                     label_col=label_col,
                                                                     train_size=train_size,
                                                                     seed=seed)
    return encoder, train_data, val_data, test_data
    

class Herbarium2022Dataset(Dataset):
    def __init__(self,
                 subset: str="train",
                 label_col="scientificName",
                 train_size=0.7,
                 shuffle: bool=True,
                 seed=14,
                 transform=None):
        
        self.x_col = "path"
        self.y_col = "y"
        self.id_col = "image_id"
        
        encoder, train_data, val_data, test_data = get_metadata(label_col=label_col,
                                                                train_size=train_size,
                                                                seed=seed)

        self.subset = subset
        self.encoder = encoder
        # self.train_df, self.val_df, self.test_df = train_data, val_data, test_data
        
        self.is_supervised = True
        if subset == 'train':
            self.df = train_data
        elif subset == "val":
            self.df = val_data
        else:
            self.df = test_data
            self.is_supervised = False

        if shuffle:
            self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=False)
            
        if self.is_supervised:
            self.num_classes = len(set(self.df[self.y_col]))
            
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def parse_sample(self, index: int):
        return self.df.iloc[index, :]
        
    def fetch_item(self, index: int) -> Tuple[str]:
        """
        Returns identically-structured namedtuple as __getitem__, with the following differences:
            - PIL Image w/o any transforms vs. torch.Tensor after all transforms
            - target text label vs, target int label
            - image path
            - image catalog_number
        
        """
        sample = self.parse_sample(index)
        
        path = getattr(sample, self.x_col)
        catalog_number = getattr(sample, self.id_col)
        
        image = Image.open(path)
        metadata={
                  "path":path,
                  "catalog_number":catalog_number
                 }
        
        if self.is_supervised:
            label = getattr(sample, self.y_col)
            return image, label #, metadata
        
        return image, metadata
    

    def __getitem__(self, index: int):
        
        item = self.fetch_item(index)

        if self.is_supervised:
            image, target, metadata = item
        else:
            image, metadata = item
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.is_supervised:
            return image, target, metadata        
        return image, metadata

    
from torchvision import transforms as T

def get_default_transforms(subset: str="train"):
    
    if subset == "train":
        return T.Compose([
            T.Resize(512),
            T.RandomPerspective(),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # T.Normalize([0.431, 0.498,  0.313], [0.237, 0.239, 0.227]),  # custom
        ])

    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # T.Normalize([0.431, 0.498,  0.313], [0.237, 0.239, 0.227]),  # custom
    ])
    
    
    


class Herbarium2022DataModule(pl.LightningDataModule):
    dataset_cls = Herbarium2022Dataset
    def __init__(self,
                 label_col="scientificName",
                 train_size=0.7,
                 shuffle: bool=True,
                 seed=14,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 batch_size: int = 128,
                 num_workers: int = None
    ):
        super().__init__()
                
        self.label_col = label_col
        self.train_size = train_size
        self.shuffle = shuffle
        self.seed = seed
        self.train_transform = get_default_transforms(subset="train") if train_transform is None else train_transform
        self.val_transform = get_default_transforms(subset="val") if val_transform is None else val_transform
        self.test_transform = get_default_transforms(subset="test") if test_transform is None else test_transform

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else mproc.cpu_count()

    def prepare_data(self):
        pass

    @property
    def num_classes(self) -> int:
        assert self.train_dataset and self.valid_dataset
        return max(self.train_dataset.num_classes, self.valid_dataset.num_classes)

    def setup(self, stage=None):
        
        self.train_dataset = Herbarium2022Dataset(subset="train",
                                                  label_col=self.label_col,
                                                  train_size=self.train_size,
                                                  shuffle=self.shuffle,
                                                  seed=self.seed,
                                                  transform=self.train_transform)
        self.val_dataset = Herbarium2022Dataset(subset="val",
                                                  label_col=self.label_col,
                                                  train_size=self.train_size,
                                                  shuffle=self.shuffle,
                                                  seed=self.seed,
                                                  transform=self.val_transform)
        self.test_dataset = Herbarium2022Dataset(subset="test",
                                                 label_col=self.label_col,
                                                 train_size=self.train_size,
                                                 shuffle=self.shuffle,
                                                 seed=self.seed,
                                                 transform=self.test_transform)
        
        print(f"training dataset: {len(self.train_dataset)}")
        print(f"validation dataset: {len(self.val_dataset)}")
        print(f"test dataset: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
