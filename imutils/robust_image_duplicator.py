"""

Class definition of a RobustImageDuplicator, which executes an I/O-intensive image file duplication process.
Using a pandas dataframe as input, iterates through each row of the dataframe and copies the file located
at a `source_path` to a `target_path`, with each being a local file path (as opposed to a directory).


imutils/robust_image_duplicator.py
local path: `/media/data/jacob/GitHub/image-utils/imutils/robust_image_duplicator.py`

Author: Jacob A Rose
Created: Friday Dec 17th, 2021

"""
__all__ = ["RobustImageDuplicator"]

import logging
import os
import shutil

# from pathlib import Path
from typing import Union

# import modin.pandas as pd
import pandas as pd
import PIL
import PIL.Image

try:
    import swifter
except ImportError:
    pass
from tqdm.auto import tqdm

##############################################

level = logging.INFO
log = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(level)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)

##############################################


class RobustImageDuplicator:
    """
    Tools for duplicating a dataset of image files to a new target location on disk.
    Automatically checks for and attempts to repair corrupted images due to interupted processes.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        source_col: str = "source_path",
        target_col: str = "target_path",
        log_every: int = 200,
        use_swifter: bool = True,
    ):
        self.data = data
        self.source_col = source_col
        self.target_col = target_col

        self.corrupted_files = []

        self.log_every = log_every
        self.use_swifter = use_swifter
        tqdm.pandas(desc=f"copying {self.data.shape[0]} files")

    def step(self, row: pd.Series) -> bool:
        if self.imcopy(row[self.source_col], row[self.target_col]):
            return True
        return False

    def run(self):
        if self.use_swifter:
            results = self.data.swifter.progress_bar(
                enable=True,
                desc=f"duplicating {self.data.shape[0]} records",
            ).apply(self.step, axis=1)
        else:
            results = self.data.progress_apply(self.step, axis=1)

        correct = results.sum()
        incorrect = (~results).sum()

        log.info(f"# correct: {correct}, # incorrect: {incorrect}")
        return correct, incorrect

    def imread(self, path: str) -> Union[str, PIL.Image.Image]:
        try:
            img = PIL.Image.open(path)
            img.load()
            return img
        except OSError:
            self.corrupted_files.append(path)
        return str(path)

    def iscorrupted(self, path: str) -> bool:
        if isinstance(self.imread(path), str):
            return True
        else:
            return False

    def imcopy(self, source_path: str, target_path: str) -> bool:
        if os.path.isfile(target_path):
            if not self.iscorrupted(target_path):
                # Check if image at target_path can be loaded without error, skip shutil copy execution if true. O/w, Perform the copy operation if a corrupted target image is detected.
                return True
            else:
                log.info(
                    f"Found corrupted target file, attempting to perform original copy operation."
                )
        shutil.copy2(source_path, target_path)
        return not self.iscorrupted(target_path)


if __name__ == "__main__":

    local_catalog = pd.DataFrame.from_records(
        [
            {"source_path": "", "target_path": ""},
            {"source_path": "", "target_path": ""},
        ]
    )
    image_duplicator = RobustImageDuplicator(data=local_catalog, log_every=1, use_swifter=True)

    correct, incorrect = image_duplicator.run()

    print(f"# correct: {correct}, # incorrect: {incorrect}")
