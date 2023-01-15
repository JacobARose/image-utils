"""

image-utils/tests/unit/test_datamodule.py

Created by: Jacob A Rose
Created on: Wednesday May 18th, 2022

"""

import os
from pathlib import Path

import pandas as pd
import pytest
# from imutils import ASSETS_DIR, SAMPLE_IMAGE_PATHS
from imutils.ml.data.datamodule import *


@pytest.fixture
def sample_image_paths():
    return SAMPLE_IMAGE_PATHS


@pytest.fixture
def sample_catalog(sample_image_paths, tmpdir):
    assert len(sample_image_paths) == 2
    assert all([os.path.isfile(path) for path in sample_image_paths])
    print(f"tmpdir: {tmpdir}")
    return pd.DataFrame.from_records(
        [
            {
                "source_path": str(sample_image_paths[0]),
                "target_path": Path(tmpdir) / sample_image_paths[0].name,
            },
            {
                "source_path": str(sample_image_paths[1]),
                "target_path": Path(tmpdir) / sample_image_paths[1].name,
            },
        ]
    )


def test_sample_catalog(sample_catalog):

    assert sample_catalog.source_path.apply(os.path.isfile).all()
    assert (~sample_catalog.target_path.apply(os.path.isfile)).all()
    print(sample_catalog)


def test_RobustImageDuplicator_run(sample_catalog):

    image_duplicator = RobustImageDuplicator(data=sample_catalog, log_every=1, use_swifter=False)
    correct, incorrect = image_duplicator.run()
    assert sample_catalog.target_path.apply(os.path.isfile).all()
    print(f"# correct: {correct}, # incorrect: {incorrect}")
