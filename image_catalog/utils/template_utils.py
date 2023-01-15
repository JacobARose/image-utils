"""

image-utils/image_catalog/utils/template_utils.py



"""

import logging
import warnings
from typing import List, Sequence, Union
import os
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=logging.INFO):
	"""Initializes python logger."""

	logger = logging.getLogger(name)
	
	handler = logging.StreamHandler()
	formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	
	logger.setLevel(level)

	# this ensures all logging levels get marked with the rank zero decorator
	# otherwise logs would get multiplied for each GPU process in multi-GPU setup
	for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
		setattr(logger, level, rank_zero_only(getattr(logger, level)))

	return logger


log = get_logger()

