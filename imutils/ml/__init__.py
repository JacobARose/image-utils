"""

imutils/ml/__init__.py

"""

import os

IMUTILS_ML_ROOT = os.path.dirname(__file__)
BASE_ML_CONF_PATH = os.path.join(IMUTILS_ML_ROOT, "conf/base_conf.yaml")

print(f"IMUTILS_ML_ROOT: {IMUTILS_ML_ROOT}")

from rich import print as pp
pp(dir())