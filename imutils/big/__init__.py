"""

imutils/big/__init__.py

"""

# from ..ml.utils.common import load_envs
# load_envs()

import dotenv
import os
import logging


dotenv.load_dotenv(dotenv_path=os.path.join(
	os.path.dirname(__file__), "default.env"),
				   override=True)

if not os.path.exists(os.environ["HERBARIUM_ROOT_DEFAULT"]):
	dotenv.load_dotenv(dotenv_path=os.path.join(
		os.path.dirname(__file__), ".env"),
					   override=True)
	
logger = logging.getLogger(__name__) #__file__)
logger.info(f"Using HERBARIUM_ROOT_DEFAULT location: {os.environ['HERBARIUM_ROOT_DEFAULT']}")
