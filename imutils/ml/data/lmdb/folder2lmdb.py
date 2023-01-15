"""

imutils/data/lmdb/folder2lmdb.py


Created on: Sunday April 18th, 2022  
Adapted by: Jacob Alexander Rose  


Based on source: https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/data/lmdb_dataset.py
"""

from imutils.ml.data.lmdb.convert_lmdb_tools import load_pyarrow, dumps_pyarrow, get_key, decode_img_from_buf


import os
import lmdb
from imutils.ml.data.lmdb.convert_lmdb_tools import decode_img_from_buf
from torch.utils.data import Dataset
import torch
from typing import *
import numpy as np
import PIL
# import PIL.Image.Image


class ImageLMDBDataset(Dataset):
	"""
	LMDB format for image folder.
	"""
	def __init__(self, db_dir, subset: str="train", transform=None, target_transform=None, backend='cv2'):
		self.subset = subset
		self.db_path = os.path.join(db_dir, f'{subset}.lmdb')
		self.env = lmdb.open(self.db_path,
							 subdir=False,
							 readonly=True,
							 lock=False,
							 readahead=False,
							 meminit=False)
		with self.env.begin() as txn:
			self.length = load_pyarrow(txn.get(b'__len__'))
			try:
				self.classes = load_pyarrow(txn.get(b'classes'))
				self.class2idx = load_pyarrow(txn.get(b'class2idx'))
				self.taxonomy = load_pyarrow(txn.get(b'taxonomy'))
			except AssertionError:
				pass

		self.map_list = [get_key(i) for i in range(self.length)]
		self.transform = transform
		self.target_transform = target_transform
		self.backend = backend
		self.output_image_type = torch.Tensor
		self.output_image_range = (0,1)

	def __len__(self):
		return self.length

	def __getitem__(self, item):
		with self.env.begin() as txn:
			byteflow = txn.get(self.map_list[item])
		unpacked = load_pyarrow(byteflow)
		# import pdb; pdb.set_trace()
		img, target, metadata = unpacked[0]
		# img = decode_img_from_buf(imgbuf, self.backend)

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)

		img = self.parse_output_image(img)
		return img, target, metadata


	def parse_output_image(self, img: Any):
		if isinstance(img, self.output_image_type):
			return img
		if self.output_image_type == torch.Tensor:
			if isinstance(img, np.ndarray):
				if img.dtype == "uint8":
					img = img.astype("float32")
				if np.allclose(img.max(), 255.0):
					img = img / 255.0
				return torch.from_numpy(img).permute(2,0,1)
			elif isinstance(img, PIL.Image.Image):
				return T.ToTensor()(img)
		elif self.output_image_type == np.ndarray:
			if isinstance(img, torch.Tensor):
				img = img.permute(1,2,0).numpy()
			elif isinstance(img, PIL.Image.Image):
				img = np.array(img)
			
			if self.output_image_range == (0,1):
				if img.dtype == "uint8":
					img = img.astype("float32")
				if np.allclose(img.max(), 255.0):
					img = img / 255.0
			return img

		else:
			raise Exception(f"Warning, parse_output_image received unexpected image of type {type(img)=}")


# class ImageFolderLMDB(data.Dataset):
#	 def __init__(self, db_path, transform=None, target_transform=None):
#		 self.db_path = db_path
#		 self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
#							  readonly=True, lock=False,
#							  readahead=False, meminit=False)
#		 with self.env.begin(write=False) as txn:
#			 self.length = load_pyarrow(txn.get(b'__len__'))
#			 self.keys = load_pyarrow(txn.get(b'__keys__'))

#		 self.transform = transform
#		 self.target_transform = target_transform

#	 def __getitem__(self, index):
#		 env = self.env
#		 with env.begin(write=False) as txn:
#			 byteflow = txn.get(self.keys[index])

#		 unpacked = load_pyarrow(byteflow)

#		 # load img
#		 imgbuf = unpacked[0]
#		 buf = six.BytesIO()
#		 buf.write(imgbuf)
#		 buf.seek(0)
#		 img = Image.open(buf).convert('RGB')

#		 # load label
#		 target = unpacked[1]

#		 if self.transform is not None:
#			 img = self.transform(img)

#		 im2arr = np.array(img)

#		 if self.target_transform is not None:
#			 target = self.target_transform(target)

#		 # return img, target
#		 return im2arr, target

#	 def __len__(self):
#		 return self.length

#	 def __repr__(self):
#		 return self.__class__.__name__ + ' (' + self.db_path + ')'
