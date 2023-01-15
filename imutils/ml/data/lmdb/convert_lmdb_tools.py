# -*- coding: utf-8 -*-
"""

imutils/data/lmdb/folder2lmdb.py


Added on: Sunday April 18th, 2022  
Adapted by: Jacob Alexander Rose  

Original author: DevinYang (pistonyang@gmail.com)

"""



__all__ = ['get_key', 'load_pyarrow', 'dumps_pyarrow', 'generate_lmdb_dataset', 'raw_reader', 'decode_img_from_buf']

import lmdb
import os
import pyarrow as pa
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import *
from imutils.ml.utils.common import check_dir

# 1 TB
DEFAULT_MAP_SIZE = 1024**4
assert DEFAULT_MAP_SIZE == 1099511627776



# -*- coding: utf-8 -*-
# __all__ = ['decode_img_from_buf', 'pil_loader', 'cv_loader']

import cv2
import six
import numpy as np
from PIL import Image


def decode_img_from_buf(buf, backend='cv2'):
    if backend == 'pil':
        buf = six.BytesIO(buf)
        img = Image.open(buf).convert('RGB')
    elif backend == 'cv2':
        buf = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(buf, 1)[..., ::-1]
    else:
        raise NotImplementedError
    return img



def get_key(index):
	return u'{}'.format(index).encode('ascii')


def raw_reader(path):
	with open(path, 'rb') as f:
		bin_data = f.read()
	return bin_data


def dumps_pyarrow(obj: Any) -> ByteString:
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()



def load_pyarrow(buf):
	assert buf is not None, 'buf should not be None.'
	return pa.deserialize(buf)


def generate_lmdb_dataset(dataset: Dataset,
						  save_dir: str,
						  name: str,
						  collate_fn: Callable=lambda x: x,
						  num_workers: int=0,
						  max_size_rate: float=1.0,
						  write_frequency: int=5000,
						  pbar_position: int=0):
	"""
	User provides a PyTorch Dataset-like instance that can be fed into a DataLoader, which is iterated through and saved into an LMDB.
	"""

	data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=collate_fn)
	num_samples = len(dataset)
	
	leave_pbar = bool(pbar_position == 0)
	
	check_dir(save_dir)
	lmdb_path = os.path.join(save_dir, f'{name}.lmdb')
	db = lmdb.open(lmdb_path,
				   subdir=False,
				   map_size=int(DEFAULT_MAP_SIZE * max_size_rate),
				   readonly=False,
				   meminit=True,
				   map_async=True)

	txn = db.begin(write=True)
	for idx, data in enumerate(tqdm(data_loader, position=pbar_position, leave=leave_pbar)):
		txn.put(get_key(idx),
				dumps_pyarrow(data),
				overwrite=False)
		
		if idx % write_frequency == 0 and idx > 0:
			txn.commit()
			txn = db.begin(write=True)

	txn.put(b'__len__', dumps_pyarrow(num_samples))
	try:
		if hasattr(dataset, "classes"):
			classes = dataset.classes
			txn.put(b'classes', dumps_pyarrow(classes))
		
		if hasattr(dataset, "class2idx"):
			class2idx = dataset.class2idx
			txn.put(b'class2idx', dumps_pyarrow(class2idx))
		
		if hasattr(dataset, "taxonomy"):
			txn.put(b'taxonomy', dumps_pyarrow(dataset.taxonomy.to_pyarrow()))
		
	except AttributeError:
		pass

	txn.commit()
	db.sync()
	db.close()