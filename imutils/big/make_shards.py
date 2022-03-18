"""
imutils/big/make_shards.py

Generate one or more webdataset-compatible tar archive shards from an image classification dataset.

Based on script: https://github.com/tmbdev-archive/webdataset-examples/blob/7f56e9a8b978254c06aa0a98572a1331968b0eb3/makeshards.py

Added on: Sunday March 6th, 2022

Example usage:

python "/media/data/jacob/GitHub/image-utils/imutils/big/make_shards.py" \
	--subsets=train,val,test \
	--maxsize='1e9' \
	--maxcount=50000 \
	--shard_dir="/media/data_cifs/projects/prj_fossils/users/jacob/data/herbarium_2022/webdataset" \
	--catalog_dir="/media/data_cifs/projects/prj_fossils/users/jacob/data/herbarium_2022/catalog" \
	--debug


"""


import sys
import os
import os.path
import random
import argparse

from torchvision import datasets

import webdataset as wds


import numpy as np
import os
from typing import Optional, Tuple, Any, Dict
from tqdm import trange, tqdm

import tarfile
tarfile.DEFAULT_FORMAT = tarfile.GNU_FORMAT

import webdataset as wds
# from imutils.big.datamodule import Herbarium2022DataModule, Herbarium2022Dataset
from imutils.ml.data.datamodule import Herbarium2022DataModule, Herbarium2022Dataset

def read_file_binary(fname):
	"Read a binary file from disk."
	with open(fname, "rb") as stream:
		return stream.read()

all_keys = set()

def prepare_sample(dataset, index, subset: str="train", filekey: bool=False) -> Dict[str, Any]:
	image_binary, label, metadata = dataset[index]

	key = metadata["catalog_number"]
	assert key not in all_keys
	all_keys.add(key)

	xkey = key if filekey else "%07d" % index
	sample = {"__key__": xkey, 
			  "image.jpg": image_binary}
	
	if subset != "test":
		assert label == dataset.targets[index]
		sample["label.cls"] = int(label)
	
	return sample

	
def write_dataset(catalog_dir: Optional[str]=None,
				  shard_dir: Optional[str]=None,
				  subset="train",
				  maxsize=1e9,
				  maxcount=100000,
				  limit_num_samples: Optional[int]=np.inf,
				  filekey: bool=False,
				  dataset=None):

	if dataset is None:
		datamodule = Herbarium2022DataModule(catalog_dir=catalog_dir,
											 num_workers=4,
											 image_reader=read_file_binary,
											 remove_transforms=True)
		datamodule.setup()
		dataset = datamodule.get_dataset(subset=subset)
	
	num_samples = len(dataset)
	print(f"With subset={subset}, Total num_samples: {num_samples}")
	
	if limit_num_samples < num_samples:
		num_samples = limit_num_samples
		print(f"Limiting this run to num_samples: {num_samples}")
	indices = list(range(num_samples))

	os.makedirs(shard_dir, exist_ok=True)
	pattern = os.path.join(shard_dir, f"herbarium_2022-{subset}-%06d.tar")

	with wds.ShardWriter(pattern, maxsize=maxsize, maxcount=maxcount) as sink:
		for i in tqdm(indices, desc=f"idx(Total={num_samples})"):
			
			sample = prepare_sample(dataset, index=i, subset=subset, filekey=filekey)
			sink.write(sample)


	return dataset, indices



def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser("""Generate sharded dataset from supervised image dataset.""")
	parser.add_argument("--subsets", default="train,val,test", help="which subsets to write")
	parser.add_argument(
		"--filekey", action="store_true", help="use file as key (default: index)"
	)
	parser.add_argument("--maxsize", type=float, default=1e9)
	parser.add_argument("--maxcount", type=float, default=100000)
	parser.add_argument(
		"--shard_dir",
		default="/media/data_cifs/projects/prj_fossils/users/jacob/data/herbarium_2022/webdataset",
		help="directory where shards are written"
	)
	parser.add_argument(
		"--catalog_dir",
		default="/media/data_cifs/projects/prj_fossils/users/jacob/data/herbarium_2022/catalog",
		help="directory containing csv versions of the original train & test metadata json files from herbarium 2022",
	)
	parser.add_argument("--debug",  action="store_true", default=False,
					   help="Provide this boolean flag to produce a debugging shard dataset of only a maximum of 200 samples per data subset. [TODO] Switch to temp directories when this flag is passed.")
	
	args = parser.parse_args()
	return args


def main(args):
	
	# args = parse_args()
	
	assert args.maxsize > 10000000 # Shards must be a minimum of 10+ MB
	assert args.maxcount < 1000000 # Shards must contain a maximum of 1,000,000 samples each

	limit_num_samples = 200 if args.debug else np.inf

# 	if not os.path.isdir(os.path.join(args.data, "train")):
# 		print(f"{args.data}: should be directory containing ImageNet", file=sys.stderr)
# 		print(f"suitable as argument for torchvision.datasets.ImageNet(...)", file=sys.stderr)
# 		sys.exit(1)

# 	if not os.path.isdir(os.path.join(args.shards, ".")):
# 		print(f"{args.shards}: should be a writable destination directory for shards", file=sys.stderr)
# 		sys.exit(1)

		
	subsets = args.subsets.split(",")

	for subset in tqdm(subsets, leave=True, desc=f"Processing {len(subsets)} subsets"):
		# print("# subset", subset)
		
		dataset, indices = write_dataset(catalog_dir=args.catalog_dir,
										 shard_dir=args.shard_dir,
 										 subset=subset,
										 maxsize=args.maxsize,
										 maxcount=args.maxcount,
										 limit_num_samples=limit_num_samples,
										 filekey=args.filekey)
		
		
CATALOG_DIR = "/media/data_cifs/projects/prj_fossils/users/jacob/data/herbarium_2022/catalog"
# SHARD_DIR = "/media/data_cifs/projects/prj_fossils/users/jacob/data/herbarium_2022/webdataset"

if __name__ == "__main__":
	
	args = parse_args()
	main(args)

	written_files = os.listdir(args.shard_dir)
	files_per_subset = {"train":[],
					    "val":[],
					    "test":[]}
	for subset,v in files_per_subset.items():
		files_per_subset[subset] = len([f for f in written_files if subset in f])
		
	from rich import print as pp
	print(f"SUCCESS! TARGET SHARD DIR CONTAINS THE FOLLOWING:")
	pp(files_per_subset)
