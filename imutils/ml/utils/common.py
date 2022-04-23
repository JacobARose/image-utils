"""

imutils/ml/utils/common.py

Created on: Wednesday March 16th, 2022  
Created by: Jacob Alexander Rose  


"""


import os
from typing import Any, Dict, List, Optional, Union, Sequence

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import math
import numpy as np
import torch.optim as optim


__all__ = ["check_dir", "remove_file",
           "get_env", "load_envs", 
           "render_images", "iterate_elements_in_batches",
           "TwoCropTransform", "AverageMeter"]



def check_dir(*path):
    """Check dir(s) exist or not, if not make one(them).
    Args:
        path: full path(s) to check.
    """
    for p in path:
        os.makedirs(p, exist_ok=True)


def remove_file(file_path: str, show_detail=False):
    if not os.path.exists(file_path):
        if show_detail:
            print(f'File {file_path} not exist.')
        return
    os.remove(file_path)





def get_env(env_name: str) -> str:
	"""
	Safely read an environment variable.
	Raises errors if it is not defined or it is empty.

	:param env_name: the name of the environment variable
	:return: the value of the environment variable
	"""
	if env_name not in os.environ:
		raise KeyError(f"{env_name} not defined")
	env_value: str = os.environ[env_name]
	if not env_value:
		raise ValueError(f"{env_name} has yet to be configured")
	return env_value


def load_envs(env_file: Optional[str] = None) -> None:
	"""
	Load all the environment variables defined in the `env_file`.
	This is equivalent to `. env_file` in bash.

	It is possible to define all the system specific variables in the `env_file`.

	:param env_file: the file that defines the environment variables to use. If None
					 it searches for a `.env` file in the project.
	"""
	dotenv.load_dotenv(dotenv_path=env_file, override=True)


def render_images(
	batch: torch.Tensor, nrow=8, title: str = "Images", autoshow: bool = True, normalize: bool = True
) -> np.ndarray:
	"""
	Utility function to render and plot a batch of images in a grid

	:param batch: batch of images
	:param nrow: number of images per row
	:param title: title of the image
	:param autoshow: if True calls the show method
	:return: the image grid
	"""
	image = (
		torchvision.utils.make_grid(
			batch.detach().cpu(), nrow=nrow, padding=2, normalize=normalize
		)
		.permute((1, 2, 0))
		.numpy()
		# .astype(np.uint8)
	)

	if autoshow:
		plt.figure(figsize=(8, 8))
		plt.axis("off")
		plt.title(title)
		plt.imshow(image)
		plt.show()
	return image


def iterate_elements_in_batches(
	outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
	batch_size: int,
	n_elements: int
) -> Dict[str, torch.Tensor]:
	"""
	Iterate over elements across multiple batches in order, independently to the
	size of each batch

	:param outputs: a list of outputs dictionaries
	:param batch_size: the size of each batch
	:param n_elements: the number of elements to iterate over

	:return: yields one element at the time
	"""
	if isinstance(outputs, dict):
		outputs = [outputs]
	count = 0
	for output in outputs:
		for i in range(batch_size):
			count += 1
			if count >= n_elements:
				return
			result = {}

			for key, value in output.items():
				if isinstance(value, int):
					result[key] = value
				elif isinstance(value, List):
					result[key] = value[i]
				elif len(value.shape) == 0:
					result[key] = value
				else:
					result[key] = value[i]
			yield result
			
			
			# yield {
			# 	key: value if len(value.shape) == 0 else value[i]
			# 	for key, value in output.items()
			# }








class TwoCropTransform:
	"""Create two crops of the same image"""
	def __init__(self, transform):
		self.transform = transform

	def __call__(self, x):
		return [self.transform(x), self.transform(x)]


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def adjust_learning_rate(args, optimizer, epoch):
	lr = args.learning_rate
	if args.cosine:
		eta_min = lr * (args.lr_decay_rate ** 3)
		lr = eta_min + (lr - eta_min) * (
				1 + math.cos(math.pi * epoch / args.epochs)) / 2
	else:
		steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
		if steps > 0:
			lr = lr * (args.lr_decay_rate ** steps)

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
	if args.warm and epoch <= args.warm_epochs:
		p = (batch_id + (epoch - 1) * total_batches) / \
			(args.warm_epochs * total_batches)
		lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr


def set_optimizer(opt, model):
	optimizer = optim.SGD(model.parameters(),
						  lr=opt.learning_rate,
						  momentum=opt.momentum,
						  weight_decay=opt.weight_decay)
	return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
	print('==> Saving...')
	state = {
		'opt': opt,
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': epoch,
	}
	torch.save(state, save_file)
	del state
