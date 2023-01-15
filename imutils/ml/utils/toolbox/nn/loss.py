# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['LabelSmoothingLoss', 'CBCrossEntropyLoss', 'SigmoidCrossEntropy', 'FocalLoss', 'L0Loss', 'RingLoss', 'CenterLoss', 'CircleLoss']

from imutils.ml.utils.toolbox.nn import functional as BF
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch
from typing import Sequence


class SigmoidCrossEntropy(_WeightedLoss):
	def __init__(self, classes, weight=None, reduction='mean'):
		super(SigmoidCrossEntropy, self).__init__(weight=weight, reduction=reduction)
		self.classes = classes

	def forward(self, pred, target):
		zt = BF.logits_distribution(pred, target, self.classes)
		return BF.logits_nll_loss(-F.logsigmoid(zt), target, self.weight, self.reduction)


class FocalLoss(_WeightedLoss):
	def __init__(self,
				 num_classes: int,
				 gamma,
				 weight=None,
				 reduction='mean'):
		super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
		self.num_classes = num_classes
		self.gamma = gamma

	def forward(self, pred, target):
		zt = BF.logits_distribution(pred, target, self.num_classes)
		ret = -(1 - torch.sigmoid(zt)).pow(self.gamma) * F.logsigmoid(zt)
		return BF.logits_nll_loss(ret, target, self.weight, self.reduction)


class L0Loss(nn.Module):
	"""L0loss from
	"Noise2Noise: Learning Image Restoration without Clean Data"
	<https://arxiv.org/pdf/1803.04189>`_ paper.

	"""
	def __init__(self, gamma=2, eps=1e-8):
		super(L0Loss, self).__init__()
		self.gamma = gamma
		self.eps = eps

	def forward(self, pred, target):
		loss = (torch.abs(pred - target) + self.eps).pow(self.gamma)
		return torch.mean(loss)


class LabelSmoothingLoss(nn.Module):
	"""This is label smoothing loss function.
	"""
	def __init__(self,
				 num_classes: int,
				 smoothing: float=0.0,
				 dim: int=-1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.num_classes = num_classes
		self.dim = dim

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		pred = pred.log_softmax(dim=self.dim)
		true_dist = BF.smooth_one_hot(target, self.num_classes, self.smoothing)
		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



class CBCrossEntropyLoss(nn.Module):
	"""This is a class implementation of the Class Balanced Cross Entropy Loss.
	
	Behavior:
		- beta=0.0 (default), results in the equivalent of unbalanced-CE, with weights=1 for all classes.
		- beta=1.0, results in the equivalent of class-balanced-CE, with all weights equal to the inverse frequency of their corresponding class.
		
	
	"""
	def __init__(self,
				 targets: Sequence[int],
				 beta: float=0.0,
				 reduction: str="mean"):
		super(CBCrossEntropyLoss, self).__init__()
		self.beta = beta
		self.reduction = reduction
		
		self.weights = torch.nn.Parameter(
			self.calculate_class_weights(targets=targets, beta=beta),
			requires_grad=False
		)
		
	def calculate_class_weights(self, targets: Sequence[int], beta: float=0.0) -> torch.Tensor:
		"""
		Provided a representative sequence of target labels + an interpolation parameter beta, produce a class-balanced sequence of weights, 1 for each class see in targets. Shouldn't normally be called directly by the user, as it's run once in the Loss class __init__ method.
		
		[TBD] -- Add in some kind of masking to allow calculating weights for sequences that have at least 1 class absent.
		"""
		self.classes, self.class_counts = BF.class_counts(targets)
		weights = BF.class_balanced_weight(beta=beta, samples_per_class=self.class_counts)
		# self.weights = torch.from_numpy(weights)
		# import pdb; pdb.set_trace()
		
		# with open("class_weights.csv", "w") as f:
		# 	f
		
		return weights

	def forward(self, y_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		
# 		print(f"y_logits.device: {y_logits.device}",
# 			  f"targets.device: {targets.device}",
# 			  f"self.weights.device: {self.weights.device}")
		
# 		import pdb; pdb.set_trace()
		
		loss = BF.class_balanced_cross_entropy_loss(y_logits,
													y_true=targets,
													weights=self.weights,
													reduction=self.reduction)
		
		
		
		return loss






class CircleLoss(nn.Module):
	r"""CircleLoss from
	`"Circle Loss: A Unified Perspective of Pair Similarity Optimization"
	<https://arxiv.org/pdf/2002.10857>`_ paper.

	Parameters
	----------
	m: float.
		Margin parameter for loss.
	gamma: int.
		Scale parameter for loss.

	Outputs:
		- **loss**: scalar.
	"""
	def __init__(self, m, gamma):
		super(CircleLoss, self).__init__()
		self.m = m
		self.gamma = gamma
		self.dp = 1 - m
		self.dn = m

	def forward(self, x, target):
		similarity_matrix = x @ x.T  # need gard here
		label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
		negative_matrix = label_matrix.logical_not()
		positive_matrix = label_matrix.fill_diagonal_(False)

		sp = torch.where(positive_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))
		sn = torch.where(negative_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))

		ap = torch.clamp_min(1 + self.m - sp.detach(), min=0.)
		an = torch.clamp_min(sn.detach() + self.m, min=0.)

		logit_p = -self.gamma * ap * (sp - self.dp)
		logit_n = self.gamma * an * (sn - self.dn)

		logit_p = torch.where(positive_matrix, logit_p, torch.zeros_like(logit_p))
		logit_n = torch.where(negative_matrix, logit_n, torch.zeros_like(logit_n))

		loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
		return loss


class RingLoss(nn.Module):
	"""Computes the Ring Loss from
	`"Ring loss: Convex Feature Normalization for Face Recognition"

	Parameters
	----------
	lamda: float
		The loss weight enforcing a trade-off between the softmax loss and ring loss.
	l2_norm: bool
		Whether use l2 norm to embedding.
	weight_initializer (None or torch.Tensor): If not None a torch.Tensor should be provided.

	Outputs:
		- **loss**: scalar.
	"""
	def __init__(self, lamda, l2_norm=True, weight_initializer=None):
		super(RingLoss, self).__init__()
		self.lamda = lamda
		self.l2_norm = l2_norm
		if weight_initializer is None:
			self.R = self.parameters(torch.rand(1))
		else:
			assert torch.is_tensor(weight_initializer), 'weight_initializer should be a Tensor.'
			self.R = self.parameters(weight_initializer)

	def forward(self, embedding):
		if self.l2_norm:
			embedding = F.normalize(embedding, 2, dim=-1)
		loss = (embedding - self.R).pow(2).sum(1).mean(0) * self.lamda * 0.5
		return loss


class CenterLoss(nn.Module):
	"""Computes the Center Loss from
	`"A Discriminative Feature Learning Approach for Deep Face Recognition"
	<http://ydwen.github.io/papers/WenECCV16.pdf>`_paper.
	Implementation is refer to
	'https://github.com/lyakaap/image-feature-learning-pytorch/blob/master/code/center_loss.py'

	Parameters
	----------
	classes: int.
		Number of classes.
	embedding_dim: int
		embedding_dim.
	lamda: float
		The loss weight enforcing a trade-off between the softmax loss and center loss.

	Outputs:
		- **loss**: loss tensor with shape (batch_size,). Dimensions other than
		  batch_axis are averaged out.
	"""
	def __init__(self, classes, embedding_dim, lamda):
		super(CenterLoss, self).__init__()
		self.lamda = lamda
		self.centers = nn.Parameter(torch.randn(classes, embedding_dim))

	def forward(self, embedding, target):
		expanded_centers = self.centers.index_select(0, target)
		intra_distances = embedding.dist(expanded_centers)
		loss = self.lamda * 0.5 * intra_distances / target.size()[0]
		return loss


class KnowledgeDistillationLoss(nn.Module):
	def __init__(self, temperature=1):
		super().__init__()
		self.temperature = temperature

	def forward(self, student_output, teacher_output):
		return self.temperature**2 * torch.mean(
			torch.sum(-F.softmax(teacher_output / self.temperature) * F.log_softmax(student_output / self.temperature), dim=1))
