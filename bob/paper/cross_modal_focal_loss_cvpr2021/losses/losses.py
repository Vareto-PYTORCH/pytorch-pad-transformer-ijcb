import torch
import torch.nn as nn
import numpy as np 

from torch.nn import functional as F

import torch.nn

from torch.autograd import Variable


class CMFL(nn.Module):
	"""
	Cross Modal Focal Loss

	"""

	def __init__(self, alpha=1, gamma=2, binary=True, multiplier=2):
		super(CMFL, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.binary = binary
		self.multiplier =multiplier

	def forward(self, inputs_a,inputs_b, targets):

		bce_loss_a = F.binary_cross_entropy(inputs_a, targets, reduce=False)
		bce_loss_b = F.binary_cross_entropy(inputs_b, targets, reduce=False)

		pt_a = torch.exp(-bce_loss_a)
		pt_b = torch.exp(-bce_loss_b)

		eps =0.000000001

		wt_a=((pt_b + eps)*(self.multiplier*pt_a*pt_b))/(pt_a + pt_b + eps)
		wt_b=((pt_b + eps)*(self.multiplier*pt_a*pt_a))/(pt_a + pt_b + eps)

		if self.binary:
			wt_a=wt_a * (1-targets)
			wt_b=wt_b * (1-targets)

		f_loss_a = self.alpha * (1-wt_a)**self.gamma * bce_loss_a
		f_loss_b = self.alpha * (1-wt_b)**self.gamma * bce_loss_b

		loss= 0.5*torch.mean(f_loss_a) + 0.5*torch.mean(f_loss_b) 
		
		return loss


