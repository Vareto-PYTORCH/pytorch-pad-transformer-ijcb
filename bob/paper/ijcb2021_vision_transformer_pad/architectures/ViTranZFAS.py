import torch
from torch import nn
from torchvision import models
import numpy as np
import copy

import timm

from functools import partial
from timm.models.vision_transformer import VisionTransformer, _conv_filter

from timm.models.vision_transformer import default_cfgs

from timm.models.helpers import load_pretrained


# This works with pip install timm==0.3.4 for this to work with the pretrained stuff

class ViTranZFAS(nn.Module):

    """ ViT model extended for PAD 


    Attributes
    ----------
    pretrained: bool
        If set to `True` uses the pretrained DenseNet model as the base. If set to `False`, the network
        will be trained from scratch. 
        default: True    
    """

    def __init__(self, pretrained=True):
        """ Init function

        Parameters
        ----------
        pretrained: bool
            If set to `True` uses the pretrained densenet model as the base. Else, it uses the default network
            default: True
        """
        super(ViTranZFAS, self).__init__()
        self.vit = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vit.default_cfg = default_cfgs['vit_base_patch16_224']
        load_pretrained(self.vit, num_classes=1000, in_chans=3, filter_fn=_conv_filter)
        self.sigmoid = nn.Sigmoid()
        self.vit.head=nn.Linear(768,1,True)
    
    def forward(self, x):
        """ Propagate data through the network

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 
          The data to forward through the network. Expects RGB image of size 3x224x224

        Returns
        -------
        feat: :py:class:`torch.Tensor` 
            Embedding
        op: :py:class:`torch.Tensor`
            Final binary score.  

        """
        feat = self.vit.forward_features(x)

        op = self.vit.head(feat)

        op=self.sigmoid(op)
 
        return feat, op




# input=torch.randn(10, 3,224,224, dtype=torch.float)
# model=ViTranZFAS(pretrained=True)
# y=model(input)
# print(y[0].shape)
# print(y[1].shape)


