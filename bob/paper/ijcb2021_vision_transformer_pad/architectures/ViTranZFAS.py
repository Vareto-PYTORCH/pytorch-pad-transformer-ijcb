import torch
from torch import nn
from torchvision import models
import numpy as np
import copy

from bob.pad.local.architectures.utils import SELayer
import timm

from functools import partial
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import default_cfgs

from timm.models.helpers import load_pretrained

# DONOT USE THIS
# use pip install timm==0.4.5
class ViTranZFAS(nn.Module):

    """ Deep and Dense MultiChannel PAD
    Attack Detection:

    More extension of the stuff, with Bilinear Pooling

    This extends the following paper to multi-channel/ multi-spectral images with cross modal pretraining.

    Reference: Anjith George and Sébastien Marcel. "Deep Pixel-wise Binary Supervision for 
    Face Presentation Attack Detection." In 2019 International Conference on Biometrics (ICB).IEEE, 2019.

    The initialization uses `Cross modality pre-training` idea from the following paper:

    Wang L, Xiong Y, Wang Z, Qiao Y, Lin D, Tang X, Van Gool L. Temporal segment networks: 
    Towards good practices for deep action recognition. InEuropean conference on computer 
    vision 2016 Oct 8 (pp. 20-36). Springer, Cham.


    Attributes
    ----------
    pretrained: bool
        If set to `True` uses the pretrained DenseNet model as the base. If set to `False`, the network
        will be trained from scratch. 
        default: True 
    num_channels: int
        Number of channels in the input.      
    """

    def __init__(self, pretrained=True):
        """ Init function

        Parameters
        ----------
        pretrained: bool
            If set to `True` uses the pretrained densenet model as the base. Else, it uses the default network
            default: True
        num_channels: int
            Number of channels in the input. 
        """
        super(ViTranZFAS, self).__init__()

        # vit=timm.models.vit_base_patch16_384(pretrained=True)
        # Old


        self.vit = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vit.default_cfg = default_cfgs['vit_base_patch16_224']
        # load_pretrained(self.vit, num_classes=1000, in_chans=3, filter_fn=_conv_filter)

        #feat = list(vit.children())


        #self.enc = nn.Sequential(*feat[0:4])

        self.sigmoid = nn.Sigmoid()

        #self.linear=nn.Linear(768,1)
        self.vit.head=nn.Linear(768,1,True)
    
    def forward(self, x):
        """ Propagate data through the network

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 
          The data to forward through the network. Expects RGB image of size 3x224x224

        Returns
        -------
        dec: :py:class:`torch.Tensor` 
            Binary map of size 1x14x14
        op: :py:class:`torch.Tensor`
            Final binary score.  

        """
        feat = self.vit.forward_features(x)

        op = self.vit.head(feat)

        op=self.sigmoid(op)
 
        return feat, op




# input=torch.randn(10, 3,224,224, dtype=torch.float)

# model=VisionTransformerPAD224FE(pretrained=True)
# y=model(input)

# print(y[0].shape)

# print(y[1].shape)



# ## test

# input=torch.randn(10, 4,224,224, dtype=torch.float)

# model=ResBiDe2MCPAD(pretrained=True, num_channels=4, layers_to_keep=4)
# y=model(input)

# print(y[0].shape)
# print(y[1].shape)



# # dd=model.layer_dict["ch_0_dense"].state_dict()['0.weight']

# # dd=model.layer_dict["ch_0_dense"].state_dict()

# # dd['0.weight']=dd['0.weight']*2

# # model.layer_dict["ch_0_dense"].state_dict()['0.weight']=model.layer_dict["ch_0_dense"].state_dict()['0.weight']*2.0
# #model.layer_dict["ch_1_dense"].load_state_dict(dd)

# #model.layer_dict["ch_0_dense"].state_dict()['0.weight'][0,0,:,:]=0
# print(model.layer_dict["ch_0_dense"].state_dict()['0.weight'][0,0,:,:])

# print(model.layer_dict["ch_1_dense"].state_dict()['0.weight'][0,0,:,:])


# #model.layer_dict["ch_1_dense"].load_state_dict(model.layer_dict["ch_0_dense"].state_dict())


# y=model(input)

# print(y[0].shape)
# print(y[1].shape)

# import torch
# from torch import nn
# from torchvision import models
# import numpy as np
# import copy





# from bob.pad.local.architectures.utils import SELayer

# input=torch.randn(10, 3,224,224, dtype=torch.float)

# densenet = models.densenet161(pretrained=False)



# for layers_to_keep in range(1,10):

#         features = nn.Sequential(*list(resnet.children())[:layers_to_keep])

#         y=features(input)

#         print(layers_to_keep,y.shape)


