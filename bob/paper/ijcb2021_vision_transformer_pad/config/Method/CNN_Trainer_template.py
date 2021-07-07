from torchvision import transforms

from bob.learn.pytorch.datasets import ChannelSelect, RandomHorizontalFlipImage

import torch.nn as nn

import torch.optim as optim

import numpy as np

from torch.autograd import Variable

from bob.extension import rc

import bob.ip.stereo

from bob.io.stream import Stream

from sklearn.preprocessing import Normalizer

import bob.core

logger = bob.core.log.setup("bob.paper.ijcb2021_vision_transformer_pad")

import numpy as np

from bob.io.stream import Stream

from bob.learn.pytorch.datasets import DataFolder

from bob.paper.ijcb2021_vision_transformer_pad.architectures import ViTranZFAS

from bob.paper.ijcb2021_vision_transformer_pad.datasets import DataFolder



color = Stream('color')

streams = { 'color'     : color}


#==============================================================================
# Load the dataset

""" The steps are as follows

1. Initialize a databae instance, with the protocol, groups and number of frames 
	(currently for the ones in 'bob.pad.face', and point 'data_folder_train' to the preprocessed directory )	
	Note: Here we assume that we have already preprocessed the with `spoof.py` script and dumped it to location 
	pointed to by 'data_folder_train'.

2. Specify the transform to be used on the images. It can be instances of `torchvision.transforms.Compose` or custom functions.

3. Initialize the `data_folder` class with the database instance and all other parameters. This dataset instance is used in
 the trainer class

4. Initialize the network architecture with required arguments.

5. Define the parameters for the trainer. 

"""

#==============================================================================
# Initialize the bob database instance 

PREPROCESSED_DIR= {{PREPROCESSED_DIR}}

CNN_OUTPUT_DIR= {{CNN_OUTPUT_DIR}}

ANNOTATIONS_DIR={{ANNOTATIONS_DIR}}

ORIGINAL_EXTENSION='.h5'

train_groups=['train']

val_groups=['dev'] 

DO_CROSS_VALIDATION=True

PROTOCOL= {{PROTOCOL}}


from bob.extension import rc as _rc
from bob.paper.ijcb2021_vision_transformer_pad.database import HQWMCAPadDatabase
database = HQWMCAPadDatabase(protocol=PROTOCOL,
                             original_directory=PREPROCESSED_DIR,
                             original_extension=ORIGINAL_EXTENSION,
                             annotations_dir = ANNOTATIONS_DIR,
                             streams=streams,
                             n_frames=10)




USE_GPU=False


if DO_CROSS_VALIDATION:
	phases=['train','val']
else:
	phases=['train']

groups={"train":train_groups,"val":val_groups}



#==============================================================================
# Initialize the torch dataset, subselect channels from the pretrained files if needed.

SELECTED_CHANNELS = [0,1,2] 
####################################################################


def custom_function(np_img,target):

	img={}
	img['image']=np_img

	if target==1:
		mask=np.ones((14,14),dtype='float')*0.9
	else:
		mask=np.ones((14,14),dtype='float')*0.1

	labels={}

	labels['pixel_mask']=mask
	labels['binary_target']=target
	
	return img, labels


img_transform={}

img_transform['train'] = transforms.Compose([ChannelSelect(selected_channels = SELECTED_CHANNELS),transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

img_transform['val'] = transforms.Compose([ChannelSelect(selected_channels = SELECTED_CHANNELS),transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

data_balance={"train":True,"val":True}


dataset={}

for phase in phases:

	dataset[phase] = DataFolder(data_folder=PREPROCESSED_DIR,
						 transform=img_transform[phase],
						 extension='.h5',
						 bob_hldi_instance=database,
						 groups=groups[phase],
						 protocol=PROTOCOL,
						 purposes=['real', 'attack'],
						 allow_missing_files=True, custom_func=custom_function)



#==============================================================================
# Specify other training parameters

NUM_CHANNELS = len(SELECTED_CHANNELS)

####################################################################
do_crossvalidation=DO_CROSS_VALIDATION
batch_size = 16
num_workers = 4
epochs=20
learning_rate=0.0001
weight_decay = 0.000001
seed = 3
use_gpu = False
save_interval = 20
verbose = 2
training_logs= CNN_OUTPUT_DIR+'/train_log_dir/'
output_dir = CNN_OUTPUT_DIR


assert(len(SELECTED_CHANNELS)==NUM_CHANNELS)

#==============================================================================
# Load the architecture

network=ViTranZFAS(pretrained=True)

# set trainable parameters, here the MLP part is only retrained, modify the following section
# to change which layers are trained


for name,param in  network.named_parameters():
	param.requires_grad = False
	# print(name)

	if 'head' in name:
		# print('grad',name)
		param.requires_grad = True


# loss definitions

criterion_bce= nn.BCELoss()


# optimizer initialization

optimizer = optim.Adam(filter(lambda p: p.requires_grad, network.parameters()),lr = learning_rate, weight_decay=weight_decay)




							
def compute_loss(network,img, labels, device):
	"""
	Compute the losses, given the network, data and labels and 
	device in which the computation will be performed. 
	"""

	imagesv = Variable(img['image'].to(device))

	labelsv_pixel = Variable(labels['pixel_mask'].to(device))

	labelsv_binary = Variable(labels['binary_target'].to(device))

	out = network(imagesv)

	loss_bce = criterion_bce(out[1],labelsv_binary.unsqueeze(1).float())
						
	return loss_bce



