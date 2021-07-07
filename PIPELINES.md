Preprocessing
--------------

./bin/bob_dbmanage.py hqwmca download




SETSHELL grid
./bin/bob pad vanilla-pad \
-d /idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/config/DB/database_template.py \
-p /idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/config/Preprocessor/Preprocessor.py \
-f transform \
-g train -g dev -g eval \
-o /idiap/temp/ageorge/IJCB_ViT_PaperPackage/pipeline/ -vvv -l /idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/distributed/sge_big.py








Running Pipeline
------------------
Using DB and extractor (no preprocessor )




Run this and see

before the number of files was: 2,807 items, totalling 5.3 GB

new annotation folder was empty at the time of this:



# New trained models

./bin/bob pad vanilla-pad \
/idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/config/Method/Pipeline.py \
-o /idiap/temp/ageorge/IJCB_ViT_PaperPackage/pipeline-extractor/LOO_Makeup/ -vvv -l sge-gpu



./bin/bob pad metrics -e -c apcer100 -r attack /idiap/temp/ageorge/IJCB_ViT_PaperPackage/pipeline-extractor/LOO_Makeup/scores-{dev,eval}.csv

==============  ===============  ================
..              Development      Evaluation
==============  ===============  ================
APCER (attack)  9.9%             47.5%
APCER_AP        9.9%             47.5%
BPCER           0.9%             4.7%
ACER            5.4%             26.1%
FTA             0.0%             0.0%
FPR             9.9% (460/4630)  47.5% (485/1020)
FNR             0.9% (12/1290)   4.7% (77/1640)
HTER            5.4%             26.1%
FAR             9.9%             47.5%
FRR             0.9%             4.7%
PRECISION       0.7              0.8
RECALL          1.0              1.0
F1_SCORE        0.8              0.8
AUC             1.0              0.9
AUC-LOG-SCALE   3.4              1.5
==============  ===============  ================


# Previous weights

./bin/bob pad vanilla-pad \
/idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/config/Method/Pipeline.py \
-o /idiap/temp/ageorge/IJCB_ViT_PaperPackage/pipeline-extractor-old/LOO_Makeup/ -vvv -l sge-gpu

./bin/bob pad metrics -e -c apcer100 -r attack /idiap/temp/ageorge/IJCB_ViT_PaperPackage/pipeline-extractor-old/LOO_Makeup/scores-{dev,eval}.csv

==============  ================  ================
..              Development       Evaluation
==============  ================  ================
APCER (attack)  11.9%             46.9%
APCER_AP        11.9%             46.9%
BPCER           0.9%              4.5%
ACER            6.4%              25.7%
FTA             0.0%              0.0%
FPR             11.9% (550/4630)  46.9% (478/1020)
FNR             0.9% (12/1290)    4.5% (74/1640)
HTER            6.4%              25.7%
FAR             11.9%             46.9%
FRR             0.9%              4.5%
PRECISION       0.7               0.8
RECALL          1.0               1.0
F1_SCORE        0.8               0.9
AUC             1.0               0.9
AUC-LOG-SCALE   3.4               1.7
==============  ================  ================


# model copier




protocols=['grand_test-curated','LOO_Flexiblemask', 'LOO_Glasses', 'LOO_Makeup', 'LOO_Mannequin', 'LOO_Papermask','LOO_Print', 'LOO_Rigidmask', 'LOO_Tattoo','LOO_Replay']

import shutil

for protocol in protocols:
    src_path="/idiap/temp/ageorge/HQWMCA_RGBD/NEWPIPELINES/Baselines/VisionTransformerPAD224-L/GENERATED-CURATED/CNN/VisionTransformerPAD224-L_{}_SEED_0_LTK_4_Gamma_0_BIN_True_MULT_2/model_0_0.pth".format(protocol)
    dst_path='/idiap/temp/ageorge/IJCB_ViT_PaperPackage/models_hqwmca/'+protocol+'.pth'
    print(src_path,dst_path )
    shutil.copyfile(src_path,dst_path )



from bob.learn.pytorch.datasets import ChannelSelect, RandomHorizontalFlipImage

from torchvision import transforms

import numpy as np 

from bob.paper.ijcb2021_vision_transformer_pad.extractor import GenericExtractorMod

from bob.paper.ijcb2021_vision_transformer_pad.architectures import  ViTranZFAS

from bob.bio.base.transformers import ExtractorTransformer



# path to the model for the specific protocol

MODEL_FILE= '/idiap/temp/ageorge/IJCB_ViT_PaperPackage/models_hqwmca/LOO_Glasses.pth'#
####################################################################



SELECTED_CHANNELS = [0,1,2] 
####################################################################
_img_transform =transforms.Compose([ChannelSelect(selected_channels = SELECTED_CHANNELS),transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])



# function defining type of scoring, default is from the RGB-D branch
def extractor_function(output,kwargs):

  #print("scoring_method",kwargs['scoring_method'])

  scoring_method=kwargs['scoring_method']
  #gap, op_rgbd, op_rgb, op_d

  embedding = output[0].data.numpy().flatten()
  binary = output[1].data.numpy().flatten()


  if scoring_method=='binary':
    score=np.mean(binary)
  elif scoring_method=='embedding':
    score=embedding.copy()
  else:
    raise ValueError('Scoring method {} is not implemented.'.format(scoring_method))

  return score


network=ViTranZFAS(pretrained=True)

_image_extracor=GenericExtractorMod(network=network,extractor_function=extractor_function,transforms=_img_transform, extractor_file=MODEL_FILE,scoring_method='binary')



$ bdt dav makedirs -vvv data/bob/bob.paper.ijcb2021_vision_transformer_pad/
# check
$ bdt dav upload -vvv --checksum /idiap/temp/ageorge/IJCB_ViT_PaperPackage/models_hqwmca.tar.gz data/bob/bob.paper.ijcb2021_vision_transformer_pad/
# execute
$ ./bin/bdt dav upload -vvv --checksum /idiap/temp/ageorge/IJCB_ViT_PaperPackage/models_hqwmca_part1.tar.gz data/bob/bob.paper.ijcb2021_vision_transformer_pad/ --execute

$ ./bin/bdt dav upload -vvv --checksum /idiap/temp/ageorge/IJCB_ViT_PaperPackage/models/*.pth data/bob/bob.paper.ijcb2021_vision_transformer_pad/ --execute

Copied the models for all protocols