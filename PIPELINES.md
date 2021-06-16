Preprocessing
--------------

./bin/bob_dbmanage.py hqwmca download


SETSHELL grid
./bin/bob pad vanilla-pad \
-d /idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/config/DB/database_template.py \
-p /idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/config/Preprocessor/Preprocessor.py \
-f transform \
-o /idiap/temp/ageorge/IJCB_ViT_PaperPackage/pipeline/ -vvv -l sge



No assume it works, and then I have the preprocessed files ported from the old ones here:



/idiap/temp/ageorge/CVPR_CMFL_PaperPackage/new_video_like_container/




Running Pipeline
------------------
Using DB and extractor (no preprocessor )



SETSHELL grid
./bin/bob pad vanilla-pad \
/idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob.paper.ijcb2021_vision_transformer_pad/config/Method/Pipeline.py \
-f transform \
-o /idiap/temp/ageorge/CVPR_CMFL_PaperPackage/pipeline-ported/LOO_Makeup/ -vvv -c -l sge





Run this and see

before the number of files was: 2,807 items, totalling 5.3 GB

new annotation folder was empty at the time of this:


./bin/bob pad vanilla-pad \
/idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob.paper.ijcb2021_vision_transformer_pad/config/Method/Pipeline.py \
-o /idiap/temp/ageorge/CVPR_CMFL_PaperPackage/pipeline-ported/LOO_Makeup/ -vvv -l sge





./bin/bob pad vanilla-pad \
/idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob.paper.ijcb2021_vision_transformer_pad/config/Method/Pipeline.py \
-o /idiap/temp/ageorge/CVPR_CMFL_PaperPackage/pipeline-ported/LOO_Makeup2/ -vvv -l sge


./bin/bob pad metrics -e -c apcer100 -r attack /idiap/temp/ageorge/CVPR_CMFL_PaperPackage/pipeline-ported/LOO_Makeup2/scores-{dev,eval}



x-special/nautilus-clipboard
copy
sftp://ageorge@login.idiap.ch/idiap/temp/ageorge/CVPR_CMFL_PaperPackage/pipeline-ported/LOO_Makeup/scores-dev



./bin/bob pad metrics -e -c apcer100 -r attack /idiap/temp/ageorge/CVPR_CMFL_PaperPackage/pipeline-ported/LOO_Makeup/scores-{dev,eval}


Now, replace None with nan- this is done with bob.pad.base

==============  =============  ==============
..              Development    Evaluation
==============  =============  ==============
APCER (attack)  0.0%           68.9%
APCER_AP        0.0%           68.9%
BPCER           0.7%           3.9%
ACER            0.4%           36.4%
FTA             2.8%           3.5%
FPR             0.0% (0/510)   68.9% (84/122)
FNR             0.7% (1/142)   3.9% (7/181)
HTER            0.4%           36.4%
FAR             0.0%           66.4%
FRR             3.5%           7.2%
PRECISION       1.0            0.7
RECALL          1.0            1.0
F1_SCORE        1.0            0.8
AUC             1.0            0.8
AUC-LOG-SCALE   2.7            1.0
==============  =============  ==============
