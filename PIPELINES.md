Preprocessing
--------------

./bin/bob_dbmanage.py hqwmca download




SETSHELL grid
./bin/bob pad vanilla-pad \
-d /idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/config/DB/database_template.py \
-p /idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/config/Preprocessor/Preprocessor.py \
-f transform \
-o /idiap/temp/ageorge/IJCB_ViT_PaperPackage/pipeline/ -vvv -l sge-gpu







Running Pipeline
------------------
Using DB and extractor (no preprocessor )




Run this and see

before the number of files was: 2,807 items, totalling 5.3 GB

new annotation folder was empty at the time of this:





./bin/bob pad vanilla-pad \
/idiap/user/ageorge/WORK/BATL_ENV_Bob9/src/bob.paper.ijcb2021_vision_transformer_pad/bob/paper/ijcb2021_vision_transformer_pad/config/Method/Pipeline.py \
-o /idiap/temp/ageorge/IJCB_ViT_PaperPackage/pipeline-extractor/LOO_Makeup/ -vvv -l sge


./bin/bob pad metrics -e -c apcer100 -r attack /idiap/temp/ageorge/CVPR_CMFL_PaperPackage/pipeline-ported/LOO_Makeup2/scores-{dev,eval}


./bin/bob pad metrics -e -c apcer100 -r attack /idiap/temp/ageorge/CVPR_CMFL_PaperPackage/pipeline-ported/LOO_Makeup/scores-{dev,eval}


