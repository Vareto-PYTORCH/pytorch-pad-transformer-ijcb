# =============================================================================
# define instance of the preprocessor:

from torchvision import transforms

import numpy as np 

from bob.extension import rc

from bob.learn.pytorch.datasets import ChannelSelect, RandomHorizontalFlipImage

from bob.paper.ijcb2021_vision_transformer_pad.preprocessor.FaceCropAlign import auto_norm_image as _norm_func

from bob.extension import rc

from bob.paper.ijcb2021_vision_transformer_pad.preprocessor import FaceCropAlign 

from bob.paper.ijcb2021_vision_transformer_pad.database import HQWMCAPadDatabase

from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector

from sklearn.pipeline import Pipeline

from bob.bio.base.transformers import PreprocessorTransformer, ExtractorTransformer

from bob.bio.video.transformer import VideoWrapper

from bob.paper.ijcb2021_vision_transformer_pad.preprocessor import FaceCropAlign, VideoFaceCropAlignBlockPatch

from bob.io.stream import Stream

from sklearn.preprocessing import Normalizer


import bob.core
logger = bob.core.log.setup("bob.learn.pytorch")
import numpy as np


# Database Initialize

color = Stream('color')

streams = { 'color'     : color}

PROTOCOL = {{PROTOCOL}}
ANNOTATIONS_DIR={{ANNOTATIONS_DIR}}

PREPROCESSED_DIR={{PREPROCESSED_DIR}}

from bob.extension import rc as _rc
from bob.paper.ijcb2021_vision_transformer_pad.database import HQWMCAPadDatabase
database = DatabaseConnector(HQWMCAPadDatabase(protocol=PROTOCOL,
                             original_directory=_rc['bob.db.hqwmca.directory'],
                             original_extension='.h5',
                             annotations_dir = ANNOTATIONS_DIR,
                             streams=streams,
                             n_frames=10))


protocol = PROTOCOL


groups = ["train", "dev", "eval"]


_channel_names = ['color']

_preprocessors={}

FACE_SIZE = 224  # The size of the resulting face
RGB_OUTPUT_FLAG = True  # BW output
USE_FACE_ALIGNMENT = True  # use annotations
MAX_IMAGE_SIZE = None  # no limiting here
FACE_DETECTION_METHOD = None#'mtcnn'  # use ANNOTATIONS
MIN_FACE_SIZE = 50  # skip small faces
ALIGNMENT_TYPE = 'default'

_image_preprocessor = FaceCropAlign(face_size=FACE_SIZE,
                                    rgb_output_flag=RGB_OUTPUT_FLAG,
                                    use_face_alignment=USE_FACE_ALIGNMENT,
                                    alignment_type=ALIGNMENT_TYPE,
                                    max_image_size=MAX_IMAGE_SIZE,
                                    face_detection_method=FACE_DETECTION_METHOD,
                                    min_face_size=MIN_FACE_SIZE)


_preprocessors[_channel_names[0]] = VideoWrapper(PreprocessorTransformer(_image_preprocessor))



preprocessor = PreprocessorTransformer(VideoFaceCropAlignBlockPatch(
        preprocessors=_preprocessors, channel_names=_channel_names, return_multi_channel_flag=True
    ))


preprocessor = bob.pipelines.wrap(
    ["sample"], preprocessor, transform_extra_arguments=(("annotations", "annotations"),),
)

preprocessor = bob.pipelines.CheckpointWrapper(
    preprocessor,
    features_dir=PREPROCESSED_DIR,
    load_func=bob.bio.video.VideoLikeContainer.load,
    save_func=bob.bio.video.VideoLikeContainer.save_function,
)

#====================================================================================
# Extractor

from bob.paper.ijcb2021_vision_transformer_pad.extractor import GenericExtractorMod

from bob.paper.ijcb2021_vision_transformer_pad.architectures import  ViTranZFAS

from bob.bio.base.transformers import ExtractorTransformer


MODEL_FILE= {{MODEL_FILE}}

SELECTED_CHANNELS = [0,1,2] 
####################################################################
_img_transform =transforms.Compose([ChannelSelect(selected_channels = SELECTED_CHANNELS),transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])



def extractor_function(output,kwargs):

  scoring_method=kwargs['scoring_method']

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

extractor=VideoWrapper(ExtractorTransformer(_image_extracor))

extractor = bob.pipelines.wrap(["sample"], extractor)


#=======================================================================================
# Dummy algorithm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class DummyClassifier(BaseEstimator, ClassifierMixin):

  def __init__(self, framelevel_score=True):
    self.framelevel_score = framelevel_score

  def fit(self, X, y=None):
    self.X_ = X
    return self

  def predict(self, X):

    X = check_array(X)
    return list(X)

  def decision_function(self,X):

    return X




classifier = DummyClassifier()


classifier = DummyClassifier()

# from bob.pad.face.transformer import VideoToFrames

from sklearn.base import TransformerMixin, BaseEstimator
import bob.pipelines as mario
from bob.pipelines.wrappers import _frmt
import logging

logger = logging.getLogger(__name__)

class VideoToFrames(TransformerMixin, BaseEstimator):
    """Expands video samples to frame-based samples only when transform is called.
    """

    def transform(self, video_samples):
        logger.debug(f"{_frmt(self)}.transform")
        output = []
        for sample in video_samples:
            annotations = getattr(sample, "annotations", {}) or {}
            video = sample.data

            if video is not None:
              for frame, frame_id in zip(video, video.indices):
                  new_sample = mario.Sample(
                      frame,
                      frame_id=frame_id,
                      annotations=annotations.get(str(frame_id)),
                      parent=sample,
                  )
                  output.append(new_sample)

        return output

    def fit(self, X, y=None, **fit_params):
        return self

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}


classifier = bob.pipelines.wrap(["sample"], classifier)
frame_cont_to_array = VideoToFrames()

from sklearn.pipeline import Pipeline

pipeline = Pipeline([("preprocessor", preprocessor),("extractor", extractor), ("frame_cont_to_array", frame_cont_to_array), ("classifier", classifier)]) #,

