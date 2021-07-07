# =============================================================================
# define instance of the preprocessor:

from torchvision import transforms

import numpy as np 

from bob.extension import rc

from bob.learn.pytorch.datasets import ChannelSelect, RandomHorizontalFlipImage

from bob.paper.ijcb2021_vision_transformer_pad.preprocessor.FaceCropAlign import auto_norm_image as _norm_func

from bob.extension import rc

from bob.paper.ijcb2021_vision_transformer_pad.preprocessor import FaceCropAlign, VideoFaceCropAlignBlockPatch

from bob.paper.ijcb2021_vision_transformer_pad.database import HQWMCAPadDatabase

from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector

from sklearn.pipeline import Pipeline

from bob.bio.base.transformers import PreprocessorTransformer, ExtractorTransformer

from bob.bio.video.transformer import VideoWrapper

from bob.io.stream import Stream
import bob.ip.stereo
from sklearn.preprocessing import Normalizer
import bob.core
logger = bob.core.log.setup("bob.learn.pytorch")
import numpy as np

PREPROCESSED_DIR={{PREPROCESSED_DIR}}

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

preprocessor = PreprocessorTransformer(
    VideoFaceCropAlignBlockPatch(
        preprocessors=_preprocessors, channel_names=_channel_names, return_multi_channel_flag=True
    )
)

preprocessor = bob.pipelines.wrap(
    ["sample"], preprocessor, transform_extra_arguments=(("annotations", "annotations"),),
)

preprocessor = bob.pipelines.CheckpointWrapper(
    preprocessor,
    features_dir=PREPROCESSED_DIR,
    load_func=bob.bio.video.VideoLikeContainer.load,
    save_func=bob.bio.video.VideoLikeContainer.save_function,
)


pipeline = Pipeline([("preprocessor", preprocessor)])