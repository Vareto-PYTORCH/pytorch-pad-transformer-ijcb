#!/usr/bin/env python
"""
HQWMCA Db is a database for face PAD experiments.
"""
from bob.io.stream import Stream
from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector


from sklearn.preprocessing import Normalizer
import bob.core
logger = bob.core.log.setup("bob.learn.pytorch")
import numpy as np

color = Stream('color')

streams = { 'color'     : color}

# *****

PROTOCOL = {{PROTOCOL}}
ANNOTATIONS_DIR={{ANNOTATIONS_DIR}}

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

"""The default groups to use for reproducing the baselines.
"""