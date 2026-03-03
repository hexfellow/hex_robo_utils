#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

from .data_base import HexDataWriterBase
from .pandas_utils import HexPandasRecordReader
from .pandas_utils import HexPandasTrainReader
from .pandas_utils import HexPandasTrainWriter

__all__ = [
    'HexDataWriterBase',
    'HexPandasRecordReader',
    'HexPandasTrainReader',
    'HexPandasTrainWriter',
]

# Check optional dependencies availability
from importlib.util import find_spec

_HAS_H5PY = find_spec("h5py") is not None
_HAS_RERUN = find_spec("rerun_sdk") is not None

# Optional: hdf5
if _HAS_H5PY:
    from .hdf5_util import HexHdf5Writer, hdf5_to_pd
    __all__.extend([
        'HexHdf5Writer',
        'hdf5_to_pd',
    ])

# Optional: rerun
if _HAS_RERUN:
    from .rerun_util import HexRerunWriter, rerun_to_pd
    __all__.extend([
        'HexRerunWriter',
        'rerun_to_pd',
    ])
