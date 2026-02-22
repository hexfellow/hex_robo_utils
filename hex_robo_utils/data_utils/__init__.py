#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

__all__ = []

# Check optional dependencies availability
from importlib.util import find_spec

_HAS_H5PY = find_spec("h5py") is not None
_HAS_RERUN = find_spec("rerun-sdk") is not None

# Optional: hdf5
if _HAS_H5PY:
    from .hdf5_reader import HexHdf5Reader
    from .hdf5_writer import HexHdf5Writer
    __all__.extend([
        'HexHdf5Reader',
        'HexHdf5Writer',
    ])

# Optional: rerun
if _HAS_RERUN:
    from .rerun_reader import HexRerunReader
    from .rerun_writer import HexRerunWriter
    __all__.extend([
        'HexRerunReader',
        'HexRerunWriter',
    ])
