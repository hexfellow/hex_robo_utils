#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-01-14
################################################################

# utils
from .dyn_util import HexDynUtil
from .dyn_util import HexMirrorUtil
from .dyn_util import HexFricUtil
from .dyn_util import HexFeedbackUtil
from .obs_util import HexObsUtilJoint
from .obs_util import HexObsUtilLowpassFilter
from .plan_util import HexPlanUtilBvp
from .ctrl_util import HexCtrlUtilMitJoint
from .ctrl_util import HexCtrlUtilPidJoint
from .ctrl_util import HexCtrlUtilIntJoint
from .ctrl_util import HexCtrlUtilMitWork
from .ctrl_util import HexCtrlUtilIntWork

# basic
from .math_utils import hat
from .math_utils import vee
from .math_utils import rad2deg
from .math_utils import deg2rad
from .math_utils import angle_norm
from .math_utils import quat_slerp
from .math_utils import quat_mul
from .math_utils import quat_inv
from .math_utils import trans_inv

# rotation
from .math_utils import rot2quat
from .math_utils import rot2axis
from .math_utils import rot2so3
from .math_utils import quat2rot
from .math_utils import quat2axis
from .math_utils import quat2so3
from .math_utils import axis2rot
from .math_utils import axis2quat
from .math_utils import axis2so3
from .math_utils import so32rot
from .math_utils import so32quat
from .math_utils import so32axis

# pose
from .math_utils import trans2part
from .math_utils import trans2se3
from .math_utils import part2trans
from .math_utils import part2se3
from .math_utils import se32trans
from .math_utils import se32part

# euler
from .math_utils import yaw2quat
from .math_utils import quat2yaw
from .math_utils import euler2rot
from .math_utils import rot2euler
from .math_utils import single_euler2rot
from .math_utils import single_rot2euler

__all__ = [
    # version
    '__version__',

    # utils
    'HexDynUtil',
    'HexMirrorUtil',
    'HexFricUtil',
    'HexFeedbackUtil',
    'HexObsUtilJoint',
    'HexObsUtilLowpassFilter',
    'HexPlanUtilBvp',
    'HexCtrlUtilMitJoint',
    'HexCtrlUtilPidJoint',
    'HexCtrlUtilIntJoint',
    'HexCtrlUtilMitWork',
    'HexCtrlUtilIntWork',
    'HexCtrlUtilPid',
    'HexCtrlUtilInt',

    # math basic
    'hat',
    'vee',
    'rad2deg',
    'deg2rad',
    'angle_norm',
    'quat_slerp',
    'quat_mul',
    'quat_inv',
    'trans_inv',

    # math rotation
    'rot2quat',
    'rot2axis',
    'rot2so3',
    'quat2rot',
    'quat2axis',
    'quat2so3',
    'axis2rot',
    'axis2quat',
    'axis2so3',
    'so32rot',
    'so32quat',
    'so32axis',

    # math pose
    'trans2part',
    'trans2se3',
    'part2trans',
    'part2se3',
    'se32trans',
    'se32part',

    # math euler
    'yaw2quat',
    'quat2yaw',
    'euler2rot',
    'rot2euler',
    'single_euler2rot',
    'single_rot2euler',
]

# Check optional dependencies availability
from importlib.util import find_spec

# Optional: hdf5
_HAS_H5PY = find_spec("h5py") is not None
if _HAS_H5PY:
    from .hdf5_reader import HexHdf5Reader
    from .hdf5_writer import HexHdf5Writer
    from .hdf5_writer import HexHdf5MultiWriter
    __all__.extend([
        'HexHdf5Reader',
        'HexHdf5Writer',
        'HexHdf5MultiWriter',
    ])

# print("#### Thanks for using HEXFELLOW Utilities :) ####")
