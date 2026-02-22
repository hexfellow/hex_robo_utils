#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

from .keyboard_util import HexTeleopUtilKeyboard

__all__ = [
    'HexTeleopUtilKeyboard',
]

from importlib.util import find_spec

_HAS_JOYSTICK = find_spec("pygame") is not None
_HAS_HELLO = find_spec("hex_device") is not None

if _HAS_JOYSTICK:
    from .joystick_util import HexTeleopUtilJoystick
    __all__.extend([
        'HexTeleopUtilJoystick',
    ])

if _HAS_HELLO:
    from .hello_util import HexTeleopUtilHello
    __all__.extend([
        'HexTeleopUtilHello',
    ])
