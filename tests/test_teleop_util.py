#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import os
import numpy as np

from importlib.util import find_spec

_HAS_JOYSTICK = find_spec("pygame") is not None
try:
    from hex_robo_utils import HexRate
    from hex_robo_utils import HexTeleopUtilKeyboard
    from hex_robo_utils import HexTeleopUtilHello
except ImportError:
    import sys
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hex_robo_utils import HexRate
    from hex_robo_utils import HexTeleopUtilKeyboard
    from hex_robo_utils import HexTeleopUtilHello


def keyboard_main():
    keyboard_util = HexTeleopUtilKeyboard()
    keyboard_util.start()

    rate = HexRate(2000.0)
    try:
        while keyboard_util.is_working():
            rate.sleep()
            value = keyboard_util.pop_value()
            if value is not None:
                print(f"Got value: {value}")
                if value["key"] == 'q':
                    break
    finally:
        keyboard_util.close()


def hello_main():
    hello_util = HexTeleopUtilHello()
    hello_util.start()
    rate = HexRate(2000.0)
    cnt = 0
    try:
        while hello_util.is_working():
            rate.sleep()

            cnt += 1
            if cnt >= 1000:
                cnt = 0
                hello_util.append_msg(np.array([1.0, -1.0, -1.0, 1.0]))
            value = hello_util.pop_value()
            if value is not None:
                print(f"Got value: {value}")
                if value["button_W"] == True:
                    break
    finally:
        hello_util.close()


if _HAS_JOYSTICK:
    from hex_robo_utils import HexTeleopUtilJoystick

    def joystick_main():
        joystick_util = HexTeleopUtilJoystick()
        joystick_util.start()
        rate = HexRate(2000.0)
        try:
            while joystick_util.is_working():
                rate.sleep()
                value = joystick_util.pop_value()
                if value is not None:
                    print(f"Got value: {value}")
                    if value["button_B"] == True:
                        break
        finally:
            joystick_util.close()


if __name__ == '__main__':
    # keyboard_main()
    hello_main()
    # if _HAS_JOYSTICK:
    #     joystick_main()
