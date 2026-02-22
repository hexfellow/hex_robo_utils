#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import os

from importlib.util import find_spec

_HAS_JOYSTICK = find_spec("pygame") is not None
_HAS_HELLO = find_spec("hex_device") is not None
try:
    from hex_robo_utils import HexRate
    from hex_robo_utils import HexTeleopUtilKeyboard
except ImportError:
    import sys
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hex_robo_utils import HexRate
    from hex_robo_utils import HexTeleopUtilKeyboard


def keyboard_main():
    keyboard_util = HexTeleopUtilKeyboard()
    keyboard_util.start()

    rate = HexRate(2000.0)
    try:
        while keyboard_util.is_working():
            value = keyboard_util.pop_value()
            if value is not None:
                print(f"Got value: {value}")
                if value["key"] == 'q':
                    break
            rate.sleep()
    finally:
        keyboard_util.close()


if _HAS_JOYSTICK:
    from hex_robo_utils import HexTeleopUtilJoystick

    def joystick_main():
        joystick_util = HexTeleopUtilJoystick()
        joystick_util.start()
        rate = HexRate(2000.0)
        try:
            while joystick_util.is_working():
                value = joystick_util.pop_value()
                if value is not None:
                    print(f"Got value: {value}")
                    if value["button_B"] == True:
                        break
                rate.sleep()
        finally:
            joystick_util.close()


if _HAS_HELLO:
    from hex_robo_utils import HexTeleopUtilHello

    def hello_main():
        hello_util = HexTeleopUtilHello(host="172.18.24.90", port=8439)
        hello_util.start()
        rate = HexRate(2000.0)
        try:
            while hello_util.is_working():
                value = hello_util.pop_value()
                if value is not None:
                    print(f"Got value: {value}")
                    if value["button_W"] == True:
                        break
                rate.sleep()
        finally:
            hello_util.close()


if __name__ == '__main__':
    keyboard_main()
    # if _HAS_JOYSTICK:
    #     joystick_main()
    # if _HAS_HELLO:
    #     hello_main()
