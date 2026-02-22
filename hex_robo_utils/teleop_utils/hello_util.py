#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import copy, time
import numpy as np
from hex_device import HexDeviceApi

from .teleop_base import HexTeleopBase
from ..time_utils import HexRate

HELLO_DEVICE_TYPE_DICT = {
    # 6 dof
    "archer_y6": 26,
    "archer_d6y": 26,
    "archer_l6y": 26,
    "firefly_y6": 26,
}


class HexTeleopUtilHello(HexTeleopBase):

    def __init__(self,
                 host: str,
                 port: int = 8439,
                 type_name: str = "archer_y6"):
        super().__init__()

        self.__hello_axis_mapping = {
            "T": 0,
            "X": 1,
            "Y": 2,
        }
        self.__hello_button_mapping = {
            "W": 4,
            "X": 5,
            "Y": 6,
            "Z": 3,
        }

        # open device api
        self.__hex_api = HexDeviceApi(
            ws_url=f"ws://{host}:{port}",
            control_hz=250.0,
        )

        # open arm
        arm_type = HELLO_DEVICE_TYPE_DICT[type_name]
        while self.__hex_api.find_device_by_robot_type(arm_type) is None:
            print("\033[33mArm not found\033[0m")
            time.sleep(1)
        self.__arm = self.__hex_api.find_device_by_robot_type(arm_type)
        self.__arm.start()

        # open gripper
        while self.__hex_api.find_optional_device_by_id(1) is None:
            print("\033[33mGripper not found\033[0m")
            time.sleep(1)
        self.__gripper = self.__hex_api.find_optional_device_by_id(1)
        self.__gripper.set_rgb_stripe_command([0] * 6, [255] * 6, [0] * 6)

        arm_dofs = len(self.__arm)
        self._value = {
            "arm_pos": np.zeros(arm_dofs),
            "arm_vel": np.zeros(arm_dofs),
            **{
                f"axis_{key}": 0.0
                for key in self.__hello_axis_mapping.keys()
            },
            **{
                f"button_{key}": False
                for key in self.__hello_button_mapping.keys()
            },
        }

    def close(self):
        super().close()
        if self.__gripper is not None:
            self.__gripper.set_rgb_stripe_command([255] * 6, [0] * 6, [0] * 6)
            time.sleep(0.2)
        if self.__arm is not None:
            self.__arm.stop()
        if self.__hex_api is not None:
            self.__hex_api.close()

    def _teleop_listener(self):
        rate = HexRate(200.0)
        while self.is_working():
            value = copy.deepcopy(self._value)

            arm_state = self.__arm.get_simple_motor_status()
            gripper_state = self.__gripper.get_simple_motor_status()

            changed = False
            if arm_state is not None:
                value["arm_pos"] = arm_state['pos']
                value["arm_vel"] = arm_state['vel']
                changed = True
            if gripper_state is not None:
                gripper_pos = gripper_state['pos']
                for key, idx in self.__hello_button_mapping.items():
                    value[f"button_{key}"] = gripper_pos[idx] > 0.0
                for key, idx in self.__hello_axis_mapping.items():
                    value[f"axis_{key}"] = gripper_pos[idx]
                changed = True

            if changed:
                with self._lock:
                    self._value = copy.deepcopy(value)
                    self._value_event.set()

            rate.sleep()
