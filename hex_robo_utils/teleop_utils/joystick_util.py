#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import copy
import pygame
import pygame.constants as pygconst

from .teleop_base import HexTeleopBase
from ..time_utils import HexRate


class HexTeleopUtilJoystick(HexTeleopBase):

    def __init__(
            self,
            joy_button_mapping: dict[str, int] = {
                "L1": 4,
                "L2": 6,
                "R1": 5,
                "R2": 7,
                "A": 0,
                "B": 1,
                "X": 3,
                "Y": 2,
                "LA": 11,
                "RA": 12,
            },
            joy_axis_mapping: dict[str, int] = {
                "LX": 1,
                "LY": 0,
                "RX": 4,
                "RY": 3,
                "L2": 2,
                "R2": 5,
            },
            joy_hat_mapping: dict[str, int] = {
                "HX": 0,
                "HY": 1,
            }):
        super().__init__()

        self.__joy_button_mapping = joy_button_mapping
        self.__joy_axis_mapping = joy_axis_mapping
        self.__joy_hat_mapping = joy_hat_mapping
        self._value = {
            **{
                f"axis_{key}": 0.0
                for key in self.__joy_axis_mapping.keys()
            },
            **{
                f"button_{key}": False
                for key in self.__joy_button_mapping.keys()
            },
            **{
                f"hat_{key}": 0.0
                for key in self.__joy_hat_mapping.keys()
            },
            "pygame_quit": False,
        }

        self.__joystick = None
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() != 0:
            self.__joystick = pygame.joystick.Joystick(0)
            self.__joystick.init()
            print(f"Connected to joystick: {self.__joystick.get_name()}")
        else:
            raise Exception(
                "No joystick detected. Please connect the joystick and restart the program."
            )

    def close(self):
        super().close()
        if self.__joystick is not None:
            self.__joystick.quit()
        pygame.quit()

    def _teleop_listener(self):
        rate = HexRate(200.0)
        while self.is_working():
            value = copy.deepcopy(self._value)

            changed = False
            for event in pygame.event.get():
                if event.type == pygconst.QUIT:
                    value["pygame_quit"] = True
                    changed = True
                elif event.type == pygconst.JOYAXISMOTION:
                    for key, idx in self.__joy_axis_mapping.items():
                        if event.axis == idx:
                            value[f"axis_{key}"] = event.value
                            changed = True
                elif event.type == pygconst.JOYBUTTONDOWN:
                    for key, idx in self.__joy_button_mapping.items():
                        if event.button == idx:
                            value[f"button_{key}"] = True
                            changed = True
                elif event.type == pygconst.JOYBUTTONUP:
                    for key, idx in self.__joy_button_mapping.items():
                        if event.button == idx:
                            value[f"button_{key}"] = False
                            changed = True
                elif event.type == pygconst.JOYHATMOTION:
                    for key, idx in self.__joy_hat_mapping.items():
                        value[f"hat_{key}"] = event.value[idx]
                        changed = True

            if changed:
                with self._lock:
                    self._value = copy.deepcopy(value)
                    self._value_event.set()

            rate.sleep()
