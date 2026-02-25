#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import copy
import numpy as np
from collections import deque

from .teleop_base import HexTeleopBase
from ..time_utils import HexRate


class HexTeleopUtilHello(HexTeleopBase):
    _HELLO_BUTTON_MAPPING = {
        "W": 1,
        "X": 2,
        "Y": 3,
        "Z": 0,
    }

    def __init__(self):
        super().__init__()

        self.__data_queue = deque(maxlen=10)

        self._value = {
            **{
                f"button_{key}": False
                for key in self._HELLO_BUTTON_MAPPING.keys()
            },
        }

    def close(self):
        super().close()

    def _teleop_listener(self):
        rate = HexRate(200.0)
        while self.is_working():
            rate.sleep()

            data = None
            try:
                data = self.__data_queue.popleft()
            except IndexError:
                continue

            value = copy.deepcopy(self._value)

            changed = False
            for key, idx in self._HELLO_BUTTON_MAPPING.items():
                new_btn = data[idx] > 0.0
                if new_btn != value[f"button_{key}"]:
                    value[f"button_{key}"] = new_btn
                    changed = True

            if changed:
                with self._lock:
                    self._value = copy.deepcopy(value)
                    self._value_event.set()

    def append_data(self, data: np.ndarray):
        self.__data_queue.append(data)
        
    def get_zero_value(self) -> dict[str, bool]:
        return {
            **{
                f"button_{key}": False
                for key in self._HELLO_BUTTON_MAPPING.keys()
            },
        }
