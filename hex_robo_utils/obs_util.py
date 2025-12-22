#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-09-18
################################################################

import copy
import numpy as np
from typing import Tuple


class HexObsUtilJoint:

    def __init__(
        self,
        mass: np.ndarray,
        damp: np.ndarray,
        stiff: np.ndarray,
        dt: float,
        q_limit: np.ndarray,
        dq_limit: np.ndarray,
        ddq_limit: np.ndarray,
    ):
        ### physical params
        self.__mass_inv = np.linalg.inv(mass)
        self.__damp = damp
        self.__stiff = stiff
        self.__dt = dt

        ### limits
        self.__q_limit = q_limit
        self.__dq_limit = dq_limit
        self.__ddq_limit = ddq_limit

        ### variables
        self.__ready = False
        self.__obs_q = None
        self.__obs_dq = None

    def get_mass(self) -> np.ndarray:
        return np.linalg.inv(self.__mass_inv)

    def set_mass(self, mass: np.ndarray):
        self.__mass_inv = np.linalg.inv(mass)

    def get_damp(self) -> np.ndarray:
        return copy.deepcopy(self.__damp)

    def set_damp(self, damp: np.ndarray):
        self.__damp = copy.deepcopy(damp)

    def get_stiff(self) -> np.ndarray:
        return copy.deepcopy(self.__stiff)

    def set_stiff(self, stiff: np.ndarray):
        self.__stiff = copy.deepcopy(stiff)

    def get_dt(self) -> float:
        return self.__dt

    def set_dt(self, dt: float):
        self.__dt = dt

    def is_ready(self) -> bool:
        return self.__ready

    def set_state(self, q: np.ndarray, dq: np.ndarray):
        self.__obs_q = q
        self.__obs_dq = dq
        self.__ready = True

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.__obs_q, self.__obs_dq

    def predict(
        self,
        q_tar: np.ndarray,
    ):
        q_cur = self.__obs_q
        dq_cur = self.__obs_dq

        # runge-kutta k1
        dq1 = dq_cur
        ddq1 = self.__ddq(q_tar - q_cur, dq_cur)

        # runge-kutta k2
        q2 = q_cur + dq1 * self.__dt * 0.5
        dq2 = dq_cur + ddq1 * self.__dt * 0.5
        ddq2 = self.__ddq(q_tar - q2, dq2)

        # runge-kutta k3
        q3 = q_cur + dq2 * self.__dt * 0.5
        dq3 = dq_cur + ddq2 * self.__dt * 0.5
        ddq3 = self.__ddq(q_tar - q3, dq3)

        # runge-kutta k4
        q4 = q_cur + dq3 * self.__dt
        dq4 = dq_cur + ddq3 * self.__dt
        ddq4 = self.__ddq(q_tar - q4, dq4)

        # runge-kutta
        q_next = q_cur + (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4) / 6.0 * self.__dt
        dq_next = dq_cur + (ddq1 + 2.0 * ddq2 + 2.0 * ddq3 +
                            ddq4) / 6.0 * self.__dt
        low_mask = q_next < self.__q_limit[:, 0]
        high_mask = q_next > self.__q_limit[:, 1]

        # clip
        q_next[low_mask] = self.__q_limit[low_mask, 0]
        q_next[high_mask] = self.__q_limit[high_mask, 1]
        dq_next[low_mask] = 0.0
        dq_next[high_mask] = 0.0
        dq_next = np.clip(
            dq_next,
            self.__dq_limit[:, 0],
            self.__dq_limit[:, 1],
        )

        # set state
        self.__obs_q = q_next
        self.__obs_dq = dq_next

    def __ddq(
        self,
        q_err: np.ndarray,
        dq_cur: np.ndarray,
    ) -> np.ndarray:
        ddq = (self.__stiff @ q_err - self.__damp @ dq_cur) @ self.__mass_inv
        ddq = np.clip(ddq, self.__ddq_limit[:, 0], self.__ddq_limit[:, 1])
        return ddq

    def update(
        self,
        q_sensor: np.ndarray,
        dq_sensor: np.ndarray,
        weight_sensor: np.ndarray,
    ):
        q_sensor = np.clip(q_sensor, self.__q_limit[:, 0], self.__q_limit[:,
                                                                          1])
        dq_sensor = np.clip(dq_sensor, self.__dq_limit[:, 0],
                            self.__dq_limit[:, 1])

        # update state
        weight_intgr = 1.0 - weight_sensor
        self.__obs_q = self.__obs_q * weight_intgr + q_sensor * weight_sensor
        self.__obs_dq = self.__obs_dq * weight_intgr + dq_sensor * weight_sensor


class HexObsUtilLowpassFilter:

    def __init__(self, lowpass_num: int, init_value: np.ndarray):
        self.__lowpass_num = lowpass_num
        wt_range = np.arange(1, lowpass_num + 1)
        wt_range_rev = wt_range[::-1]
        wt = np.min(np.stack((wt_range, wt_range_rev)), axis=0)
        self.__wt = (wt / np.sum(wt)).reshape(-1, 1, 1)
        self.__arr = np.array([init_value] * self.__lowpass_num)

    def reset(self, init_value: np.ndarray):
        self.__arr = np.array([init_value] * self.__lowpass_num)

    def append(self, value: np.ndarray):
        self.__arr[:-1, :] = self.__arr[1:, :]
        self.__arr[-1, :] = value

    def get_value(self) -> np.ndarray:
        return (self.__arr * self.__wt).sum(axis=0)
