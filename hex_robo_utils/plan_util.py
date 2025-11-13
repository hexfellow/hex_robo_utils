#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-11-13
################################################################

import numpy as np


class HexPlanUtilBvp:

    def __init__(
        self,
        dt,
        jnt_num: int = 6,
        tar_num: int = 2,
        adjust_weights: np.ndarray = None,
    ):
        # q = a0 + a1*t + a2*t^2 + a3*t^3
        self.__jnt_num = jnt_num
        self.__tar_num = tar_num
        self.__dt_arr = np.arange(self.__tar_num) * dt + dt
        self.__dt2_arr = self.__dt_arr**2
        self.__dt3_arr = self.__dt_arr**3
        self.__a_mat = np.vstack((self.__dt2_arr, self.__dt3_arr)).T

        h_mat = self.__a_mat @ np.linalg.inv(
            self.__a_mat.T @ self.__a_mat) @ self.__a_mat.T
        leverage = np.diag(h_mat)
        self.__weights_leverage = 1.0 / (leverage + 1e-6)

        if adjust_weights is not None:
            assert adjust_weights.shape == (self.__tar_num, )
            assert np.all(
                adjust_weights >= 0), "adjust weights must be non-negative"
            self.__weights_leverage = self.__weights_leverage * adjust_weights

        self.__weights_leverage = self.__weights_leverage / np.linalg.norm(
            self.__weights_leverage)

    def __call__(
        self,
        q0: np.ndarray,
        dq0: np.ndarray,
        q_tar_arr: np.ndarray,
    ):
        assert q0.shape == (self.__jnt_num, )
        assert dq0.shape == (self.__jnt_num, )
        assert q_tar_arr.shape == (self.__tar_num, self.__jnt_num)

        a0_arr = q0.copy()
        a1_arr = dq0.copy()

        y = q_tar_arr - (a0_arr[None, :] +
                         a1_arr[None, :] * self.__dt_arr[:, None])

        # weights for least squares
        sqrt_weights = np.sqrt(self.__weights_leverage)
        a_mat_weighted = self.__a_mat * sqrt_weights[:, None]
        y_weighted = y * sqrt_weights[:, None]

        # least squares
        sol, residuals, ranks, svals = np.linalg.lstsq(
            a_mat_weighted,
            y_weighted,
            rcond=None,
        )
        a2_arr = sol[0, :]
        a3_arr = sol[1, :]

        if residuals.ndim == 0:
            residuals_arr = np.full(self.__jnt_num, residuals)
        else:
            residuals_arr = residuals

        return {
            "format": "a0 + a1*t + a2*t^2 + a3*t^3",
            "a0": a0_arr,
            "a1": a1_arr,
            "a2": a2_arr,
            "a3": a3_arr,
            "residuals": residuals_arr,
            "ranks": ranks,
            "singular_values": svals,
        }

    def calc_state(self, dt: float, params: dict):
        a0 = params["a0"]
        a1 = params["a1"]
        a2 = params["a2"]
        a3 = params["a3"]
        q = a0 + a1 * dt + a2 * dt**2 + a3 * dt**3
        dq = a1 + 2 * a2 * dt + 3 * a3 * dt**2
        ddq = 2 * a2 + 6 * a3 * dt
        return {
            "q": q,
            "dq": dq,
            "ddq": ddq,
        }
