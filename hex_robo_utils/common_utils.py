#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import time
import cv2
import numpy as np
from .math_utils import angle_norm


def wait_client(client, timeout: float = 5.0):
    for _ in range(int(timeout * 10)):
        if client.is_working():
            if hasattr(client, "seq_clear"):
                client.seq_clear()
            return True
        else:
            time.sleep(0.1)
    return False


def depth_to_cmap(
        depth: np.ndarray,
        range: tuple[int, int] = (70, 1000),
) -> np.ndarray:
    depth_values = depth.astype(np.float32)
    depth_norm = np.clip((depth_values - range[0]) / (range[1] - range[0]),
                         0.0, 1.0)
    depth_8u = (depth_norm * 255.0).astype(np.uint8)
    depth_cmap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    return depth_cmap


def deadzone(var: float | np.ndarray,
             deadzone: float | np.ndarray = 0.1) -> float | np.ndarray:
    if isinstance(var, float):
        assert isinstance(
            deadzone, float), "When var is a float, deadzone must be a float"
        return 0.0 if np.fabs(
            var) < deadzone else var - np.sign(var) * deadzone
    elif isinstance(var, np.ndarray):
        assert isinstance(deadzone, float) or (
            isinstance(deadzone, np.ndarray) and var.shape == deadzone.shape
        ), "When var is an array, deadzone must be a float or an array with the same shape as var"
        res = var.copy()
        zero_mask = np.fabs(res) < deadzone
        res[zero_mask] = 0.0
        res[~zero_mask] -= np.sign(res[~zero_mask]) * deadzone[~zero_mask]
        return res
    else:
        raise ValueError("Unsupported variable type")


def time_interp(
    search_ts: float | np.ndarray,
    ts_arr: np.ndarray,
    data_arr: np.ndarray,
) -> np.ndarray:
    assert ts_arr.shape[0] == data_arr.shape[
        0], "**search_ts** and **data_arr** must have the same length"

    idx = np.searchsorted(ts_arr, search_ts)
    idx_later = np.clip(idx, 1, ts_arr.shape[0] - 1)
    idx_earlier = idx_later - 1

    ts_earlier, ts_later = ts_arr[idx_earlier], ts_arr[idx_later]
    ts_diff = ts_later - ts_earlier
    ts_diff = np.where(ts_diff == 0, 1, ts_diff)
    weight_later = ((search_ts - ts_earlier) / ts_diff).reshape(-1, 1)
    weight_earlier = 1.0 - weight_later
    return data_arr[idx_later] * weight_later + data_arr[
        idx_earlier] * weight_earlier


def remap(
    value: float | np.ndarray, range: tuple[float | np.ndarray,
                                            float | np.ndarray],
    new_range: tuple[float | np.ndarray, float | np.ndarray]
) -> float | np.ndarray:
    ratio = np.clip((value - range[0]) / (range[1] - range[0]), 0.0, 1.0)
    return ratio * (new_range[1] - new_range[0]) + new_range[0]


def interp_joint(
    cur_q: np.ndarray | float,
    tar_joint: np.ndarray | float,
    err_limit: np.ndarray | float = 0.05,
    arrive_limit: np.ndarray | float = 0.10,
) -> tuple[np.ndarray | float, bool, bool]:
    assert isinstance(cur_q, np.ndarray) or isinstance(
        cur_q, float), "**cur_q** must be a numpy array or a float"
    assert isinstance(err_limit, np.ndarray) or isinstance(
        err_limit, float), "**err_limit** must be a numpy array or a float"
    assert isinstance(
        tar_joint, type(cur_q)), "**tar_joint** must be the same type as cur_q"
    assert cur_q.shape == tar_joint.shape if isinstance(
        cur_q, np.ndarray
    ) else True, "**tar_joint** must have the same shape as cur_q"
    assert cur_q.shape == err_limit.shape if isinstance(
        err_limit, np.ndarray
    ) else True, "**err_limit** must have the same shape as cur_q when err_limit is a numpy array"

    err = tar_joint - cur_q
    max_err_fab = np.fabs(err).max()
    arrive_flag = max_err_fab < arrive_limit
    if max_err_fab > err_limit:
        err_norm = err / max_err_fab
        return cur_q + err_norm * err_limit, True, arrive_flag
    else:
        return tar_joint, False, arrive_flag


def mit_cmd(
    pos: None | np.ndarray,
    vel: None | np.ndarray = None,
    tau: None | np.ndarray = None,
    kp: None | np.ndarray = None,
    kd: None | np.ndarray = None,
) -> np.ndarray:
    if pos is not None:
        pos_len = pos.shape[0]
        if kp is not None and kd is not None:
            mit_cmd = np.zeros((pos_len, 5))
            mit_cmd[:, 0] = pos
            if vel is not None:
                assert vel.shape[
                    0] == pos_len, "**vel** must have the same length as pos"
                mit_cmd[:, 1] = vel
            if tau is not None:
                assert tau.shape[
                    0] == pos_len, "**tau** must have the same length as pos"
                mit_cmd[:, 2] = tau
            if kp is not None:
                assert kp.shape[
                    0] == pos_len, "**kp** must have the same length as pos"
                mit_cmd[:, 3] = kp
            if kd is not None:
                assert kd.shape[
                    0] == pos_len, "**kd** must have the same length as pos"
                mit_cmd[:, 4] = kd
        else:
            mit_cmd = np.zeros((pos_len, 3))
            mit_cmd[:, 0] = pos
            if vel is not None:
                assert vel.shape[
                    0] == pos_len, "**vel** must have the same length as pos"
                mit_cmd[:, 1] = vel
            if tau is not None:
                assert tau.shape[
                    0] == pos_len, "**tau** must have the same length as pos"
                mit_cmd[:, 2] = tau
    elif tau is not None:
        mit_cmd = np.zeros((tau.shape[0], 5))
        mit_cmd[:, 2] = tau
    else:
        raise ValueError("Unsupported command type")

    return mit_cmd


def dof_parser(dof_arr: np.ndarray, is_hello: bool = False) -> int:
    if is_hello:
        return {
            "robot_arm": dof_arr[0],
            "robot_gripper": 1,
            "robot_sum": dof_arr[0] + 1,
            "joy_axis": 2,
            "joy_button": 4,
            "hello_sum": dof_arr[0] + dof_arr[1],
        }
    else:
        has_gripper = dof_arr.shape[0] > 1
        return {
            "robot_arm": dof_arr[0],
            "robot_gripper": dof_arr[1] if has_gripper else None,
            "robot_sum": dof_arr[0] + (dof_arr[1] if has_gripper else 0),
        }


def arm_pos_limit(
    arm_pos: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    normed_rads = angle_norm(arm_pos)
    outside = (normed_rads < lower_bound) | (normed_rads > upper_bound)
    if not np.any(outside):
        return normed_rads

    lower_dist = np.fabs(angle_norm((normed_rads - lower_bound)[outside]))
    upper_dist = np.fabs(angle_norm((normed_rads - upper_bound)[outside]))
    choose_lower = lower_dist < upper_dist
    choose_upper = ~choose_lower

    outside_full = np.flatnonzero(outside)
    outside_lower = outside_full[choose_lower]
    outside_upper = outside_full[choose_upper]
    normed_rads[outside_lower] = lower_bound[outside_lower]
    normed_rads[outside_upper] = upper_bound[outside_upper]

    return normed_rads


def gripper_pos_limit(
    gripper_pos: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    return np.clip(gripper_pos, lower_bound, upper_bound)
