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


def deadzone(var, deadzone=0.1):
    if type(var) != np.ndarray:
        res = 0.0 if np.fabs(var) < deadzone else var - np.sign(var) * deadzone
    else:
        res = var.copy()
        zero_mask = np.fabs(res) < deadzone
        res[zero_mask] = 0.0
        res[~zero_mask] -= np.sign(res[~zero_mask]) * deadzone
    return res


def time_interp(
    search_ts: float | np.ndarray,
    ts_arr: np.ndarray,
    data_arr: np.ndarray,
) -> np.ndarray:
    assert search_ts.shape[0] == data_arr.shape[0]

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


def interp_joint(cur_q, tar_joint, err_limit=0.05):
    err = tar_joint - cur_q
    max_err_fab = np.fabs(err).max()
    if max_err_fab < err_limit:
        return tar_joint, False
    else:
        err_norm = err / max_err_fab
        return cur_q + err_norm * err_limit, True


def mit_cmd(pos, vel=None, tau=None, kp=None, kd=None) -> np.ndarray:
    if pos is not None:
        pos_len = pos.shape[0]
        if kp is not None and kd is not None:
            mit_cmd = np.zeros((pos_len, 5))
            mit_cmd[:, 0] = pos
            if vel is not None:
                assert vel.shape[
                    0] == pos_len, "vel must have the same length as pos"
                mit_cmd[:, 1] = vel
            if tau is not None:
                assert tau.shape[
                    0] == pos_len, "tau must have the same length as pos"
                mit_cmd[:, 2] = tau
            if kp is not None:
                assert kp.shape[
                    0] == pos_len, "kp must have the same length as pos"
                mit_cmd[:, 3] = kp
            if kd is not None:
                assert kd.shape[
                    0] == pos_len, "kd must have the same length as pos"
                mit_cmd[:, 4] = kd
        else:
            mit_cmd = np.zeros((pos_len, 3))
            mit_cmd[:, 0] = pos
            if vel is not None:
                assert vel.shape[
                    0] == pos_len, "vel must have the same length as pos"
                mit_cmd[:, 1] = vel
            if tau is not None:
                assert tau.shape[
                    0] == pos_len, "tau must have the same length as pos"
                mit_cmd[:, 2] = tau
    elif tau is not None:
        mit_cmd = np.zeros((tau.shape[0], 5))
        mit_cmd[:, 2] = tau
    else:
        raise ValueError("Unsupported command type")

    return mit_cmd
