#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-01-14
################################################################

import numpy as np
from typing import Tuple


def hat(vec: np.ndarray):
    """so(3) vector → skew-symmetric matrix"""
    assert (vec.ndim == 1 and vec.shape == (3, )) or (
        vec.ndim == 2 and vec.shape[1] == 3), "cross_matrix vec shape err"

    if vec.ndim == 1:
        mat = np.array([
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ])
    else:
        # Batch processing: input shape (N, 3) -> output shape (N, 3, 3)
        num = vec.shape[0]
        mat = np.zeros((num, 3, 3))
        mat[:, 0, 1] = -vec[:, 2]
        mat[:, 0, 2] = vec[:, 1]
        mat[:, 1, 0] = vec[:, 2]
        mat[:, 1, 2] = -vec[:, 0]
        mat[:, 2, 0] = -vec[:, 1]
        mat[:, 2, 1] = vec[:, 0]
    return mat


def vee(mat: np.ndarray):
    """skew-symmetric matrix → so(3) vector"""
    assert (mat.ndim == 2 and mat.shape == (3, 3)) or (
        mat.ndim == 3 and mat.shape[1:] == (3, 3)), "vee mat shape err"

    if mat.ndim == 2:
        vec = np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
    else:
        # Batch processing: input shape (N, 3, 3) -> output shape (N, 3)
        vec = np.zeros((mat.shape[0], 3))
        vec[:, 0] = mat[:, 2, 1]
        vec[:, 1] = mat[:, 0, 2]
        vec[:, 2] = mat[:, 1, 0]
    return vec


def rad2deg(rad):
    deg = rad * 180.0 / np.pi
    return deg


def deg2rad(deg):
    rad = deg * np.pi / 180.0
    return rad


def angle_norm(rad):
    normed_rad = (rad + np.pi) % (2 * np.pi) - np.pi
    return normed_rad


def vec_norm(vec: np.ndarray) -> np.ndarray:
    assert (vec.ndim == 1 and vec.shape == (3, )) or (
        vec.ndim == 2 and vec.shape[1] == 3), "vec_norm vec shape err"

    single = vec.ndim == 1
    if single:
        vec = vec[np.newaxis, :]

    normed_vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return normed_vec[0] if single else normed_vec


def quat_norm(quat: np.ndarray) -> np.ndarray:
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat_norm quat shape err"

    single = quat.ndim == 1
    if single:
        quat = quat[np.newaxis, :]

    # Normalize quaternion
    normed_quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    # Ensure first component (w) is non-negative using vectorized operation
    sign = np.where(normed_quat[:, 0:1] < 0.0, -1.0, 1.0)
    normed_quat = normed_quat * sign

    return normed_quat[0] if single else normed_quat


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t) -> np.ndarray:
    assert ((q1.ndim == 1 and q1.shape == (4, ))
            or (q1.ndim == 2 and q1.shape[1] == 4)), "quat_slerp q1 shape err"
    assert ((q2.ndim == 1 and q2.shape == (4, ))
            or (q2.ndim == 2 and q2.shape[1] == 4)), "quat_slerp q2 shape err"
    assert q1.ndim == q2.ndim, "quat_slerp q1 and q2 must have same ndim"

    single = q1.ndim == 1
    if single:
        assert isinstance(
            t, (int, float)), "quat_slerp t must be scalar for 1D quat"
        q1 = q1[np.newaxis, :]
        q2 = q2[np.newaxis, :]
        t = np.array([t])
    else:
        assert q1.shape[0] == q2.shape[0], "quat_slerp batch size mismatch"
        assert (isinstance(t, np.ndarray) and t.ndim == 1 and
                t.shape[0] == q1.shape[0]) or isinstance(t, (int, float)), \
            "quat_slerp t shape err"
        if isinstance(t, (int, float)):
            t = np.full(q1.shape[0], t)

    # Normalize using quat_norm
    q1_norm = quat_norm(q1)
    q2_norm = quat_norm(q2)

    # dot
    dot = np.sum(q1_norm * q2_norm, axis=1)
    neg_dot_mask = dot < 0.0
    q2_norm[neg_dot_mask] = -q2_norm[neg_dot_mask]
    dot[neg_dot_mask] = -dot[neg_dot_mask]
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    # slerp
    small_mask = np.fabs(theta) < 1e-6
    q = np.zeros_like(q1)

    if np.any(small_mask):
        q[small_mask] = q1_norm[small_mask] + t[small_mask, np.newaxis] * (
            q2_norm[small_mask] - q1_norm[small_mask])
        q[small_mask] = quat_norm(q[small_mask])

    if np.any(~small_mask):
        sin_theta = np.sin(theta[~small_mask])
        q1_factor = np.sin(
            (1 - t[~small_mask]) * theta[~small_mask]) / sin_theta
        q2_factor = np.sin(t[~small_mask] * theta[~small_mask]) / sin_theta
        q[~small_mask] = (q1_factor[:, np.newaxis] * q1_norm[~small_mask] +
                          q2_factor[:, np.newaxis] * q2_norm[~small_mask])

    return q[0] if single else q


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    assert ((q1.ndim == 1 and q1.shape == (4, ))
            or (q1.ndim == 2 and q1.shape[1] == 4)), "quat_mul q1 shape err"
    assert ((q2.ndim == 1 and q2.shape == (4, ))
            or (q2.ndim == 2 and q2.shape[1] == 4)), "quat_mul q2 shape err"
    assert q1.ndim == q2.ndim, "quat_mul q1 and q2 must have same ndim"

    single = q1.ndim == 1
    if single:
        q1 = q1[np.newaxis, :]
        q2 = q2[np.newaxis, :]
    else:
        assert q1.shape[0] == q2.shape[0], "quat_mul batch size mismatch"

    # Normalize using quat_norm
    q1_norm = quat_norm(q1)
    q2_norm = quat_norm(q2)

    # mul
    w1, x1, y1, z1 = q1_norm[:, 0], q1_norm[:, 1], q1_norm[:, 2], q1_norm[:, 3]
    w2, x2, y2, z2 = q2_norm[:, 0], q2_norm[:, 1], q2_norm[:, 2], q2_norm[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q = np.stack([w, x, y, z], axis=1)

    return q[0] if single else q


def quat_inv(quat: np.ndarray) -> np.ndarray:
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat_inv quat shape err"

    single = quat.ndim == 1
    if single:
        quat = quat[np.newaxis, :]

    # Normalize using quat_norm
    q = quat_norm(quat)

    # inv
    inv = np.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], axis=1)

    return inv[0] if single else inv


def trans_inv(trans: np.ndarray) -> np.ndarray:
    assert (trans.ndim == 2 and trans.shape == (4, 4)) or (
        trans.ndim == 3
        and trans.shape[1:] == (4, 4)), "trans_inv trans shape err"

    single = trans.ndim == 2
    if single:
        trans = trans[np.newaxis, ...]

    pos = trans[:, :3, 3]
    rot = trans[:, :3, :3]

    # inv
    inv = np.zeros_like(trans)
    inv[:, :3, :3] = rot.transpose(0, 2, 1)
    inv[:, :3, 3] = -np.einsum('ijk,ik->ij', inv[:, :3, :3], pos)
    inv[:, 3, :3] = 0.0
    inv[:, 3, 3] = 1.0

    return inv[0] if single else inv


def rot2quat(rot: np.ndarray) -> np.ndarray:
    assert (rot.ndim == 2 and rot.shape == (3, 3)) or (
        rot.ndim == 3 and rot.shape[1:] == (3, 3)), "rot2quat rot shape err"

    single = rot.ndim == 2
    if single:
        rot = rot[np.newaxis, ...]

    # Batch processing: input shape (N, 3, 3) -> output shape (N, 4)
    num = rot.shape[0]
    quat = np.zeros((num, 4))
    trace = np.trace(rot, axis1=1, axis2=2)

    # Case 1: trace > 0
    mask1 = trace > 0
    if np.any(mask1):
        temp = 2.0 * np.sqrt(1 + trace[mask1])
        quat[mask1, 0] = 0.25 * temp
        quat[mask1, 1] = (rot[mask1, 2, 1] - rot[mask1, 1, 2]) / temp
        quat[mask1, 2] = (rot[mask1, 0, 2] - rot[mask1, 2, 0]) / temp
        quat[mask1, 3] = (rot[mask1, 1, 0] - rot[mask1, 0, 1]) / temp

    # Case 2: trace <= 0
    mask2 = ~mask1
    if np.any(mask2):
        diag = np.diagonal(rot, axis1=1, axis2=2)
        mask2a = mask2 & (diag[:, 0] > diag[:, 1]) & (diag[:, 0] > diag[:, 2])
        mask2b = mask2 & (~mask2a) & (diag[:, 1] > diag[:, 2])
        mask2c = mask2 & (~mask2a) & (~mask2b)

        # Case 2a: rot[0,0] > rot[1,1] and rot[0,0] > rot[2,2]
        if np.any(mask2a):
            temp = 2.0 * np.sqrt(1 + diag[mask2a, 0] - diag[mask2a, 1] -
                                 diag[mask2a, 2])
            quat[mask2a, 0] = (rot[mask2a, 2, 1] - rot[mask2a, 1, 2]) / temp
            quat[mask2a, 1] = 0.25 * temp
            quat[mask2a, 2] = (rot[mask2a, 1, 0] + rot[mask2a, 0, 1]) / temp
            quat[mask2a, 3] = (rot[mask2a, 0, 2] + rot[mask2a, 2, 0]) / temp

        # Case 2b: rot[1,1] > rot[2,2]
        if np.any(mask2b):
            temp = 2.0 * np.sqrt(1 + diag[mask2b, 1] - diag[mask2b, 0] -
                                 diag[mask2b, 2])
            quat[mask2b, 0] = (rot[mask2b, 0, 2] - rot[mask2b, 2, 0]) / temp
            quat[mask2b, 1] = (rot[mask2b, 1, 0] + rot[mask2b, 0, 1]) / temp
            quat[mask2b, 2] = 0.25 * temp
            quat[mask2b, 3] = (rot[mask2b, 2, 1] + rot[mask2b, 1, 2]) / temp

        # Case 2c: rot[2,2] >= rot[0,0] and rot[2,2] >= rot[1,1]
        if np.any(mask2c):
            temp = 2.0 * np.sqrt(1 + diag[mask2c, 2] - diag[mask2c, 0] -
                                 diag[mask2c, 1])
            quat[mask2c, 0] = (rot[mask2c, 1, 0] - rot[mask2c, 0, 1]) / temp
            quat[mask2c, 1] = (rot[mask2c, 0, 2] + rot[mask2c, 2, 0]) / temp
            quat[mask2c, 2] = (rot[mask2c, 2, 1] + rot[mask2c, 1, 2]) / temp
            quat[mask2c, 3] = 0.25 * temp

    return quat[0] if single else quat


def rot2axis(rot: np.ndarray) -> Tuple[np.ndarray, float]:
    assert (rot.ndim == 2 and rot.shape == (3, 3)) or (
        rot.ndim == 3 and rot.shape[1:] == (3, 3)), "rot2axis rot shape err"

    single = rot.ndim == 2
    if single:
        rot = rot[np.newaxis, ...]

    cos_theta = 0.5 * (np.trace(rot, axis1=1, axis2=2) - 1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    small_mask = theta < 1e-6
    axis = np.zeros((rot.shape[0], 3))
    axis[small_mask] = np.array([1.0, 0.0, 0.0])

    if np.any(~small_mask):
        axis_matrix = (rot[~small_mask] - rot[~small_mask].transpose(
            0, 2, 1)) / (2 *
                         np.sin(theta[~small_mask])[:, np.newaxis, np.newaxis])
        axis[~small_mask] = vee(axis_matrix)

    return (axis[0], theta[0]) if single else (axis, theta)


def rot2so3(rot: np.ndarray) -> np.ndarray:
    assert (rot.ndim == 2 and rot.shape == (3, 3)) or (
        rot.ndim == 3 and rot.shape[1:] == (3, 3)), "rot2so3 rot shape err"

    single = rot.ndim == 2
    axis, theta = rot2axis(rot)
    return theta * axis if single else theta[:, np.newaxis] * axis


def quat2rot(quat: np.ndarray) -> np.ndarray:
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat2rot quat shape err"

    single = quat.ndim == 1
    if single:
        quat = quat[np.newaxis, :]

    # Normalize using quat_norm
    q = quat_norm(quat)

    # temp vars
    qx2 = q[:, 1] * q[:, 1]
    qy2 = q[:, 2] * q[:, 2]
    qz2 = q[:, 3] * q[:, 3]
    qxqw = q[:, 1] * q[:, 0]
    qyqw = q[:, 2] * q[:, 0]
    qzqw = q[:, 3] * q[:, 0]
    qxqy = q[:, 1] * q[:, 2]
    qyqz = q[:, 2] * q[:, 3]
    qzqx = q[:, 3] * q[:, 1]

    # rot
    num = q.shape[0]
    rot = np.zeros((num, 3, 3))
    rot[:, 0, 0] = 1 - 2 * (qy2 + qz2)
    rot[:, 0, 1] = 2 * (qxqy - qzqw)
    rot[:, 0, 2] = 2 * (qzqx + qyqw)
    rot[:, 1, 0] = 2 * (qxqy + qzqw)
    rot[:, 1, 1] = 1 - 2 * (qx2 + qz2)
    rot[:, 1, 2] = 2 * (qyqz - qxqw)
    rot[:, 2, 0] = 2 * (qzqx - qyqw)
    rot[:, 2, 1] = 2 * (qyqz + qxqw)
    rot[:, 2, 2] = 1 - 2 * (qx2 + qy2)

    return rot[0] if single else rot


def quat2axis(quat: np.ndarray) -> Tuple[np.ndarray, float]:
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat2axis quat shape err"

    single = quat.ndim == 1
    if single:
        quat = quat[np.newaxis, :]

    # Normalize using quat_norm
    q = quat_norm(quat)

    vec = q[:, 1:]
    norm_vec = np.linalg.norm(vec, axis=1)
    small_mask = norm_vec < 1e-6

    axis = np.zeros((q.shape[0], 3))
    axis[~small_mask] = vec_norm(vec[~small_mask])
    axis[small_mask] = np.array([1.0, 0.0, 0.0])

    theta = np.zeros(q.shape[0])
    theta[~small_mask] = 2 * np.arctan2(norm_vec[~small_mask], q[~small_mask,
                                                                 0])

    return (axis[0], theta[0]) if single else (axis, theta)


def quat2so3(quat: np.ndarray) -> np.ndarray:
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat2so3 quat shape err"

    single = quat.ndim == 1
    axis, theta = quat2axis(quat)

    return theta * axis if single else theta[:, np.newaxis] * axis


def axis2rot(axis: np.ndarray, theta) -> np.ndarray:
    assert (axis.ndim == 1 and axis.shape == (3, )) or (
        axis.ndim == 2 and axis.shape[1] == 3), "axis2rot axis shape err"

    single = axis.ndim == 1
    if single:
        assert isinstance(
            theta, (int, float)), "axis2rot theta must be scalar for 1D axis"
        axis = axis[np.newaxis, :]
        theta = np.array([theta])
    else:
        assert (isinstance(theta, np.ndarray) and theta.ndim == 1 and
                theta.shape[0] == axis.shape[0]) or isinstance(theta, (int, float)), \
            "axis2rot theta shape err"
        if isinstance(theta, (int, float)):
            theta = np.full(axis.shape[0], theta)

    # Normalize axis using vec_norm
    normed_axis = vec_norm(axis)

    num = axis.shape[0]
    rot = np.zeros((num, 3, 3))
    small_mask = theta < 1e-6
    rot[small_mask] = np.eye(3)

    if np.any(~small_mask):
        axis_matrix = hat(normed_axis[~small_mask])
        sin_theta = np.sin(theta[~small_mask])
        cos_theta = np.cos(theta[~small_mask])
        # Batch matrix multiplication
        axis_matrix_sq = axis_matrix @ axis_matrix
        rot[~small_mask] = (
            np.eye(3)[np.newaxis, :, :] +
            sin_theta[:, np.newaxis, np.newaxis] * axis_matrix +
            (1 - cos_theta)[:, np.newaxis, np.newaxis] * axis_matrix_sq)

    return rot[0] if single else rot


def axis2quat(axis: np.ndarray, theta) -> np.ndarray:
    assert (axis.ndim == 1 and axis.shape == (3, )) or (
        axis.ndim == 2 and axis.shape[1] == 3), "axis2quat axis shape err"

    single = axis.ndim == 1
    if single:
        assert isinstance(
            theta, (int, float)), "axis2quat theta must be scalar for 1D axis"
        axis = axis[np.newaxis, :]
        theta = np.array([theta])
    else:
        assert (isinstance(theta, np.ndarray) and theta.ndim == 1 and
                theta.shape[0] == axis.shape[0]) or isinstance(theta, (int, float)), \
            "axis2quat theta shape err"
        if isinstance(theta, (int, float)):
            theta = np.full(axis.shape[0], theta)

    # Normalize axis using vec_norm
    normed_axis = vec_norm(axis)

    num = axis.shape[0]
    quat = np.zeros((num, 4))
    small_mask = theta < 1e-6
    quat[small_mask] = np.array([1.0, 0.0, 0.0, 0.0])

    if np.any(~small_mask):
        half_theta = theta[~small_mask] / 2
        quat[~small_mask, 0] = np.cos(half_theta)
        quat[~small_mask,
             1:] = normed_axis[~small_mask] * np.sin(half_theta)[:, np.newaxis]

    return quat[0] if single else quat


def axis2so3(axis: np.ndarray, theta) -> np.ndarray:
    assert (axis.ndim == 1 and axis.shape == (3, )) or (
        axis.ndim == 2 and axis.shape[1] == 3), "axis2so3 axis shape err"

    single = axis.ndim == 1
    if single:
        assert isinstance(
            theta, (int, float)), "axis2so3 theta must be scalar for 1D axis"
        axis = axis[np.newaxis, :]
        theta = np.array([theta])
    else:
        assert (isinstance(theta, np.ndarray) and theta.ndim == 1 and
                theta.shape[0] == axis.shape[0]) or isinstance(theta, (int, float)), \
            "axis2so3 theta shape err"
        if isinstance(theta, (int, float)):
            theta = np.full(axis.shape[0], theta)

    normed_axis = vec_norm(axis)
    so3 = theta[:, np.newaxis] * normed_axis
    return so3[0] if single else so3


def so32rot(so3: np.ndarray) -> np.ndarray:
    assert (so3.ndim == 1 and so3.shape == (3, )) or (
        so3.ndim == 2 and so3.shape[1] == 3), "so32rot so3 shape err"

    single = so3.ndim == 1
    if single:
        so3 = so3[np.newaxis, :]

    theta = np.linalg.norm(so3, axis=1)
    small_mask = theta < 1e-6

    # Normalize axis using vec_norm
    axis = np.zeros_like(so3)
    axis[~small_mask] = vec_norm(so3[~small_mask])
    axis[small_mask] = np.array([1.0, 0.0, 0.0])

    return axis2rot(axis, theta)


def so32quat(so3: np.ndarray) -> np.ndarray:
    assert (so3.ndim == 1 and so3.shape == (3, )) or (
        so3.ndim == 2 and so3.shape[1] == 3), "so32quat so3 shape err"

    axis, theta = so32axis(so3)
    return axis2quat(axis, theta)


def so32axis(so3: np.ndarray) -> Tuple[np.ndarray, float]:
    assert (so3.ndim == 1 and so3.shape == (3, )) or (
        so3.ndim == 2 and so3.shape[1] == 3), "so32axis so3 shape err"

    single = so3.ndim == 1
    if single:
        so3 = so3[np.newaxis, :]

    theta = np.linalg.norm(so3, axis=1)
    small_mask = theta < 1e-6

    # Normalize axis using vec_norm
    axis = np.zeros((so3.shape[0], 3))
    axis[~small_mask] = vec_norm(so3[~small_mask])
    axis[small_mask] = np.array([1.0, 0.0, 0.0])

    return (axis[0], theta[0]) if single else (axis, theta)


def trans2part(trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert (trans.ndim == 2 and trans.shape == (4, 4)) or (
        trans.ndim == 3
        and trans.shape[1:] == (4, 4)), "trans2part trans shape err"

    single = trans.ndim == 2
    if single:
        trans = trans[np.newaxis, ...]

    # Batch processing: input shape (N, 4, 4) -> output (N, 3), (N, 4)
    pos = trans[:, :3, 3]
    quat = rot2quat(trans[:, :3, :3])

    return (pos[0], quat[0]) if single else (pos, quat)


def trans2se3(trans: np.ndarray) -> np.ndarray:
    assert (trans.ndim == 2 and trans.shape == (4, 4)) or (
        trans.ndim == 3
        and trans.shape[1:] == (4, 4)), "trans2se3 trans shape err"

    single = trans.ndim == 2
    if single:
        trans = trans[np.newaxis, ...]

    # Batch processing: input shape (N, 4, 4) -> output shape (N, 6)
    pos = trans[:, :3, 3]
    so3 = rot2so3(trans[:, :3, :3])
    se3 = np.concatenate((pos, so3), axis=1)
    return se3[0] if single else se3


def part2trans(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    assert (pos.ndim == 1 and pos.shape == (3, )) or (
        pos.ndim == 2 and pos.shape[1] == 3), "part2trans pos shape err"
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "part2trans quat shape err"
    assert pos.ndim == quat.ndim, "part2trans pos and quat must have same ndim"
    if pos.ndim == 2:
        assert pos.shape[0] == quat.shape[0], "part2trans batch size mismatch"

    single = pos.ndim == 1
    if single:
        pos = pos[np.newaxis, :]
        quat = quat[np.newaxis, :]

    num = pos.shape[0]
    trans = np.zeros((num, 4, 4))
    trans[:, :3, 3] = pos
    trans[:, :3, :3] = quat2rot(quat)
    trans[:, 3, :3] = 0.0
    trans[:, 3, 3] = 1.0
    return trans[0] if single else trans


def part2se3(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    assert (pos.ndim == 1 and pos.shape == (3, )) or (
        pos.ndim == 2 and pos.shape[1] == 3), "part2se3 pos shape err"
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "part2se3 quat shape err"
    assert pos.ndim == quat.ndim, "part2se3 pos and quat must have same ndim"
    if pos.ndim == 2:
        assert pos.shape[0] == quat.shape[0], "part2se3 batch size mismatch"

    single = pos.ndim == 1
    if single:
        pos = pos[np.newaxis, :]
        quat = quat[np.newaxis, :]

    so3 = quat2so3(quat)
    se3 = np.concatenate((pos, so3), axis=1)
    return se3[0] if single else se3


def se32trans(se3: np.ndarray) -> np.ndarray:
    assert (se3.ndim == 1 and se3.shape == (6, )) or (
        se3.ndim == 2 and se3.shape[1] == 6), "se32trans se3 shape err"

    single = se3.ndim == 1
    if single:
        se3 = se3[np.newaxis, :]

    num = se3.shape[0]
    trans = np.zeros((num, 4, 4))
    trans[:, :3, 3] = se3[:, :3]
    trans[:, :3, :3] = so32rot(se3[:, 3:])
    trans[:, 3, :3] = 0.0
    trans[:, 3, 3] = 1.0
    return trans[0] if single else trans


def se32part(se3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert (se3.ndim == 1 and se3.shape == (6, )) or (
        se3.ndim == 2 and se3.shape[1] == 6), "se32part se3 shape err"

    single = se3.ndim == 1
    if single:
        se3 = se3[np.newaxis, :]

    pos = se3[:, :3]
    quat = so32quat(se3[:, 3:])
    return (pos[0], quat[0]) if single else (pos, quat)


def yaw2quat(yaw) -> np.ndarray:
    if isinstance(yaw, (int, float)):
        quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        return quat
    else:
        # Batch processing: input shape (N,) -> output shape (N, 4)
        assert isinstance(
            yaw, np.ndarray) and yaw.ndim == 1, "yaw2quat yaw shape err"
        quat = np.zeros((yaw.shape[0], 4))
        quat[:, 0] = np.cos(yaw / 2)
        quat[:, 3] = np.sin(yaw / 2)
        return quat


def quat2yaw(quat: np.ndarray):
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat2yaw quat shape err"

    single = quat.ndim == 1
    if single:
        quat = quat[np.newaxis, :]

    # Batch processing: input shape (N, 4) -> output shape (N,)
    yaw = 2 * np.arctan2(quat[:, 3], quat[:, 0])

    return yaw[0] if single else yaw


def single_euler2rot(theta: float | np.ndarray, format: str) -> np.ndarray:
    assert format in ['x', 'y',
                      'z'], f"single_euler_rot format '{format}' not supported"

    single = np.isscalar(theta)
    if single:
        theta = np.array([theta])

    rot = np.zeros((theta.shape[0], 3, 3))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    if format == 'x':
        rot[:, 0, 0] = 1
        rot[:, 1, 1] = cos_theta
        rot[:, 1, 2] = -sin_theta
        rot[:, 2, 1] = sin_theta
        rot[:, 2, 2] = cos_theta
    elif format == 'y':
        rot[:, 0, 0] = cos_theta
        rot[:, 0, 2] = sin_theta
        rot[:, 1, 1] = 1
        rot[:, 2, 0] = -sin_theta
        rot[:, 2, 2] = cos_theta
    elif format == 'z':
        rot[:, 0, 0] = cos_theta
        rot[:, 0, 1] = -sin_theta
        rot[:, 1, 0] = sin_theta
        rot[:, 1, 1] = cos_theta
        rot[:, 2, 2] = 1

    return rot[0] if single else rot


def single_rot2euler(rot: np.ndarray, format: str) -> np.ndarray:
    assert format in ['x', 'y',
                      'z'], f"single_rot2euler format '{format}' not supported"
    assert (rot.ndim == 2 and rot.shape == (3, 3)) or (
        rot.ndim == 3
        and rot.shape[1:] == (3, 3)), "single_rot2euler rot shape err"
    single = rot.ndim == 2
    if single:
        rot = rot[np.newaxis, ...]

    theta = np.zeros((rot.shape[0], 3))
    if format == 'x':
        theta[:, 0] = np.arctan2(rot[:, 2, 1], rot[:, 1, 1])
    elif format == 'y':
        theta[:, 1] = np.arctan2(rot[:, 0, 2], rot[:, 0, 0])
    elif format == 'z':
        theta[:, 2] = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])

    return theta[0] if single else theta


def euler2rot(euler: np.ndarray, format: str = 'xyx') -> np.ndarray:
    """
    format: 'xyx', 'xyz', 'xzx', 'xzy', 'yxy', 'yxz', 'yzx', 'yzy', 'zxy', 'zxz', 'zyx', 'zyz'
    """
    format = format.lower()
    assert format in [
        'xyx', 'xyz', 'xzx', 'xzy', 'yxy', 'yxz', 'yzx', 'yzy', 'zxy', 'zxz',
        'zyx', 'zyz'
    ], f"euler2rot format '{format}' not supported"

    assert (euler.ndim == 1 and euler.shape == (3, )) or (
        euler.ndim == 2 and euler.shape[1] == 3), "euler2rot euler shape err"

    single = euler.ndim == 1
    if single:
        euler = euler[np.newaxis, :]
    theta1, theta2, theta3 = euler[:, 0], euler[:, 1], euler[:, 2]

    rot_1 = single_euler2rot(theta1, format[0])
    rot_2 = single_euler2rot(theta2, format[1])
    rot_3 = single_euler2rot(theta3, format[2])
    result = rot_1 @ rot_2 @ rot_3
    return result[0] if single else result


def rot2euler(
    rot: np.ndarray,
    format: str = 'xyx',
    last_euler: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    format: 'xyx', 'xyz', 'xzx', 'xzy', 'yxy', 'yxz', 'yzx', 'yzy', 'zxy', 'zxz', 'zyx', 'zyz'
    """
    format = format.lower()
    assert format in [
        'xyx', 'xyz', 'xzx', 'xzy', 'yxy', 'yxz', 'yzx', 'yzy', 'zxy', 'zxz',
        'zyx', 'zyz'
    ], f"rot2euler format '{format}' not supported"

    assert (rot.ndim == 2 and rot.shape == (3, 3)) or (
        rot.ndim == 3 and rot.shape[1:] == (3, 3)), "rot2euler rot shape err"
    single = rot.ndim == 2
    if single:
        rot = rot[np.newaxis, ...]
    rot_num = rot.shape[0]
    if last_euler is not None:
        if single:
            assert last_euler.ndim == 1 and last_euler.shape == (
                3, ), "rot2euler last_euler shape err"
            last_euler = last_euler[np.newaxis, :]
        else:
            assert last_euler.ndim == 2 and last_euler.shape[
                1] == 3, "rot2euler last_euler shape err"
            assert last_euler.shape[
                0] == rot_num, "rot2euler last_euler and rot must have same batch size"
    else:
        last_euler = np.zeros((rot_num, 3))

    theta = last_euler.copy()

    # Proper Euler angles
    if format == 'xyx':
        theta[:, 1] = np.arccos(np.clip(rot[:, 0, 0], -1.0, 1.0))
        singular = np.abs(np.sin(theta[:, 1])) < eps
        # Non-singular cases (vectorized)
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(rot[:, 1, 0], -rot[:, 2, 0]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(rot[:, 0, 1], rot[:, 0, 2]))
        # Singular cases (loop only over singular points)
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            theta13 = np.arctan2(rot[i, 2, 1], rot[i, 1, 1])
            theta[i, 2] = theta13 - theta[i, 0]

    elif format == 'xzx':
        theta[:, 1] = np.arccos(np.clip(rot[:, 0, 0], -1.0, 1.0))
        singular = np.abs(np.sin(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(rot[:, 2, 0], rot[:, 1, 0]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(rot[:, 0, 2], -rot[:, 0, 1]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            theta13 = np.arctan2(rot[i, 2, 1], rot[i, 1, 1])
            theta[i, 2] = theta13 - theta[i, 0]

    elif format == 'yxy':
        theta[:, 1] = np.arccos(np.clip(rot[:, 1, 1], -1.0, 1.0))
        singular = np.abs(np.sin(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(rot[:, 0, 1], rot[:, 2, 1]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(rot[:, 1, 0], -rot[:, 1, 2]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            theta13 = np.arctan2(rot[i, 0, 2], rot[i, 0, 0])
            theta[i, 2] = theta13 - theta[i, 0]

    elif format == 'yzy':
        theta[:, 1] = np.arccos(np.clip(rot[:, 1, 1], -1.0, 1.0))
        singular = np.abs(np.sin(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(rot[:, 2, 1], -rot[:, 0, 1]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(rot[:, 1, 2], rot[:, 1, 0]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            theta13 = np.arctan2(rot[i, 0, 2], rot[i, 0, 0])
            theta[i, 2] = theta13 - theta[i, 0]

    elif format == 'zxz':
        theta[:, 1] = np.arccos(np.clip(rot[:, 2, 2], -1.0, 1.0))
        singular = np.abs(np.sin(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(rot[:, 0, 2], -rot[:, 1, 2]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(rot[:, 2, 0], rot[:, 2, 1]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            theta13 = np.arctan2(rot[i, 1, 0], rot[i, 0, 0])
            theta[i, 2] = theta13 - theta[i, 0]

    elif format == 'zyz':
        theta[:, 1] = np.arccos(np.clip(rot[:, 2, 2], -1.0, 1.0))
        singular = np.abs(np.sin(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(rot[:, 1, 2], rot[:, 0, 2]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(rot[:, 2, 1], -rot[:, 2, 0]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            theta13 = np.arctan2(rot[i, 1, 0], rot[i, 0, 0])
            theta[i, 2] = theta13 - theta[i, 0]

    # Tait-Bryan angles
    elif format == 'xyz':
        theta[:, 1] = np.arcsin(np.clip(rot[:, 0, 2], -1.0, 1.0))
        singular = np.abs(np.cos(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(-rot[:, 1, 2], rot[:, 2, 2]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(-rot[:, 0, 1], rot[:, 0, 0]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            if theta[i, 1] > 1.5:
                theta13 = np.arctan2(rot[i, 1, 0], rot[i, 1, 1])
                theta[i, 2] = theta13 - theta[i, 0]
            elif theta[i, 1] < -1.5:
                theta3_1 = np.arctan2(rot[i, 1, 0], rot[i, 1, 1])
                theta[i, 2] = theta3_1 + theta[i, 0]

    elif format == 'xzy':
        theta[:, 1] = np.arcsin(np.clip(-rot[:, 0, 1], -1.0, 1.0))
        singular = np.abs(np.cos(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(rot[:, 2, 1], rot[:, 1, 1]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(rot[:, 0, 2], rot[:, 0, 0]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            if theta[i, 1] > 1.5:
                theta3_1 = np.arctan2(rot[i, 1, 2], rot[i, 1, 0])
                theta[i, 2] = theta3_1 + theta[i, 0]
            elif theta[i, 1] < -1.5:
                theta13 = np.arctan2(-rot[i, 1, 2], -rot[i, 1, 0])
                theta[i, 2] = theta13 - theta[i, 0]

    elif format == 'yxz':
        theta[:, 1] = np.arcsin(np.clip(-rot[:, 1, 2], -1.0, 1.0))
        singular = np.abs(np.cos(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(rot[:, 0, 2], rot[:, 2, 2]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(rot[:, 1, 0], rot[:, 1, 1]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            if theta[i, 1] > 1.5:
                theta3_1 = np.arctan2(rot[i, 2, 0], rot[i, 2, 1])
                theta[i, 2] = theta3_1 + theta[i, 0]
            elif theta[i, 1] < -1.5:
                theta13 = np.arctan2(-rot[i, 2, 0], -rot[i, 2, 1])
                theta[i, 2] = theta13 - theta[i, 0]

    elif format == 'yzx':
        theta[:, 1] = np.arcsin(np.clip(rot[:, 1, 0], -1.0, 1.0))
        singular = np.abs(np.cos(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(-rot[:, 2, 0], rot[:, 0, 0]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(-rot[:, 1, 2], rot[:, 1, 1]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            if theta[i, 1] > 1.5:
                theta13 = np.arctan2(rot[i, 2, 1], rot[i, 2, 2])
                theta[i, 2] = theta13 - theta[i, 0]
            elif theta[i, 1] < -1.5:
                theta3_1 = np.arctan2(rot[i, 2, 1], rot[i, 2, 2])
                theta[i, 2] = theta3_1 + theta[i, 0]

    elif format == 'zxy':
        theta[:, 1] = np.arcsin(np.clip(rot[:, 2, 1], -1.0, 1.0))
        singular = np.abs(np.cos(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(-rot[:, 0, 1], rot[:, 1, 1]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(-rot[:, 2, 0], rot[:, 2, 2]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            if theta[i, 1] > 1.5:
                theta13 = np.arctan2(rot[i, 1, 0], rot[i, 0, 0])
                theta[i, 2] = theta13 - theta[i, 0]
            elif theta[i, 1] < -1.5:
                theta3_1 = np.arctan2(-rot[i, 1, 0], rot[i, 0, 0])
                theta[i, 2] = theta3_1 + theta[i, 0]

    elif format == 'zyx':
        theta[:, 1] = np.arcsin(np.clip(-rot[:, 2, 0], -1.0, 1.0))
        singular = np.abs(np.cos(theta[:, 1])) < eps
        theta[:, 0] = np.where(singular, theta[:, 0],
                               np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))
        theta[:, 2] = np.where(singular, theta[:, 2],
                               np.arctan2(rot[:, 2, 1], rot[:, 2, 2]))
        singular_idx = np.where(singular)[0]
        for i in singular_idx:
            if theta[i, 1] > 1.5:
                theta3_1 = np.arctan2(rot[i, 0, 1], rot[i, 1, 1])
                theta[i, 2] = theta3_1 + theta[i, 0]
            elif theta[i, 1] < -1.5:
                theta13 = np.arctan2(-rot[i, 0, 1], rot[i, 1, 1])
                theta[i, 2] = theta13 - theta[i, 0]

    return theta[0] if single else theta
