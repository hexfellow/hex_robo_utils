#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-11-18
################################################################

import numpy as np
import pytest

from hex_robo_utils import math_utils as mu


def test_hat_and_vee_single_and_batch():
    vec = np.array([1.0, 2.0, 3.0])
    expected = np.array([
        [0.0, -3.0, 2.0],
        [3.0, 0.0, -1.0],
        [-2.0, 1.0, 0.0],
    ])
    assert np.allclose(mu.hat(vec), expected)

    batch_vec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mats = mu.hat(batch_vec)
    assert mats.shape == (2, 3, 3)
    assert np.allclose(mu.vee(mats), batch_vec)


def test_angle_conversions_and_norm():
    assert mu.rad2deg(np.pi) == pytest.approx(180.0)
    assert mu.deg2rad(180.0) == pytest.approx(np.pi)
    assert mu.angle_norm(3 * np.pi) == pytest.approx(-np.pi)


def test_vec_and_quat_norm():
    vec = np.array([3.0, 4.0, 0.0])
    assert np.allclose(mu.vec_norm(vec), np.array([0.6, 0.8, 0.0]))

    quat = np.array([-2.0, 0.0, 0.0, 0.0])
    normed = mu.quat_norm(quat)
    assert np.allclose(normed, np.array([1.0, 0.0, 0.0, 0.0]))


def test_quat_slerp_midpoint():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.0, 0.0, 0.0, 1.0])
    result = mu.quat_slerp(q1, q2, 0.5)
    expected = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    assert np.allclose(result, expected, atol=1e-6)


def test_quat_mul_and_inv_identity():
    yaw_quat = mu.yaw2quat(np.pi / 2)
    inv = mu.quat_inv(yaw_quat)
    identity = mu.quat_mul(yaw_quat, inv)
    assert np.allclose(identity, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


def test_trans_inv_round_trip():
    pos = np.array([1.0, -2.0, 0.5])
    rot = mu.axis2rot(np.array([0.0, 0.0, 1.0]), np.pi / 4)
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, 3] = pos

    inv = mu.trans_inv(trans)
    identity = trans @ inv
    assert np.allclose(identity, np.eye(4), atol=1e-6)


def test_rotation_quaternion_conversions():
    axis = np.array([0.0, 0.0, 1.0])
    theta = np.pi / 3
    rot = mu.axis2rot(axis, theta)
    quat = mu.rot2quat(rot)
    recon_rot = mu.quat2rot(quat)
    assert np.allclose(rot, recon_rot, atol=1e-6)


def test_rot_axis_so3_conversions():
    axis = np.array([0.0, 1.0, 0.0])
    theta = 0.3
    rot = mu.axis2rot(axis, theta)
    axis_out, theta_out = mu.rot2axis(rot)
    assert np.allclose(theta_out, theta, atol=1e-6)
    assert np.allclose(axis_out, axis, atol=1e-6)

    so3 = mu.rot2so3(rot)
    rot_from_so3 = mu.so32rot(so3)
    assert np.allclose(rot_from_so3, rot, atol=1e-6)


def test_quaternion_axis_so3_conversions():
    axis = np.array([1.0, 0.0, 0.0])
    theta = 0.4
    quat = mu.axis2quat(axis, theta)
    axis_out, theta_out = mu.quat2axis(quat)
    assert np.allclose(theta_out, theta, atol=1e-6)
    assert np.allclose(axis_out, axis, atol=1e-6)

    so3 = mu.quat2so3(quat)
    quat_from_so3 = mu.so32quat(so3)
    assert np.allclose(quat_from_so3, mu.quat_norm(quat), atol=1e-6)

    axis_from_so3, theta_from_so3 = mu.so32axis(so3)
    assert np.allclose(theta_from_so3, theta, atol=1e-6)
    assert np.allclose(axis_from_so3, axis, atol=1e-6)


def test_axis2so3_and_vec_norm_integration():
    axis = np.array([0.0, 2.0, 0.0])
    theta = 0.25
    so3 = mu.axis2so3(axis, theta)
    assert np.allclose(so3, np.array([0.0, 0.25, 0.0]), atol=1e-6)


def test_transform_and_se3_conversions():
    pos = np.array([0.2, -0.1, 0.3])
    quat = mu.axis2quat(np.array([0.0, 0.0, 1.0]), 0.2)
    trans = mu.part2trans(pos, quat)

    pos_out, quat_out = mu.trans2part(trans)
    assert np.allclose(pos_out, pos, atol=1e-6)
    assert np.allclose(quat_out, mu.quat_norm(quat), atol=1e-6)

    se3 = mu.trans2se3(trans)
    trans_from_se3 = mu.se32trans(se3)
    assert np.allclose(trans_from_se3, trans, atol=1e-6)

    se3_from_part = mu.part2se3(pos, quat)
    pos_se3, quat_se3 = mu.se32part(se3_from_part)
    assert np.allclose(pos_se3, pos, atol=1e-6)
    assert np.allclose(quat_se3, mu.quat_norm(quat), atol=1e-6)


def test_yaw_quaternion_conversion():
    yaw = 0.7
    quat = mu.yaw2quat(yaw)
    recovered = mu.quat2yaw(quat)
    assert recovered == pytest.approx(yaw, abs=1e-6)


def test_single_euler_helpers():
    theta = np.pi / 6
    rot_x = mu.single_euler2rot(theta, 'x')
    recovered = mu.single_rot2euler(rot_x, 'x')
    assert recovered[0] == pytest.approx(theta, abs=1e-6)

    # Vector input path
    theta_vec = np.array([theta, theta])
    rot_batch = mu.single_euler2rot(theta_vec, 'z')
    assert rot_batch.shape == (2, 3, 3)


def test_euler_rot_round_trip():
    euler = np.array([0.1, -0.2, 0.3])
    rot = mu.euler2rot(euler, format='xyz')
    recovered = mu.rot2euler(rot, format='xyz')
    assert np.allclose(recovered, euler, atol=1e-6)


def test_twist_swing_decomposition():
    twist_true = mu.axis2quat(np.array([0.0, 0.0, 1.0]), 0.3)
    swing_true = mu.axis2quat(np.array([1.0, 0.0, 0.0]), 0.4)
    quat = mu.quat_mul(swing_true, twist_true)

    twist, swing = mu.twist_swing_decomp(quat, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(twist, mu.quat_norm(twist_true), atol=1e-6)
    assert np.allclose(swing, mu.quat_norm(swing_true), atol=1e-6)
