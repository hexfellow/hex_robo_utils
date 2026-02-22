#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-20
################################################################

import os, threading
import cv2
import numpy as np

try:
    from hex_robo_utils import HexRate
    from hex_robo_utils import HexRerunWriter
except ImportError:
    import sys
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hex_robo_utils import HexRate
    from hex_robo_utils import HexRerunWriter


def main():
    rerun_writer = HexRerunWriter(visualize=True)

    list_theta_table = []
    tau = 2.0 * np.pi
    delta_theta = tau / 1000.0
    jnt_num = 40
    for i in range(jnt_num):
        cur_delta = delta_theta / (i + 1)
        list_theta_table.append(np.arange(0.0, tau, cur_delta))
    robot_rate_hz = 1000.0
    cam_rate_hz = 33.0
    duration_s = 30

    import time, os
    out_path = os.path.abspath("multi_arm_rgbd")
    os.makedirs(out_path, exist_ok=True)

    def robot_func(start_ns: int,
                   rerun_writer: HexRerunWriter,
                   list_theta_table: list[np.ndarray],
                   rate_hz: float = 1000.0,
                   duration_s: int = 20):
        jnt_num = len(list_theta_table)
        rate = HexRate(rate_hz)
        for i in range(int(duration_s * rate_hz)):
            joint_theta = np.array([
                list_theta_table[j][i % len(list_theta_table[j])]
                for j in range(jnt_num)
            ])
            data = {
                "ts_ns": time.perf_counter_ns() - start_ns,
                "jnt/pos": np.sin(joint_theta),
                "jnt/vel": np.cos(joint_theta),
                "jnt/eff": np.zeros_like(joint_theta),
            }
            rerun_writer.append_data(data)
            rate.sleep()

    def cam_func(start_ns: int,
                 rerun_writer: HexRerunWriter,
                 rate_hz: float = 30.0,
                 duration_s: int = 20):
        rate = HexRate(rate_hz)
        for _ in range(int(duration_s * rate_hz)):
            cam_0_rgb = np.random.randint(0,
                                          255, (480, 640, 3),
                                          dtype=np.uint8)
            cam_0_depth = np.random.randint(0,
                                            1000, (480, 640),
                                            dtype=np.uint16)
            cam_1_rgb = np.random.randint(0,
                                          255, (480, 640, 3),
                                          dtype=np.uint8)
            cam_1_depth = np.random.randint(0,
                                            1000, (480, 640),
                                            dtype=np.uint16)
            cam_2_rgb = np.random.randint(0,
                                          255, (480, 640, 3),
                                          dtype=np.uint8)
            cam_2_depth = np.random.randint(0,
                                            1000, (480, 640),
                                            dtype=np.uint16)
            cam_3_rgb = np.random.randint(0,
                                          255, (480, 640, 3),
                                          dtype=np.uint8)
            cam_3_depth = np.random.randint(0,
                                            1000, (480, 640),
                                            dtype=np.uint16)
            cv2.rectangle(cam_0_rgb, (240, 160), (400, 320), (255, 0, 255), -1)
            cv2.rectangle(cam_0_depth, (240, 160), (400, 320), 50, -1)
            cv2.rectangle(cam_1_rgb, (240, 160), (400, 320), (255, 0, 0), -1)
            cv2.rectangle(cam_1_depth, (240, 160), (400, 320), 100, -1)
            cv2.rectangle(cam_2_rgb, (240, 160), (400, 320), (0, 255, 0), -1)
            cv2.rectangle(cam_2_depth, (240, 160), (400, 320), 200, -1)
            cv2.rectangle(cam_3_rgb, (240, 160), (400, 320), (0, 0, 255), -1)
            cv2.rectangle(cam_3_depth, (240, 160), (400, 320), 400, -1)
            data = {
                "ts_ns": time.perf_counter_ns() - start_ns,
                "cam_0/rgb": cam_0_rgb,
                "cam_0/depth": cam_0_depth,
                "cam_1/rgb": cam_1_rgb,
                "cam_1/depth": cam_1_depth,
                "cam_2/rgb": cam_2_rgb,
                "cam_2/depth": cam_2_depth,
                "cam_3/rgb": cam_3_rgb,
                "cam_3/depth": cam_3_depth,
            }
            rerun_writer.append_data(data)
            rate.sleep()

    start_ns = time.perf_counter_ns()
    rerun_writer.start_record(out_path, "rerun_data")

    thread_list = [
        threading.Thread(
            target=robot_func,
            args=(
                start_ns,
                rerun_writer,
                list_theta_table,
                robot_rate_hz,
                duration_s,
            ),
        ),
        threading.Thread(
            target=cam_func,
            args=(
                start_ns,
                rerun_writer,
                cam_rate_hz,
                duration_s,
            ),
        )
    ]
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()

    rerun_writer.stop_record()
    print(f"Time taken: {(time.perf_counter_ns() - start_ns) * 1e-9}s")


if __name__ == '__main__':
    main()
