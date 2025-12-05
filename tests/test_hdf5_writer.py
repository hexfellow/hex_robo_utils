#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-12-04
################################################################

import os
import time
import threading
import numpy as np

try:
    from hex_robo_utils.hdf5_writer import HexHdf5MultiWriter
except ImportError:
    import sys
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hex_robo_utils.hdf5_writer import HexHdf5MultiWriter


class HexRate:

    def __init__(self, hz: float, spin_threshold_ns: int = 1_000_000):
        if hz <= 0:
            raise ValueError("hz must be greater than 0")
        if spin_threshold_ns < 0:
            raise ValueError("spin_threshold_ns must be non-negative")
        self.__period_ns = int(1_000_000_000 / hz)
        self.__next_ns = self.__now_ns() + self.__period_ns
        self.__spin_threshold_ns = spin_threshold_ns

    @staticmethod
    def __now_ns() -> int:
        return time.perf_counter_ns()

    def reset(self):
        self.__next_ns = self.__now_ns() + self.__period_ns

    def sleep(self):
        target_ns = self.__next_ns
        now_ns = self.__now_ns()
        remain_ns = target_ns - now_ns
        if remain_ns <= 0:
            needed_period = (now_ns - target_ns) // self.__period_ns + 1
            self.__next_ns += needed_period * self.__period_ns
            return

        spin_threshold = min(self.__spin_threshold_ns, self.__period_ns)
        coarse_sleep_ns = remain_ns - spin_threshold
        if coarse_sleep_ns > 0:
            time.sleep(coarse_sleep_ns / 1_000_000_000.0)

        while True:
            now_ns = self.__now_ns()
            if now_ns >= target_ns:
                break
            if target_ns - now_ns > 50_000:
                time.sleep(0)

        self.__next_ns += self.__period_ns


class MultiArmRGBDRecorder:

    def __init__(
        self,
        base_dir: str,
        duration_s: float = 30.0,
        num_arms: int = 6,
        num_cams: int = 4,
        arm_hz: int = 1000,
        cam_hz: int = 30,
    ):

        self.duration_ns = int(duration_s * 1_000_000_000)
        self.num_arms = num_arms
        self.num_cams = num_cams
        self.arm_hz = arm_hz
        self.cam_hz = cam_hz

        # 使用多文件 writer，将不同类型数据写入不同的 h5 文件
        self._writer = HexHdf5MultiWriter(base_dir)

        self.arm_shape = (7, 3)
        self.arm_dtype = np.float64
        self.rgb_shape = (480, 640, 3)
        self.rgb_dtype = np.uint8
        self.depth_shape = (480, 640)
        self.depth_dtype = np.uint16

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._start_time_ns: int | None = None

        self._create_datasets()

    # ----------------------- public API -----------------------

    def start(self):
        if self._start_time_ns is not None:
            return
        self._stop_event.clear()
        self._writer.start()
        # 统一使用 perf_counter_ns 作为时间基准
        self._start_time_ns = time.perf_counter_ns()

        for arm_id in range(self.num_arms):
            t = threading.Thread(target=self._arm_thread,
                                 args=(arm_id, ),
                                 daemon=True)
            self._threads.append(t)
            t.start()

        for cam_id in range(self.num_cams):
            t = threading.Thread(target=self._rgbd_thread,
                                 args=(cam_id, ),
                                 daemon=True)
            self._threads.append(t)
            t.start()

    def wait(self):
        if self._start_time_ns is None:
            return

        end_time = self._start_time_ns + self.duration_ns
        while (not self._stop_event.is_set()
               ) and time.perf_counter_ns() < end_time:
            time.sleep(0.1)

        self.stop()

    def stop(self):
        """停止所有线程并关闭 writer。"""
        if self._start_time_ns is None:
            return

        self._stop_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join()
        self._threads.clear()

        self._writer.stop()
        self._start_time_ns = None

    def run(self):
        self.start()
        self.wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # --------------------- internal helpers --------------------

    def _create_datasets(self):
        for arm_id in range(self.num_arms):
            group = f"arm_{arm_id}"
            self._writer.create_dataset(
                "robot",
                group,
                shape=self.arm_shape,
                dtype=self.arm_dtype,
                chunk_num=1024,
                max_num=None,
            )

        for cam_id in range(self.num_cams):
            rgb_group = f"cam_{cam_id}_rgb"
            depth_group = f"cam_{cam_id}_depth"

            self._writer.create_dataset(
                "rgb",
                rgb_group,
                shape=self.rgb_shape,
                dtype=self.rgb_dtype,
                chunk_num=1,
                max_num=None,
            )
            self._writer.create_dataset(
                "depth",
                depth_group,
                shape=self.depth_shape,
                dtype=self.depth_dtype,
                chunk_num=1,
                max_num=None,
            )

    def _time_remain(self) -> bool:
        if self._start_time_ns is None:
            return False
        return (time.perf_counter_ns() -
                self._start_time_ns) < self.duration_ns

    def _arm_thread(self, arm_id: int):
        group = f"arm_{arm_id}"
        hex_rate = HexRate(self.arm_hz)
        while (not self._stop_event.is_set()) and self._time_remain():
            data = np.random.randn(*self.arm_shape).astype(self.arm_dtype)
            get_ts = self._writer.now_ns()
            sen_ts = self._writer.now_ns()
            self._writer.append_data("robot", group, data, get_ts, sen_ts)
            hex_rate.sleep()

    def _rgbd_thread(self, cam_id: int):
        rgb_group = f"cam_{cam_id}_rgb"
        depth_group = f"cam_{cam_id}_depth"
        hex_rate = HexRate(self.cam_hz)
        while (not self._stop_event.is_set()) and self._time_remain():
            rgb = np.random.randint(
                0,
                256,
                size=self.rgb_shape,
                dtype=self.rgb_dtype,
            )
            depth = np.random.randint(
                0,
                65536,
                size=self.depth_shape,
                dtype=self.depth_dtype,
            )
            get_ts = self._writer.now_ns()
            sen_ts = self._writer.now_ns()
            self._writer.append_data("rgb", rgb_group, rgb, get_ts, sen_ts)
            self._writer.append_data("depth", depth_group, depth, get_ts,
                                     sen_ts)
            hex_rate.sleep()


def main():
    out_path = os.path.abspath("multi_arm_rgbd")
    print(f"Recording base: {out_path}")

    start_ns = time.perf_counter_ns()
    recorder = MultiArmRGBDRecorder(
        out_path,
        duration_s=30.0,
        num_arms=5,
        num_cams=4,
        arm_hz=1000,
        cam_hz=30,
    )
    recorder.run()
    print("#" * 50)
    print(f"Time taken: {(time.perf_counter_ns() - start_ns) * 1e-6}ms")
    print("#" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
