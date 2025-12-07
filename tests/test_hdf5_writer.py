#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-12-04
################################################################

import os
import time
import multiprocessing
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

    def __init__(self, hz: float, spin_threshold_ns: int = 10_000):
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

        self._processes: list[multiprocessing.Process] = []
        self._stop_event = multiprocessing.Event()
        self._start_time_ns: int | None = None
        self._manager = None
        self._data_queue = None
        self._queue_thread = None

        self._create_datasets()

    # ----------------------- public API -----------------------

    def start(self):
        if self._start_time_ns is not None:
            return
        self._stop_event.clear()
        self._writer.start()
        # 统一使用 perf_counter_ns 作为时间基准
        self._start_time_ns = time.perf_counter_ns()

        # 创建 Manager 用于进程间共享
        self._manager = multiprocessing.Manager()
        shared_start_time = self._manager.Value('i', self._start_time_ns)
        self._data_queue = self._manager.Queue()

        # 启动队列处理线程
        self._queue_thread = threading.Thread(target=self._queue_worker,
                                              daemon=True)
        self._queue_thread.start()

        for arm_id in range(self.num_arms):
            p = multiprocessing.Process(
                target=self._arm_process,
                args=(arm_id, self._stop_event, shared_start_time,
                      self.duration_ns, self.arm_hz, self.arm_shape,
                      self.arm_dtype, self._data_queue))
            self._processes.append(p)
            p.start()

        for cam_id in range(self.num_cams):
            p = multiprocessing.Process(
                target=self._rgbd_process,
                args=(cam_id, self._stop_event, shared_start_time,
                      self.duration_ns, self.cam_hz, self.rgb_shape,
                      self.rgb_dtype, self.depth_shape, self.depth_dtype,
                      self._data_queue))
            self._processes.append(p)
            p.start()

    def wait(self):
        if self._start_time_ns is None:
            return

        end_time = self._start_time_ns + self.duration_ns
        while (not self._stop_event.is_set()
               ) and time.perf_counter_ns() < end_time:
            time.sleep(0.1)

        self.stop()

    def stop(self):
        """停止所有进程并关闭 writer。"""
        if self._start_time_ns is None:
            return

        self._stop_event.set()
        for p in self._processes:
            if p.is_alive():
                p.join()
        self._processes.clear()

        # 等待队列处理完成
        if self._data_queue is not None:
            # 发送结束标记
            for _ in range(self.num_arms + self.num_cams):
                self._data_queue.put(None)
            if self._queue_thread is not None and self._queue_thread.is_alive(
            ):
                self._queue_thread.join(timeout=5.0)

        self._writer.stop()
        self._start_time_ns = None
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._data_queue = None

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
                "arm",
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

    def _queue_worker(self):
        """从队列中读取数据并写入 HDF5"""
        import queue
        while True:
            try:
                item = self._data_queue.get(timeout=1.0)
                if item is None:  # 结束标记
                    continue
                msg_type, group, data, get_ts, sen_ts = item
                self._writer.append_data(msg_type, group, data, get_ts, sen_ts)
            except queue.Empty:
                # 超时，检查是否应该退出
                if self._stop_event.is_set():
                    # 处理剩余数据
                    while True:
                        try:
                            item = self._data_queue.get_nowait()
                            if item is None:
                                continue
                            msg_type, group, data, get_ts, sen_ts = item
                            self._writer.append_data(msg_type, group, data,
                                                     get_ts, sen_ts)
                        except queue.Empty:
                            break
                    break
            except Exception as e:
                print(f"Queue worker error: {e}")
                if self._stop_event.is_set():
                    break

    @staticmethod
    def _time_remain(shared_start_time, duration_ns) -> bool:
        if shared_start_time.value == 0:
            return False
        return (time.perf_counter_ns() - shared_start_time.value) < duration_ns

    @staticmethod
    def _arm_process(arm_id: int, stop_event: multiprocessing.Event,
                     shared_start_time, duration_ns: int, arm_hz: int,
                     arm_shape: tuple, arm_dtype: np.dtype, data_queue):
        group = f"arm_{arm_id}"
        hex_rate = HexRate(arm_hz)
        fps_cnt = 0
        start_time_ns = time.perf_counter_ns()
        while (not stop_event.is_set()) and MultiArmRGBDRecorder._time_remain(
                shared_start_time, duration_ns):
            data = np.random.randn(*arm_shape).astype(arm_dtype)
            get_ts = time.time_ns()
            sen_ts = time.time_ns()
            # 将数据放入队列
            data_queue.put(("arm", group, data, get_ts, sen_ts))
            fps_cnt += 1
            if fps_cnt >= 3_000:
                delta_s = (time.perf_counter_ns() - start_time_ns) * 1e-9
                print(f"Arm {arm_id} FPS: {fps_cnt / delta_s}Hz")
                fps_cnt = 0
                start_time_ns = time.perf_counter_ns()
            hex_rate.sleep()

    @staticmethod
    def _rgbd_process(cam_id: int, stop_event: multiprocessing.Event,
                      shared_start_time, duration_ns: int, cam_hz: int,
                      rgb_shape: tuple, rgb_dtype: np.dtype,
                      depth_shape: tuple, depth_dtype: np.dtype, data_queue):
        rgb_group = f"cam_{cam_id}_rgb"
        depth_group = f"cam_{cam_id}_depth"
        hex_rate = HexRate(cam_hz)
        fps_cnt = 0
        start_time_ns = time.perf_counter_ns()
        while (not stop_event.is_set()) and MultiArmRGBDRecorder._time_remain(
                shared_start_time, duration_ns):
            rgb = np.random.randint(
                0,
                256,
                size=rgb_shape,
                dtype=rgb_dtype,
            )
            depth = np.random.randint(
                0,
                65536,
                size=depth_shape,
                dtype=depth_dtype,
            )
            get_ts = time.time_ns()
            sen_ts = time.time_ns()
            # 将数据放入队列
            data_queue.put(("rgb", rgb_group, rgb, get_ts, sen_ts))
            data_queue.put(("depth", depth_group, depth, get_ts, sen_ts))
            fps_cnt += 1
            if fps_cnt >= 100:
                delta_s = (time.perf_counter_ns() - start_time_ns) * 1e-9
                print(f"RGBD {cam_id} FPS: {fps_cnt / delta_s}Hz")
                fps_cnt = 0
                start_time_ns = time.perf_counter_ns()
            hex_rate.sleep()


def main():
    out_path = os.path.abspath("multi_arm_rgbd")
    print(f"Recording base: {out_path}")

    start_ns = time.perf_counter_ns()
    recorder = MultiArmRGBDRecorder(
        out_path,
        duration_s=30.0,
        num_arms=6,
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