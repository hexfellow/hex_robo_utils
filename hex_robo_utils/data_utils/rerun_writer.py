#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import os, subprocess, signal, atexit, threading
import numpy as np
import rerun as rr


class HexRerunWriter:

    def __init__(
        self,
        visualize: bool = True,
    ):
        self.__live_stream: rr.RecordingStream | None = None
        self.__viewer_proc: subprocess.Popen | None = None
        if visualize:
            self.__viewer_proc = subprocess.Popen(
                [
                    "rerun",
                    "--port",
                    "9876",
                    "--memory-limit",
                    "15%",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.__live_stream = rr.RecordingStream(
                "live",
                recording_id="data_live",
            )
            self.__live_stream.connect_grpc(
                f"rerun+http://127.0.0.1:9876/proxy")

        self.__record_stream: rr.RecordingStream | None = None
        self.__stream_lock = threading.Lock()

        atexit.register(self.close)

        def _handler(sig, frame):
            self.close()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def __del__(self):
        self.close()

    def close(self):
        with self.__stream_lock:
            if self.__live_stream is not None:
                print(
                    f"Disconnecting live stream: {self.__live_stream.get_recording_id()}"
                )
                rr.disconnect(recording=self.__live_stream)
                self.__live_stream = None
            if self.__record_stream is not None:
                print(
                    f"Disconnecting record stream: {self.__record_stream.get_recording_id()}"
                )
                rr.disconnect(recording=self.__record_stream)
                self.__record_stream = None

        if self.__viewer_proc is not None and self.__viewer_proc.poll(
        ) is None:
            pgid = os.getpgid(self.__viewer_proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.__viewer_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
            self.__viewer_proc = None

    def append_data(self, data: dict[str, np.ndarray | int | float]):
        ts_ns, log_data = self.__parse_data(data)

        with self.__stream_lock:
            if self.__live_stream is not None:
                self.__log_data(self.__live_stream, ts_ns, log_data)
            if self.__record_stream is not None:
                self.__log_data(self.__record_stream, ts_ns, log_data)

    def start_record(self, path: str, name: str):
        if self.__record_stream is not None:
            return

        with self.__stream_lock:
            self.__record_stream = rr.RecordingStream(
                "record",
                recording_id=name,
            )
            self.__record_stream.save(f"{path}/{name}.rrd")
            print(f"Record started: {path}/{name}.rrd")

    def stop_record(self):
        if self.__record_stream is None:
            return

        with self.__stream_lock:
            rr.disconnect(recording=self.__record_stream)
            print(f"Record stopped: {self.__record_stream.get_recording_id()}")
            self.__record_stream = None

    def __parse_data(
        self, data: dict[str, np.ndarray | int | float]
    ) -> tuple[int, dict[str, rr.Tensor | rr.Image | rr.DepthImage
                         | rr.EncodedImage | rr.Scalars]]:
        ts_ns = 0
        log_data = {}
        for key, value in data.items():
            if key == "ts_ns":
                ts_ns = value
            elif key.endswith("rgb"):
                img = rr.Image(np.ascontiguousarray(value, dtype=np.uint8))
                img = img.compress(jpeg_quality=75)
                log_data[key] = img
            elif key.endswith("depth"):
                log_data[key] = rr.DepthImage(
                    np.ascontiguousarray(value, dtype=np.uint16), )
            elif key.endswith("pos") or key.endswith("vel") or key.endswith(
                    "eff"):
                arr = np.ascontiguousarray(value, dtype=np.float32)
                log_data[key] = rr.Scalars(arr if arr.ndim ==
                                           1 else arr.ravel())
            else:
                log_data[key] = rr.Tensor(
                    np.ascontiguousarray(value, dtype=np.float32), )
        return ts_ns, log_data

    def __log_data(self, stream: rr.RecordingStream, ts_ns: int,
                   data: dict[str, rr.Tensor | rr.Image | rr.DepthImage
                              | rr.EncodedImage | rr.Scalars]):
        stream.set_time("ts_ns", duration=np.timedelta64(ts_ns, "ns"))
        for key, value in data.items():
            stream.log(key, value)
