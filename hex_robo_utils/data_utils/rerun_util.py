#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import os, subprocess, signal, atexit, threading
import cv2
import numpy as np
import rerun as rr
import pandas as pd
from collections import deque

from .data_base import HexDataWriterBase
from ..time_utils import ns_now, HexRate
from ..common_utils import hex_rmtree


class HexRerunWriter(HexDataWriterBase):

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
            self.__live_time = ns_now()

        self.__record_stream: rr.RecordingStream | None = None
        self.__stream_lock = threading.Lock()
        self.__record_time = ns_now()

        atexit.register(self.close)

        def _handler(sig, frame):
            self.close()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

        self.__send_thread = threading.Thread(
            target=self.__send_loop,
            daemon=True,
        )
        self.__send_queue = deque(maxlen=2000)
        self.__send_event = threading.Event()
        self.__send_event.set()
        self.__send_thread.start()

    def close(self):
        self.__send_event.clear()
        self.__send_thread.join()

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
        self.__send_queue.append(data)

    def start_record(self, path: str, name: str):
        if self.__record_stream is not None:
            print(f"Record already started")
            return

        with self.__stream_lock:
            self.__record_stream = rr.RecordingStream(
                "record",
                recording_id=name,
            )
            self.__record_stream.save(f"{path}/{name}.rrd")
            print(f"Record started: {path}/{name}.rrd")
        self.__record_time = ns_now()

    def stop_record(self):
        if self.__record_stream is None:
            print(f"Record not started")
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
                ts_ns = int(value)
            elif key.endswith("rgb"):
                img = rr.Image(np.ascontiguousarray(value, dtype=np.uint8),
                               color_model=rr.ColorModel.BGR)
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

    def __send_loop(self):
        rate = HexRate(2e3)
        while self.__send_event.is_set():
            rate.sleep()

            while True:
                try:
                    data = self.__send_queue.popleft()
                    ts_ns, log_data = self.__parse_data(data)
                    with self.__stream_lock:
                        if self.__live_stream is not None:
                            self.__log_data(
                                self.__live_stream,
                                ts_ns - self.__live_time,
                                log_data,
                            )
                        if self.__record_stream is not None:
                            self.__log_data(
                                self.__record_stream,
                                ts_ns - self.__record_time,
                                log_data,
                            )
                except IndexError:
                    break


def rerun_to_pd(rrd_path: str,
                pd_dir: str = None,
                quiet: bool = True,
                remove_old: bool = False) -> None:
    if pd_dir is None:
        pd_dir = rrd_path

    if remove_old:
        hex_rmtree(pd_dir)

    if os.path.exists(pd_dir) and any(
            f.endswith('.pkl') for f in os.listdir(pd_dir)):
        if not quiet:
            print(
                f"Found pd cache files in {pd_dir}. You can use them directly."
            )
        return

    if not quiet:
        print(f"Cache not found: {pd_dir}, generating pd files...")
    os.makedirs(pd_dir, exist_ok=True)

    server = rr.server.Server(datasets={"data": [f"{rrd_path}.rrd"]})
    try:
        client = rr.catalog.CatalogClient(server.url())
        dataset = client.get_dataset(name="data")

        schema = dataset.schema()
        entity_paths = sorted(
            set(col.entity_path for col in schema.component_columns()))
        key_set = set([p.lstrip("/") for p in entity_paths]) - {"__properties"}
        key_list = list(key_set)
        key_list.sort()

        for key in key_list:
            entity_path = "/" + key
            df = dataset.filter_contents([entity_path
                                          ]).reader(index="ts_ns").to_pandas()
            _rerun_pd_post_process(df, key, pd_dir, quiet)
    finally:
        server.shutdown()


def _rerun_pd_post_process(
    df: pd.DataFrame,
    key: str,
    pd_dir: str,
    quiet: bool = True,
) -> pd.DataFrame:
    data_name = None
    format_name = None
    data_suffixes = (":scalars", ":blob", ":buffer")
    format_suffixes = (":media_type", ":format")
    if not quiet:
        print(f"df.columns: {df.columns}")
    for col in df.columns:
        if col.startswith(f"/{key}"):
            if col.endswith(data_suffixes):
                data_name = col
            elif col.endswith(format_suffixes):
                format_name = col
        if data_name is not None and format_name is not None:
            break
    if data_name is None:
        raise ValueError(f"Data column not found for: {key}")

    sen_ts = df["ts_ns"].to_numpy()
    sen_ts = sen_ts.astype("datetime64[ns]").astype(np.int64)
    get_ts = df["log_time"].to_numpy()
    get_ts = get_ts.astype("datetime64[ns]").astype(np.int64)
    data_series = df[data_name]
    if key.endswith("depth"):
        format_msg = df[format_name][0][0]
        width, height = format_msg.get("width",
                                       -1), format_msg.get("height", -1)
        if width <= 0 or height <= 0:
            raise ValueError(f"Width or height not found for key: {key}")
        depths = []
        for blob in data_series:
            depth = np.asarray(blob[0],
                               dtype=np.uint8).view(np.uint16).reshape(
                                   height, width)
            depths.append([depth])
        data_series = pd.Series(depths)
    final_df = pd.DataFrame({
        "sen_ts": sen_ts,
        "get_ts": get_ts,
        "data": data_series,
    })

    file_name = key.replace("/", "@")
    pd.to_pickle(final_df, f"{pd_dir}/{file_name}.pkl")
