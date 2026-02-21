#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-20
################################################################

import os, subprocess, signal, atexit, threading
import cv2
import numpy as np
import pandas as pd
import rerun as rr


class HexRerunWriterUtil:

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

    def send_data(self, data: dict[str, np.ndarray | int | float]):
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


class HexRerunParserUtil:

    def __init__(self, path: str):
        dir_name = os.path.splitext(os.path.abspath(path))[0]
        if (not os.path.exists(dir_name)) or (len(os.listdir(dir_name)) == 0):
            print(f"Cache not found: {dir_name}, generating pd files...")
            os.makedirs(dir_name, exist_ok=True)
            self.__gen_pd_files(dir_name)

        print(f"Cache found: {dir_name}, loading pd files...")
        self.__df_cache = self.__load_pd_files(dir_name)

    def __del__(self):
        self.close()

    def close(self):
        self.__df_cache.clear()

    def summary(self):
        for key, df in self.__df_cache.items():
            data_name, format_name = self.__parse_columns(key, df)
            print(f"Key: {key}")
            print(f"  Data Name: {data_name}")
            print(f"  Format Name: {format_name}")
            print(f"  Data Number: {len(df[data_name])}")

    def get_keys(self) -> list[str]:
        return self.__df_cache.keys()

    def get_data(self, key: str, use_sen_ts: bool = False) -> np.ndarray:
        key = key if key.startswith("/") else "/" + key

        df = self.__df_cache[key]
        data_name, format_name = self.__parse_columns(key, df)

        ts_name = "ts_ns" if use_sen_ts else "log_time"
        ts_numpy = df[ts_name].to_numpy()
        if ts_numpy.dtype.kind == "M":
            ts_numpy = ts_numpy.astype("datetime64[ns]").astype(np.int64)
        data_series = df[data_name]
        format_series = df[format_name] if format_name is not None else None

        data = np.array([])
        if key.endswith("rgb"):
            if format_series is None:
                raise ValueError(f"Format series not found for key: {key}")

            format_msg = format_series[0][0]
            rgbs = []
            for blob in data_series:
                rgb = cv2.imdecode(blob[0], cv2.IMREAD_COLOR)
                if rgb is not None:
                    rgbs.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            data = np.stack(rgbs)
        elif key.endswith("depth"):
            if format_series is None:
                raise ValueError(f"Format series not found for key: {key}")

            format_msg = format_series[0][
                0] if format_series is not None else None
            width, height = format_msg.get("width",
                                           -1), format_msg.get("height", -1)
            if width <= 0 or height <= 0:
                raise ValueError(f"Width or height not found for key: {key}")

            depths = []
            for blob in data_series:
                depth = np.asarray(blob[0],
                                   dtype=np.uint8).view(np.uint16).reshape(
                                       height, width)
                depths.append(depth)
            data = np.stack(depths)
        else:
            data = np.stack(
                [np.asarray(v, dtype=np.float32) for v in data_series])

        return {"ts_ns": ts_numpy, "data": data}

    def get_all_data(self, use_sen_ts: bool = False) -> dict[str, np.ndarray]:
        return {key: self.get_data(key, use_sen_ts) for key in self.get_keys()}

    @staticmethod
    def __gen_pd_files(dir_name: str) -> None:
        server = rr.server.Server(datasets={"data": [f"{dir_name}.rrd"]})
        try:
            client = rr.catalog.CatalogClient(server.url())
            dataset = client.get_dataset(name="data")

            schema = dataset.schema()
            entity_paths = sorted(
                set(col.entity_path for col in schema.component_columns()))
            key_set = set([p.lstrip("/")
                           for p in entity_paths]) - {"__properties"}
            key_list = list(key_set)
            key_list.sort()

            for key in key_list:
                entity_path = "/" + key
                df = dataset.filter_contents(
                    [entity_path]).reader(index="ts_ns").to_pandas()
                file_name = key.replace("/", "@")
                pd.to_pickle(df, f"{dir_name}/{file_name}.pkl")
        finally:
            server.shutdown()

    @staticmethod
    def __load_pd_files(dir_name: str) -> dict[str, tuple]:
        df_cache = {}
        for key in os.listdir(dir_name):
            if key.endswith(".pkl"):
                df = pd.read_pickle(f"{dir_name}/{key}")
                new_key = "/" + os.path.splitext(key)[0].replace("@", "/")
                df_cache[new_key] = df
        return df_cache

    @staticmethod
    def __parse_columns(key: str, df: pd.DataFrame) -> tuple[str, str | None]:
        data_name = None
        format_name = None
        data_suffixes = (":scalars", ":blob", ":buffer")
        format_suffixes = (":media_type", ":format")
        print(f"df.columns: {df.columns}")
        for col in df.columns:
            if col.startswith(key):
                if col.endswith(data_suffixes):
                    data_name = col
                elif col.endswith(format_suffixes):
                    format_name = col
            if data_name is not None and format_name is not None:
                break
        if data_name is None:
            raise ValueError(f"Data column not found for: {key}")
        return data_name, format_name
