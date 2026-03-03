#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-09-19
################################################################

import time, os, threading
import h5py
import cv2
import numpy as np
import pandas as pd
from collections import deque
from typing import Any

from .data_base import HexDataWriterBase
from ..time_utils import ns_now


class Hdf5Writer:

    def __init__(
        self,
        file_path: str,
        print_interval: int = 1_000,
        batch_size: int = 64,
    ):
        self.__file_path = file_path
        self.__hdf5_file = h5py.File(file_path, "w", libver='latest')
        self.__group_dict = {}
        self.__dataset_dict = {}
        self.__batch_size = batch_size
        self.__print_num = 0
        self.__print_interval = print_interval

        self.__vlen_groups: set[str] = set()
        self.__queue = deque()
        self.__stop_event = threading.Event()
        self.__writer_cnt = 0
        self.__writer_thread = None
        self.__writer_exc = None

    def __del__(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.__writer_thread and self.__writer_thread.is_alive():
            return
        self.__stop_event.clear()
        self.__writer_thread = threading.Thread(
            target=self.__writer_loop,
            daemon=True,
        )
        self.__writer_thread.start()

    def stop(self):
        if self.__stop_event.is_set():
            return

        self.__stop_event.set()
        if self.__writer_thread is not None and self.__writer_thread.is_alive(
        ):
            self.__writer_thread.join()
        self.summary()
        if self.__hdf5_file is not None:
            try:
                self.__hdf5_file.flush()
                self.__hdf5_file.close()
            except Exception:
                pass
            self.__hdf5_file = None
        if self.__writer_exc:
            raise self.__writer_exc

    def append_data(
        self,
        data_dict: dict[str, Any],
    ):
        self.__queue.append(data_dict)

    def summary(self):
        if self.__hdf5_file is None:
            print("HDF5 file is closed. Cannot get summary.")
            return

        print("#" * 50)
        print(f"HDF5 File: {self.__file_path}")
        print("#" * 50)

        for group_name in self.__group_dict.keys():
            print("-" * 30)
            print(f"Group: {group_name}")
            print("-" * 30)

            dtype = self.__get_dtype(group_name)
            shape = self.__get_shape(group_name)
            print(f"  Dataset: {group_name}")
            print(f"    Dtype: {dtype}")
            print(f"    Shape: {shape}")

            if shape[0] < 2:
                print(f"    GetHz: N/A (< 2 samples)")
                print(f"    SenHz: N/A (< 2 samples)")
                continue

            get_ts = self.__get_dataset_handle(group_name, "get_ts")
            sen_ts = self.__get_dataset_handle(group_name, "sen_ts")
            get_delta_s = ((get_ts[-1] - get_ts[0]) * 1e-9)[0]
            sen_delta_s = ((sen_ts[-1] - sen_ts[0]) * 1e-9)[0]
            print(
                f"    GetHz: {shape[0] / get_delta_s if get_delta_s else 'N/A'}"
            )
            print(
                f"    SenHz: {shape[0] / sen_delta_s if sen_delta_s else 'N/A'}"
            )

        print("#" * 50)

    def __get_shape(self, group_name: str, dataset_name: str = "data"):
        dataset = self.__get_dataset_handle(group_name, dataset_name)
        return dataset.shape

    def __get_dtype(self, group_name: str, dataset_name: str = "data"):
        dataset = self.__get_dataset_handle(group_name, dataset_name)
        return dataset.dtype

    def __get_dataset_handle(
        self,
        group_name: str,
        dataset_name: str = "data",
    ):
        dataset_key = f"{group_name}/{dataset_name}"
        if dataset_key not in self.__dataset_dict:
            if group_name not in self.__hdf5_file:
                raise KeyError(f"Group '{group_name}' not found in HDF5 file")
            if dataset_name not in self.__hdf5_file[group_name]:
                raise KeyError(
                    f"Dataset '{dataset_name}' not found in group '{group_name}'"
                )
            self.__dataset_dict[dataset_key] = self.__hdf5_file[group_name][
                dataset_name]
        return self.__dataset_dict[dataset_key]

    def __drain_batch(self):
        """Pop up to batch_size items from the queue and write them. Returns items written."""
        batch_items = []
        try:
            for _ in range(self.__batch_size):
                batch_items.append(self.__queue.popleft())
        except IndexError:
            pass
        if batch_items:
            self.__write_batch(batch_items)
        return len(batch_items)

    def __writer_loop(self):
        try:
            while not self.__stop_event.is_set():
                if not self.__drain_batch():
                    time.sleep(1e-4)

            # Flush remaining items
            empty_count = 0
            while empty_count < 10:
                if self.__drain_batch():
                    empty_count = 0
                else:
                    empty_count += 1
                    time.sleep(1e-4)
        except Exception as e:
            self.__writer_exc = e
            raise

    def __write_batch(self, batch_items):
        """Write a batch of items, grouped by group_name for efficiency."""
        # Group items by group_name
        grouped_items = {}
        for item in batch_items:
            group = item["group_name"]
            if group not in grouped_items:
                grouped_items[group] = []
            grouped_items[group].append(item)

        # Write each group's batch
        for group, items in grouped_items.items():
            if group not in self.__group_dict:
                print(f"Creating dataset for {group}")
                self.__create_dataset(items[0])
            self.__write_group_batch(group, items)

    def __write_group_batch(self, group_name, items):
        """Write a batch of items for a specific group."""
        dataset_key = f"{group_name}/data"
        get_ts_key = f"{group_name}/get_ts"
        sen_ts_key = f"{group_name}/sen_ts"
        ds = self.__dataset_dict[dataset_key]
        d_get = self.__dataset_dict[get_ts_key]
        d_sen = self.__dataset_dict[sen_ts_key]

        is_vlen = group_name in self.__vlen_groups
        batch_size = len(items)
        n_old = ds.shape[0]
        n_new = n_old + batch_size

        # Resize all datasets once
        ds.resize((n_new, *ds.shape[1:]))
        d_get.resize((n_new, 1))
        d_sen.resize((n_new, 1))

        # Prepare batch arrays
        data_list = []
        gts_list = []
        sts_list = []

        for item in items:
            data, gts, sts = item["data"], int(item["get_ts"]), int(
                item["sen_ts"])
            if is_vlen:
                if isinstance(data, (bytes, bytearray)):
                    data = np.frombuffer(data, dtype=np.uint8)
                elif isinstance(data, np.ndarray) and data.ndim > 1:
                    data = data.ravel()
            data_list.append(data)
            gts_list.append(gts)
            sts_list.append(sts)

        gts_batch = np.array(gts_list, dtype=np.int64).reshape(-1, 1)
        sts_batch = np.array(sts_list, dtype=np.int64).reshape(-1, 1)

        if is_vlen:
            for i, data in enumerate(data_list):
                ds[n_old + i] = data
        else:
            data_batch = np.stack(data_list, axis=0)
            ds[n_old:n_new, ...] = data_batch
        d_get[n_old:n_new, :] = gts_batch
        d_sen[n_old:n_new, :] = sts_batch

        self.__writer_cnt += batch_size
        cur_print_num = self.__writer_cnt // self.__print_interval
        if cur_print_num > self.__print_num:
            self.__print_num = cur_print_num
            print("#" * 50)
            for gn in self.__group_dict.keys():
                print(f"{gn} len:{self.__get_shape(gn)[0]}")

    def __create_dataset(
        self,
        data_dict: dict[str, Any],
    ):
        group_name = data_dict["group_name"]
        if group_name in self.__group_dict:
            return

        self.__group_dict[group_name] = self.__hdf5_file.create_group(
            group_name)

        data_type, chunk_num, data_shape, max_shape, chunks = None, None, None, None, None
        if group_name.endswith("rgb"):
            data_type, chunk_num = h5py.vlen_dtype(np.dtype(np.uint8)), 1
            data_shape, max_shape, chunks = (0, ), (None, ), (chunk_num, )
            self.__vlen_groups.add(group_name)
        elif group_name.endswith("depth"):
            data_type, chunk_num = np.uint16, 1
            single_shape = data_dict["data"].shape
            data_shape, max_shape, chunks = (0, *single_shape), (
                None, *single_shape), (chunk_num, *single_shape)
        elif group_name.endswith("pos") or group_name.endswith(
                "vel") or group_name.endswith("eff"):
            data_type, chunk_num = np.float32, 64
            single_shape = data_dict["data"].shape
            data_shape, max_shape, chunks = (0, *single_shape), (
                None, *single_shape), (chunk_num, *single_shape)
        else:
            raise ValueError(f"Unsupported data type: {group_name}")

        dataset = self.__group_dict[group_name].create_dataset(
            "data",
            shape=data_shape,
            maxshape=max_shape,
            dtype=data_type,
            chunks=chunks,
        )
        get_ts_set = self.__group_dict[group_name].create_dataset(
            "get_ts",
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.int64,
            chunks=(chunk_num, 1),
        )
        sen_ts_set = self.__group_dict[group_name].create_dataset(
            "sen_ts",
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.int64,
            chunks=(chunk_num, 1),
        )

        # Store dataset reference for easy access
        self.__dataset_dict[f"{group_name}/data"] = dataset
        self.__dataset_dict[f"{group_name}/get_ts"] = get_ts_set
        self.__dataset_dict[f"{group_name}/sen_ts"] = sen_ts_set


class HexHdf5Writer(HexDataWriterBase):

    def __init__(self):
        self.__writers: dict[str, Hdf5Writer] | None = None
        self.__writers_ready = threading.Event()
        self.__record_time = ns_now()

    def close(self):
        self.stop_record()

    def append_data(
        self,
        data: dict[str, np.ndarray | bytes],
    ):
        if not self.__writers_ready.is_set():
            return

        ts_ns = int(data.get("ts_ns", None))
        if ts_ns > self.__record_time:
            ts_ns = ts_ns - self.__record_time
        for key, value in data.items():
            if key == "ts_ns":
                continue
            elif key.endswith("rgb"):
                self.__writers["rgb"].append_data(
                    data_dict={
                        "group_name": key,
                        "data": cv2.imencode('.jpg', value)[1].tobytes(),
                        "sen_ts": ts_ns,
                        "get_ts": ns_now(),
                    })
            elif key.endswith("depth"):
                self.__writers["depth"].append_data(
                    data_dict={
                        "group_name": key,
                        "data": value,
                        "sen_ts": ts_ns,
                        "get_ts": ns_now(),
                    })
            elif key.endswith("pos") or key.endswith("vel") or key.endswith(
                    "eff"):
                self.__writers["arm"].append_data(
                    data_dict={
                        "group_name": key,
                        "data": value,
                        "sen_ts": ts_ns,
                        "get_ts": ns_now(),
                    })
            else:
                raise ValueError(f"Unsupported data type: {key}")

    def start_record(self, path: str, name: str):
        if self.__writers_ready.is_set():
            print(f"Record already started")
            return

        record_dir = f"{path}/{name}"
        os.makedirs(record_dir, exist_ok=True)
        self.__writers = {
            "arm": Hdf5Writer(f"{record_dir}/arm.h5", 10_000, batch_size=1024),
            "rgb": Hdf5Writer(f"{record_dir}/rgb.h5", 300, batch_size=4),
            "depth": Hdf5Writer(f"{record_dir}/depth.h5", 300, batch_size=4),
        }
        for writer in self.__writers.values():
            writer.start()
        self.__cur_time = ns_now()
        self.__writers_ready.set()

    def stop_record(self):
        if not self.__writers_ready.is_set():
            print(f"Record not started")
            return

        self.__writers_ready.clear()
        for key, writer in self.__writers.items():
            print(f"Stopping writer for {key}")
            writer.stop()
        self.__writers = None


def hdf5_to_pd(hdf5_path: str, pd_dir: str = None, quiet: bool = True) -> None:
    if pd_dir is None:
        pd_dir = hdf5_path

    has_pkl = os.path.exists(pd_dir) and any(
        f.endswith('.pkl') for f in os.listdir(pd_dir))
    if has_pkl:
        if not quiet:
            print(
                f"Found pd cache files in {pd_dir}. You can use them directly."
            )
        return

    if not quiet:
        print(f"Cache not found: {pd_dir}, generating pd files...")
    os.makedirs(pd_dir, exist_ok=True)

    for h5_file in sorted(os.listdir(hdf5_path)):
        if not h5_file.endswith(".h5"):
            continue
        with h5py.File(os.path.join(hdf5_path, h5_file), "r") as f:
            _hdf5_visit_groups(f, "", pd_dir, quiet)


def _hdf5_visit_groups(
    group,
    prefix: str,
    pd_dir: str,
    quiet: bool = True,
) -> None:
    for name in group.keys():
        item = group[name]
        path = f"{prefix}/{name}" if prefix else name
        if not isinstance(item, h5py.Group):
            continue
        if "data" in item and "get_ts" in item and "sen_ts" in item:
            _hdf5_group_to_pkl(item, path, pd_dir, quiet)
        else:
            _hdf5_visit_groups(item, path, pd_dir, quiet)


def _hdf5_group_to_pkl(
    group,
    key: str,
    pd_dir: str,
    quiet: bool = True,
) -> None:
    sen_ts = group["sen_ts"][:].flatten()
    get_ts = group["get_ts"][:].flatten()
    data_ds = group["data"]
    n = data_ds.shape[0]

    if key.endswith("rgb") or key.endswith("depth"):
        data_series = pd.Series([[data_ds[i]] for i in range(n)])
    else:
        data_series = pd.Series(list(data_ds[:]))

    final_df = pd.DataFrame({
        "sen_ts": sen_ts,
        "get_ts": get_ts,
        "data": data_series,
    })

    file_name = key.replace("/", "@")
    pd.to_pickle(final_df, f"{pd_dir}/{file_name}.pkl")
    if not quiet:
        print(f"  Saved: {file_name}.pkl ({n} samples)")
