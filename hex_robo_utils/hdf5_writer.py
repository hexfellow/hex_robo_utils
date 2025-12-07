#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-10-11
################################################################

import time
import threading
import os
import h5py
import numpy as np
from collections import deque


class HexHdf5Writer:

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

    def get_shape(self, group_name: str, dataset_name: str = "data"):
        dataset = self.__get_dataset_handle(group_name, dataset_name)
        return dataset.shape

    def get_dtype(self, group_name: str, dataset_name: str = "data"):
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

    def summary(self):
        if self.__hdf5_file is None:
            print("HDF5 file is closed. Cannot get summary.")
            return

        print("#" * 50)
        print(f"HDF5 File: {self.__file_path}")
        print("#" * 50)

        for group_name in self.__hdf5_file.keys():
            print("-" * 30)
            print(f"Group: {group_name}")
            print("-" * 30)
            get_delta_s = ((self.__hdf5_file[group_name]['get_ts'][-1] -
                            self.__hdf5_file[group_name]['get_ts'][0]) *
                           1e-9)[0]
            sen_delta_s = ((self.__hdf5_file[group_name]['sen_ts'][-1] -
                            self.__hdf5_file[group_name]['sen_ts'][0]) *
                           1e-9)[0]
            print(f"get_delta: {get_delta_s}s")
            print(f"sen_delta: {sen_delta_s}s")

            dtype = self.get_dtype(group_name)
            shape = self.get_shape(group_name)
            print(f"  Dataset: {group_name}")
            print(f"    Dtype: {dtype}")
            print(f"    Shape: {shape}")
            print(f"    GetHz: {shape[0] / get_delta_s}")
            print(f"    SenHz: {shape[0] / sen_delta_s}")

        print("#" * 50)

    def __writer_loop(self):
        try:
            while not self.__stop_event.is_set():
                # Collect batch of items
                batch_items = []
                try:
                    # Try to collect up to batch_size items
                    for _ in range(self.__batch_size):
                        item = self.__queue.popleft()
                        batch_items.append(item)
                except IndexError:
                    # Queue is empty or not enough items
                    if len(batch_items) == 0:
                        time.sleep(1e-5)
                        continue

                # Write batch if we have items
                if batch_items:
                    self.__write_batch(batch_items)

            # Flush remaining items
            empty_count = 0
            max_empty_checks = 10
            while empty_count < max_empty_checks:
                batch_items = []
                try:
                    # Collect remaining items in batches
                    for _ in range(self.__batch_size):
                        item = self.__queue.popleft()
                        batch_items.append(item)
                    empty_count = 0
                except IndexError:
                    # Queue is empty, but we may have collected some items
                    # Write them before continuing
                    if batch_items:
                        self.__write_batch(batch_items)
                        batch_items = []
                    empty_count += 1
                    if empty_count >= max_empty_checks:
                        break
                    time.sleep(1e-5)
                    continue

                # Write batch if we have items
                if batch_items:
                    self.__write_batch(batch_items)
        except Exception as e:
            self.__writer_exc = e
            raise

    def __write_batch(self, batch_items):
        """Write a batch of items, grouped by group_name for efficiency."""
        # Group items by group_name
        grouped_items = {}
        for item in batch_items:
            group = item[0]
            if group not in grouped_items:
                grouped_items[group] = []
            grouped_items[group].append(item)

        # Write each group's batch
        for group, items in grouped_items.items():
            self.__write_group_batch(group, items)

    def __write_group_batch(self, group_name, items):
        """Write a batch of items for a specific group."""
        dataset_key = f"{group_name}/data"
        get_ts_key = f"{group_name}/get_ts"
        sen_ts_key = f"{group_name}/sen_ts"
        ds = self.__dataset_dict[dataset_key]
        d_get = self.__dataset_dict[get_ts_key]
        d_sen = self.__dataset_dict[sen_ts_key]

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

        for group, data, gts, sts in items:
            data_list.append(data)
            # Ensure gts and sts are scalars or 1-element arrays
            if isinstance(gts, np.ndarray):
                gts_val = gts.item() if gts.size == 1 else gts[0]
            else:
                gts_val = int(gts)
            if isinstance(sts, np.ndarray):
                sts_val = sts.item() if sts.size == 1 else sts[0]
            else:
                sts_val = int(sts)
            gts_list.append(gts_val)
            sts_list.append(sts_val)

        # Stack arrays for batch write
        data_batch = np.stack(data_list, axis=0)
        gts_batch = np.array(gts_list, dtype=np.int64).reshape(-1, 1)
        sts_batch = np.array(sts_list, dtype=np.int64).reshape(-1, 1)

        # Batch write
        ds[n_old:n_new, ...] = data_batch
        d_get[n_old:n_new, :] = gts_batch
        d_sen[n_old:n_new, :] = sts_batch

        self.__writer_cnt += batch_size
        cur_print_num = self.__writer_cnt // self.__print_interval
        if cur_print_num > self.__print_num:
            self.__print_num = cur_print_num
            print("#" * 50)
            for group_name in self.__hdf5_file.keys():
                print(f"{group_name} len:{self.get_shape(group_name)[0]}")

    def create_dataset(
        self,
        group_name: str,
        shape: tuple,
        dtype: np.dtype,
        chunk_num: int,
        max_num: int | None = None,
        compression=None,
    ):
        if group_name not in self.__group_dict:
            self.__group_dict[group_name] = self.__hdf5_file.create_group(
                group_name)

        dataset = self.__group_dict[group_name].create_dataset(
            "data",
            shape=(0, *shape),
            maxshape=(max_num, *shape),
            dtype=dtype,
            chunks=(chunk_num, *shape),
            compression=compression,
        )
        get_ts_set = self.__group_dict[group_name].create_dataset(
            "get_ts",
            shape=(0, 1),
            maxshape=(max_num, 1),
            dtype=np.int64,
            chunks=(chunk_num, 1),
        )
        sen_ts_set = self.__group_dict[group_name].create_dataset(
            "sen_ts",
            shape=(0, 1),
            maxshape=(max_num, 1),
            dtype=np.int64,
            chunks=(chunk_num, 1),
        )

        # Store dataset reference for easy access
        self.__dataset_dict[f"{group_name}/data"] = dataset
        self.__dataset_dict[f"{group_name}/get_ts"] = get_ts_set
        self.__dataset_dict[f"{group_name}/sen_ts"] = sen_ts_set

    def append_data(
        self,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray | int,
        sen_ts: np.ndarray | int,
    ):
        if isinstance(get_ts, int):
            get_ts = np.array([get_ts])
        if isinstance(sen_ts, int):
            sen_ts = np.array([sen_ts])
        item = (
            group_name,
            data,
            get_ts,
            sen_ts,
        )
        self.__queue.append(item)

    def append_batch_data(
        self,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray,
        sen_ts: np.ndarray,
    ):
        """Append batch data more efficiently by adding all items to queue at once."""
        batch_size = data.shape[0]
        # Ensure get_ts and sen_ts are properly shaped
        if get_ts.ndim == 0:
            get_ts = np.array([get_ts] * batch_size)
        elif get_ts.shape[0] != batch_size:
            raise ValueError(
                f"get_ts shape mismatch: expected {batch_size}, got {get_ts.shape[0]}"
            )

        if sen_ts.ndim == 0:
            sen_ts = np.array([sen_ts] * batch_size)
        elif sen_ts.shape[0] != batch_size:
            raise ValueError(
                f"sen_ts shape mismatch: expected {batch_size}, got {sen_ts.shape[0]}"
            )

        # Add all items to queue efficiently
        for i in range(batch_size):
            item = (
                group_name,
                data[i],
                get_ts[i] if isinstance(get_ts[i], np.ndarray) else np.array(
                    [get_ts[i]]),
                sen_ts[i] if isinstance(sen_ts[i], np.ndarray) else np.array(
                    [sen_ts[i]]),
            )
            self.__queue.append(item)


class HexHdf5MultiWriter:

    def __init__(self, base_dir: str):
        os.makedirs(base_dir, exist_ok=True)
        arm_path = f"{base_dir}/arm.h5"
        rgb_path = f"{base_dir}/rgb.h5"
        depth_path = f"{base_dir}/depth.h5"
        self.__writers: dict[str, HexHdf5Writer] = {
            "arm": HexHdf5Writer(arm_path, 10_000, batch_size=1024),
            "rgb": HexHdf5Writer(rgb_path, 300, batch_size=4),
            "depth": HexHdf5Writer(depth_path, 300, batch_size=4),
        }

    def start(self):
        for key, writer in self.__writers.items():
            print(f"Starting writer for {key}")
            writer.start()

    def stop(self):
        for key, writer in self.__writers.items():
            print(f"Stopping writer for {key}")
            writer.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # --------------- proxies ---------------
    def get_writer(self, msg_type: str) -> HexHdf5Writer:
        return self.__writers[msg_type]

    def create_dataset(
        self,
        msg_type: str,
        group_name: str,
        shape: tuple,
        dtype: np.dtype,
        chunk_num: int,
        max_num: int | None = None,
    ):
        self.__writers[msg_type].create_dataset(
            group_name=group_name,
            shape=shape,
            dtype=dtype,
            chunk_num=chunk_num,
            max_num=max_num,
            compression=None,
        )

    def append_data(
        self,
        msg_type: str,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray | int,
        sen_ts: np.ndarray | int,
    ):
        self.__writers[msg_type].append_data(
            group_name=group_name,
            data=data,
            get_ts=get_ts,
            sen_ts=sen_ts,
        )

    def append_batch_data(
        self,
        msg_type: str,
        group_name: str,
        data: np.ndarray,
        get_ts: np.ndarray,
        sen_ts: np.ndarray,
    ):
        self.__writers[msg_type].append_batch_data(
            group_name=group_name,
            data=data,
            get_ts=get_ts,
            sen_ts=sen_ts,
        )

    def now_ns(self):
        return np.array([time.time_ns()])

    def hex_ts_to_ns(self, ts: dict):
        try:
            return np.array([ts["s"] * 1e9 + ts["ns"]])
        except Exception as e:
            print(f"hex_ts_to_ns failed: {e}")
            return np.array([np.inf])
