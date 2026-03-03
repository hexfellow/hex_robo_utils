#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import json, os
import cv2
import numpy as np
import pandas as pd
from collections import OrderedDict


class HexPandasRecordReader:

    def __init__(self, pd_dir: str):
        if (not os.path.exists(pd_dir)) or (len(os.listdir(pd_dir)) == 0):
            raise FileNotFoundError(f"Cache not found: {pd_dir}")

        self.__pd_cache = {}
        self.__load_pd_cache(pd_dir)

    def __del__(self):
        self.close()

    def close(self):
        self.__pd_cache.clear()

    def summary(self):
        for key, df in self.__pd_cache.items():
            print(f"Key: {key}")
            print(f"  Data Number: {len(df['data'])}")

    def get_keys(self) -> list[str]:
        return self.__pd_cache.keys()

    def get_data(self, key: str) -> np.ndarray:
        df = self.__pd_cache[key]
        sen_ts_np = df["sen_ts"].to_numpy()
        get_ts_np = df["get_ts"].to_numpy()
        data_series = df['data']

        data = np.array([])
        if key.endswith("rgb"):
            rgbs = []
            for blob in data_series:
                rgb = cv2.imdecode(blob[0], cv2.IMREAD_COLOR)
                if rgb is not None:
                    rgbs.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            data = np.stack(rgbs)
        elif key.endswith("depth"):
            depths = []
            for blob in data_series:
                depth = blob[0]
                depths.append(depth)
            data = np.stack(depths)
        else:
            data = np.stack(
                [np.asarray(v, dtype=np.float32) for v in data_series])

        return {"sen_ts": sen_ts_np, "get_ts": get_ts_np, "data": data}

    def get_all_data(self, use_sen_ts: bool = False) -> dict[str, np.ndarray]:
        return {key: self.get_data(key, use_sen_ts) for key in self.get_keys()}

    def __load_pd_cache(self, pd_dir: str) -> None:
        self.__pd_cache.clear()
        for key in os.listdir(pd_dir):
            if key.endswith(".pkl"):
                df = pd.read_pickle(f"{pd_dir}/{key}")
                new_key = os.path.splitext(key)[0].replace("@", "/")
                self.__pd_cache[new_key] = df


class HexPandasTrainReader:

    def __init__(self, pd_dir: str, cache_size: int = 4):
        if not os.path.exists(pd_dir):
            raise FileNotFoundError(f"Directory not found: {pd_dir}")

        self.__pd_dir = pd_dir
        self.__manifest: list[dict] = []
        self.__sample_nums = [0]

        manifest_path = f"{pd_dir}/manifest.json"
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {pd_dir}/manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.__manifest = json.load(f)

        self.__manifest = sorted(self.__manifest, key=lambda x: x["idx"])
        total = 0
        for file_info in self.__manifest:
            total += file_info["samples"]
            self.__sample_nums.append(total)
        self.__sample_nums = np.array(self.__sample_nums)

        self.__cache_size = cache_size
        self.__lru_cache = OrderedDict()

    def __len__(self) -> int:
        return self.__sample_nums[-1] if self.__sample_nums else 0

    def summary(self):
        total = len(self)
        print(f"HexPandasTrainReader: {self.__pd_dir}")
        print(f"  Total files: {len(self.__manifest)}")
        print(f"  Total samples: {total}")
        for file_info in self.__manifest:
            print(f"    {file_info['file']}: {file_info['samples']} samples")

    def __load_file(self, file_idx: int) -> pd.DataFrame:
        if file_idx in self.__lru_cache:
            self.__lru_cache.move_to_end(file_idx)
            return self.__lru_cache[file_idx]

        file_name = self.__manifest[file_idx]["file"]
        df = pd.read_pickle(f"{self.__pd_dir}/{file_name}")

        if len(self.__lru_cache) >= self.__cache_size:
            self.__lru_cache.popitem(last=False)

        self.__lru_cache[file_idx] = df
        return df

    def get_data(self, idx: int) -> dict[str, np.ndarray]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        file_idx = np.searchsorted(self.__sample_nums, idx + 1) - 1
        df = self.__load_file(file_idx)

        row = df.iloc[idx - self.__sample_nums[file_idx - 1]]
        return {k: np.asarray(row[k]) for k in row.index}


class HexPandasTrainWriter:

    def __init__(
        self,
        pd_dir: str,
        prefix: str = "",
        switch_gb: float = 1.0,
        start_idx: int = 0,
    ):
        self.__pd_dir = pd_dir
        if not os.path.exists(pd_dir):
            os.makedirs(pd_dir)

        self.__prefix = f"{prefix}_" if prefix else ""

        self.__cur_idx = start_idx
        self.__cur_data: list[dict[str, np.ndarray]] = []

        self.__switch_bytes = int(max(switch_gb * 1024**3, 1024**2))

        self.__manifest_path = f"{self.__pd_dir}/manifest.json"
        self.__manifest: list[dict] = []
        self.__load_existing_manifest()

    def __del__(self):
        self.close()

    def close(self):
        self.__flush_current_file()

    def summary(self):
        total = sum(item["samples"] for item in self.__manifest)
        print(f"HexPandasWriter: {self.__pd_dir}")
        print(f"  Total files: {len(self.__manifest)}")
        print(f"  Total samples: {total}")
        for file_info in self.__manifest:
            print(f"    {file_info['file']}: {file_info['samples']} samples")

    def write_data(self, data_dict: dict[str, np.ndarray]):
        self.__cur_data.append(data_dict)
        if self.__current_size_bytes() >= self.__switch_bytes:
            self.__flush_current_file()

    def __load_existing_manifest(self):
        if os.path.exists(self.__manifest_path):
            with open(self.__manifest_path, "r", encoding="utf-8") as f:
                self.__manifest = json.load(f)
            if self.__manifest:
                self.__cur_idx = self.__manifest[-1]["idx"] + 1

    def __flush_current_file(self):
        if not self.__cur_data:
            return

        df = pd.DataFrame(self.__cur_data)
        file_name = f"{self.__prefix}{self.__cur_idx:04d}.parquet"
        df.to_parquet(f"{self.__pd_dir}/{file_name}", index=False)
        self.__manifest.append({
            "file": file_name,
            "idx": self.__cur_idx,
            "samples": len(df),
            "size": self.__current_size_bytes(),
        })
        with open(self.__manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.__manifest, f, indent=2)

        self.__cur_idx += 1
        self.__cur_data = []

    def __current_size_bytes(self) -> int:
        if not self.__cur_data:
            return 0
        return sum(arr.nbytes for data in self.__cur_data
                   for arr in data.values() if isinstance(arr, np.ndarray))
