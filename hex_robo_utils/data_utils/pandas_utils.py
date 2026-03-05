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

from ..common_utils import hex_rmtree


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

    def __init__(self, data_dir: str, cache_size: int = 4):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        self.__data_dir = data_dir
        self.__manifest: list[dict] = []
        self.__sample_nums = [0]

        manifest_path = f"{data_dir}/manifest.json"
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {data_dir}/manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        if isinstance(loaded, dict):
            self.__manifest = loaded.get("files", [])
        else:
            self.__manifest = loaded

        self.__manifest = sorted(self.__manifest, key=lambda x: x["idx"])
        total = 0
        for file_info in self.__manifest:
            total += file_info["samples"]
            self.__sample_nums.append(total)
        self.__sample_nums = np.array(self.__sample_nums)

        self.__cache_size = cache_size
        self.__lru_cache = OrderedDict()

        self.__keys = None
        _ = self.__load_file(0)

    def __len__(self) -> int:
        return self.__sample_nums[-1]

    def summary(self):
        total = len(self)
        print(f"HexPandasTrainReader: {self.__data_dir}")
        print(f"  Total files: {len(self.__manifest)}")
        print(f"  Total samples: {total}")
        for file_info in self.__manifest:
            ep = file_info.get("episode", "?")
            ei = file_info.get("end_idx", "?")
            print(f"    {file_info['file']}: {file_info['samples']} samples, episode={ep}, end_idx={ei}")

    def get_keys(self) -> list[str]:
        return self.__keys

    def get_manifest(self) -> list[dict]:
        """Return file list, each with file, idx, samples, episode, end_idx."""
        return self.__manifest.copy()

    def get_data(self, idx: int) -> dict[str, np.ndarray]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        file_idx = np.searchsorted(self.__sample_nums, idx + 1) - 1
        data_dict = self.__load_file(file_idx)

        return {k: data_dict[k][idx] for k in self.__keys}

    def __load_file(self, file_idx: int) -> dict[str, np.ndarray]:
        if file_idx in self.__lru_cache:
            self.__lru_cache.move_to_end(file_idx)
            return self.__lru_cache[file_idx]

        file_name = self.__manifest[file_idx]["file"]
        df = pd.read_pickle(f"{self.__data_dir}/{file_name}")
        self.__keys = df.columns.tolist()
        data_dict = {key: df[key][0] for key in self.__keys}

        if len(self.__lru_cache) >= self.__cache_size:
            self.__lru_cache.popitem(last=False)

        self.__lru_cache[file_idx] = data_dict
        return data_dict


class HexPandasTrainWriter:

    def __init__(
        self,
        data_dir: str,
        prefix: str = "",
        switch_gb: float = 1.0,
        start_idx: int = 0,
        remove_old: bool = False,
    ):
        self.__data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        elif remove_old:
            hex_rmtree(data_dir)
            os.makedirs(data_dir, exist_ok=True)

        self.__prefix = f"{prefix}_" if prefix else ""

        self.__cur_idx = start_idx
        self.__cur_data: dict[str, np.ndarray] = {}

        self.__switch_bytes = int(max(switch_gb * 1024**3, 1024**2))

        self.__manifest_path = f"{self.__data_dir}/manifest.json"
        self.__manifest: list[dict] = []
        self.__cur_episode = 0

        self.__load_existing_manifest()

    def __del__(self):
        self.close()

    def close(self):
        self.__flush_current_file()

    def summary(self):
        total = sum(item["samples"] for item in self.__manifest)
        print(f"HexPandasWriter: {self.__data_dir}")
        print(f"  Total files: {len(self.__manifest)}")
        print(f"  Total samples: {total}")
        for file_info in self.__manifest:
            ep = file_info.get("episode", "?")
            ei = file_info.get("end_idx", "?")
            print(f"    {file_info['file']}: {file_info['samples']} samples, episode={ep}, end_idx={ei}")

    def write_episode(self, data_dict: dict[str, np.ndarray]) -> None:
        """Write one episode. Each call writes one episode; each file gets its own episode and end_idx."""
        if not data_dict:
            return

        ref_key = next(iter(data_dict))
        episode_samples = data_dict[ref_key].shape[0]
        if not all(
            data_dict[k].shape[0] == episode_samples for k in data_dict
        ):
            raise ValueError("The shape[0] is not the same for all keys")

        for key, value in data_dict.items():
            if key not in self.__cur_data:
                self.__cur_data[key] = [value.copy()]
            else:
                self.__cur_data[key][0] = np.concatenate(
                    [self.__cur_data[key][0], value.copy()], axis=0
                )

        while self.__cur_data:
            self.__flush_current_file()
        self.__cur_episode += 1

    def __load_existing_manifest(self):
        if os.path.exists(self.__manifest_path):
            with open(self.__manifest_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.__manifest = loaded.get("files", [])
            else:
                self.__manifest = loaded
            if self.__manifest:
                self.__cur_idx = self.__manifest[-1]["idx"] + 1
                last = self.__manifest[-1]
                self.__cur_episode = last.get("episode", 0) + 1

    def __flush_current_file(self):
        if not self.__cur_data:
            return

        df = pd.DataFrame(self.__cur_data)
        file_name = f"{self.__prefix}{self.__cur_idx:04d}.pkl"
        pd.to_pickle(df, f"{self.__data_dir}/{file_name}")
        samples = self.__cur_data["idx"][0].shape[0]
        total_before = sum(item["samples"] for item in self.__manifest)
        end_idx = total_before + samples

        self.__manifest.append({
            "file": file_name,
            "idx": self.__cur_idx,
            "samples": samples,
            "size": self.__current_size_bytes(),
            "episode": self.__cur_episode,
            "end_idx": end_idx,
        })

        self.__cur_idx += 1
        self.__cur_data = {}
        self.__save_manifest()

    def __save_manifest(self) -> None:
        manifest_data = {"files": self.__manifest}
        with open(self.__manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def __current_size_bytes(self) -> int:
        if not self.__cur_data:
            return 0
        return sum(
            v[0].nbytes
            for v in self.__cur_data.values()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray)
        )
