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
        self.__manifest: dict[str, dict] = {}
        self.__sample_nums = [0]

        manifest_path = f"{data_dir}/manifest.json"
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        self.__dataset_meta: dict = {}
        self.__schema: dict = {}
        if isinstance(loaded, dict):
            self.__dataset_meta = loaded.get("dataset_meta", {})
            self.__schema = loaded.get("schema", {})
            files_data = loaded.get("files", loaded)
        else:
            files_data = loaded

        if isinstance(files_data, dict):
            self.__manifest = {
                k: v
                for k, v in files_data.items()
                if isinstance(v, dict) and "samples" in v
            }
        else:
            self.__manifest = {}
            for f in files_data:
                if isinstance(f, dict):
                    fn = f.get("file")
                    if fn:
                        self.__manifest[fn] = {
                            k: v
                            for k, v in f.items() if k != "file"
                        }
                else:
                    self.__manifest[str(f)] = {}

        self.__manifest_list = sorted(self.__manifest.items(),
                                      key=lambda x: x[1].get("idx", 0))
        total = 0
        for _file_name, file_info in self.__manifest_list:
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
        for file_name, file_info in self.__manifest_list:
            ep = file_info.get("episodes", "?")
            ei = file_info.get("episode_boundaries", "?")
            print(
                f"    {file_name}: {file_info['samples']} samples, episodes={ep}, episode_boundaries={ei}"
            )
            print(f"    Keys: {file_info.get('keys', '?')}")

    def get_keys(self) -> list[str]:
        return self.__keys

    def get_manifest(self) -> dict[str, dict]:
        """Return manifest dict: file_name -> {idx, samples, episode, end_idx, size}."""
        return self.__manifest.copy()

    def get_dataset_meta(self) -> dict:
        """Return dataset_meta: name, version, created_by, episodes, total_samples."""
        return self.__dataset_meta.copy()

    def get_schema(self) -> dict:
        """Return schema: key -> {shape, dtype}."""
        return self.__schema.copy()

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

        file_name = self.__manifest_list[file_idx][0]
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
        dataset_info: dict = {
            "name": "hex_robot_dataset",
            "version": "1.0",
            "created_by": "HexFellow",
        },
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
        self.__manifest: dict[str, dict] = {}
        self.__episode_boundaries: list[int] = []

        self.__dataset_meta = {
            "name": dataset_info.get("name", "hex_robot_dataset"),
            "version": dataset_info.get("version", "1.0"),
            "created_by": dataset_info.get("created_by", "HexFellow"),
            "episodes": 0,
            "total_samples": 0,
        }
        self.__schema = {}

        self.__load_existing_manifest()

    def __del__(self):
        self.close()

    def close(self):
        self.__flush_current_file()

    def summary(self):
        total = sum(info["samples"] for info in self.__manifest.values())
        print(f"HexPandasWriter: {self.__data_dir}")
        print(f"  Total files: {len(self.__manifest)}")
        print(f"  Total samples: {total}")
        for file_name, file_info in sorted(self.__manifest.items(),
                                           key=lambda x: x[1]["idx"]):
            ep = file_info.get("episodes", "?")
            ei = file_info.get("episode_boundaries", "?")
            print(
                f"    {file_name}: {file_info['samples']} samples, episodes={ep}, episode_boundaries={ei}"
            )
            print(f"    Keys: {file_info.get('keys', '?')}")

    def get_dataset_meta(self) -> dict:
        """Return dataset_meta: name, version, created_by, episodes, total_samples."""
        return self.__dataset_meta.copy()

    def get_schema(self) -> dict:
        """Return schema: key -> {shape, dtype}."""
        return self.__schema.copy()

    def write_episode(self, data_dict: dict[str, np.ndarray]) -> None:
        """Write one episode. Each call adds one episode; flush when over threshold. Each file records episode count and end_idx list."""
        if not data_dict:
            return

        ref_key = next(iter(data_dict))
        episode_samples = data_dict[ref_key].shape[0]
        if not all(data_dict[k].shape[0] == episode_samples
                   for k in data_dict):
            raise ValueError("The shape[0] is not the same for all keys")

        last_end = self.__episode_boundaries[
            -1] if self.__episode_boundaries else 0
        self.__episode_boundaries.append(last_end + episode_samples)

        for key, value in data_dict.items():
            if key not in self.__cur_data:
                self.__cur_data[key] = [value.copy()]
            else:
                self.__cur_data[key][0] = np.concatenate(
                    [self.__cur_data[key][0],
                     value.copy()], axis=0)

        while self.__cur_data and self.__current_size_bytes(
        ) >= self.__switch_bytes:
            self.__flush_current_file()

    def __load_existing_manifest(self):
        if os.path.exists(self.__manifest_path):
            with open(self.__manifest_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.__dataset_meta.update(loaded.get("dataset_meta", {}))
                if loaded.get("schema"):
                    self.__schema.update(loaded["schema"])
                files_data = loaded.get("files", loaded)
            else:
                files_data = loaded
            if isinstance(files_data, dict):
                self.__manifest = {
                    k: v
                    for k, v in files_data.items()
                    if isinstance(v, dict) and "samples" in v
                }
            else:
                self.__manifest = {}
                for f in files_data:
                    if isinstance(f, dict) and f.get("file"):
                        fn = f["file"]
                        self.__manifest[fn] = {
                            k: v
                            for k, v in f.items() if k != "file"
                        }
            if self.__manifest:
                self.__cur_idx = max(info["idx"]
                                     for info in self.__manifest.values()) + 1

    def __flush_current_file(self):
        if not self.__cur_data:
            return

        # Infer schema from first write if empty (per-sample shape)
        if not self.__schema:
            for key, val in self.__cur_data.items():
                if isinstance(val, list) and val and isinstance(
                        val[0], np.ndarray):
                    arr = val[0]
                    sample_shape = (list(arr.shape[1:])
                                    if arr.ndim > 1 else [1])
                    self.__schema[key] = {
                        "shape": sample_shape,
                        "dtype": str(arr.dtype),
                    }

        df = pd.DataFrame(self.__cur_data)
        file_name = f"{self.__prefix}{self.__cur_idx:04d}.pkl"
        pd.to_pickle(df, f"{self.__data_dir}/{file_name}")
        samples = next(iter(self.__cur_data.values()))[0].shape[0]

        episodes = len(self.__episode_boundaries)
        boundaries = self.__episode_boundaries.copy()
        boundaries.insert(0, 0)
        episode_ranges = [(boundaries[i], boundaries[i + 1])
                          for i in range(episodes)]

        self.__manifest[file_name] = {
            "idx": self.__cur_idx,
            "samples": samples,
            "size": self.__current_size_bytes(),
            "episodes": episodes,
            "episode_boundaries": episode_ranges,
            "keys": list(self.__cur_data.keys()),
        }

        self.__cur_idx += 1
        self.__cur_data = {}
        self.__episode_boundaries = []

        # Update dataset_meta
        self.__dataset_meta["episodes"] = self.__dataset_meta.get(
            "episodes", 0) + episodes
        self.__dataset_meta["total_samples"] = self.__dataset_meta.get(
            "total_samples", 0) + samples

        self.__save_manifest()

    def __save_manifest(self) -> None:
        manifest = {
            "dataset_meta": self.__dataset_meta,
            "schema": self.__schema,
            "files": self.__manifest,
        }
        with open(self.__manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def __current_size_bytes(self) -> int:
        if not self.__cur_data:
            return 0
        return sum(v[0].nbytes for v in self.__cur_data.values()
                   if isinstance(v, list) and len(v) > 0
                   and isinstance(v[0], np.ndarray))
