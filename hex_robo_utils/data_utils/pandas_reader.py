#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import os
import cv2
import numpy as np
import pandas as pd


class HexPandasReader:

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
