#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-01-15
################################################################

import time
import json
import socket
import numpy as np
from ipaddress import ip_address


def dict_update(dict_raw: dict, dict_new: dict, add_new: bool = False):
    for key, value in dict_new.items():
        if key in dict_raw:
            if isinstance(dict_raw[key], dict) and isinstance(value, dict):
                dict_update(dict_raw[key], value)
            else:
                dict_raw[key] = value
        elif add_new:
            dict_raw[key] = value


class HexPlotUtilPlotJuggler:

    def __init__(
        self,
        srv_ip: str = "127.0.0.1",
        srv_port: int = 9870,
    ):
        self.__srv_ip = srv_ip
        self.__srv_port = srv_port

        addr = ip_address(self.__srv_ip)
        family = socket.AF_INET6 if addr.version == 6 else socket.AF_INET
        self.__socket = socket.socket(family, socket.SOCK_DGRAM)
        self.__data_dict = {}

    def __del__(self):
        self.__socket.close()

    def clear_data(self):
        self.__data_dict.clear()

    def add_data(self, name: str, data: dict):
        if name not in self.__data_dict:
            self.__data_dict[name] = {}
        dict_update(self.__data_dict[name], data, add_new=True)

    def add_arr(self, name: str, data: np.ndarray, labels: list[str]):
        assert data.shape[1] >= len(
            labels
        ), "data shape must be greater than or equal to labels length"

        arr_dict = {}
        for i, label in enumerate(labels):
            arr_dict[label] = data[:, i].tolist()
        self.add_data(name, arr_dict)

    def update_data(self, name: str, data: dict):
        if name not in self.__data_dict:
            self.__data_dict[name] = data.copy()
        else:
            for key, value in data.items():
                if key not in self.__data_dict[name]:
                    self.__data_dict[name][key] = value
                else:
                    self.__data_dict[name][key].append(value)

    def send_data(self, ts_ns: int = None, clear: bool = False):
        if ts_ns is None:
            ts_ns = time.time_ns()
        self.__data_dict["timestamp"] = ts_ns * 1e-9
        self.__socket.sendto(
            json.dumps(self.__data_dict).encode(),
            (self.__srv_ip, self.__srv_port),
        )
        if clear:
            self.clear_data()
