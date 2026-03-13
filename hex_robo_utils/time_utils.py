#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-16
################################################################

import os, time
import ctypes, ctypes.util
import numpy as np


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class HexTimeManager(metaclass=SingletonMeta):

    class timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    def __init__(self):
        self.__use_ptp = False
        ptp_path = os.getenv("HEX_PTP_CLOCK", None)
        if ptp_path is not None:
            self.__fd = os.open(ptp_path, os.O_RDONLY | os.O_CLOEXEC)
            self.__clock_id = ((~self.__fd) << 3) | 3
            self.__libc = ctypes.CDLL(
                ctypes.util.find_library("c"),
                use_errno=True,
            )
            self.__use_ptp = True
            print(f"Using PTP clock from {ptp_path}")
        else:
            print("Using system clock")

    def __del__(self):
        if self.__use_ptp:
            os.close(self.__fd)

    def get_now_ns(self) -> int:
        if self.__use_ptp:
            ts = self.timespec()
            if self.__libc.clock_gettime(self.__clock_id,
                                         ctypes.byref(ts)) != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err))
            return int(ts.tv_sec * 1_000_000_000 + ts.tv_nsec)
        else:
            return time.perf_counter_ns()


_HEX_TIME_MANAGER = HexTimeManager()


def hex_ts_to_ns(ts: dict) -> int:
    try:
        return int(ts['s'] * 1_000_000_000 + ts['ns'])
    except Exception as e:
        print(f"hex_ts_to_ns failed: {e}")
        return np.inf


def ns_to_hex_ts(ns: int) -> dict:
    return {
        "s": int(ns // 1_000_000_000),
        "ns": int(ns % 1_000_000_000),
    }


def ns_now() -> int:
    return _HEX_TIME_MANAGER.get_now_ns()


def hex_ts_now() -> dict:
    return ns_to_hex_ts(ns_now())


def hex_ts_delta_ms(curr_ts, hdr_ts) -> float:
    try:
        return (curr_ts['s'] - hdr_ts['s']) * 1_000 + (
            curr_ts['ns'] - hdr_ts['ns']) / 1_000_000

    except Exception as e:
        print(f"hex_ts_delta_ms failed: {e}")
        return np.inf


class HexRate:

    def __init__(self, hz: float, spin_threshold_ns: int = 10_000):
        if hz <= 0:
            raise ValueError("hz must be greater than 0")
        if spin_threshold_ns < 0:
            raise ValueError("spin_threshold_ns must be non-negative")
        self.__period_ns = int(1_000_000_000 / hz)
        self.__next_ns = ns_now() + self.__period_ns
        self.__spin_threshold_ns = spin_threshold_ns

    def reset(self):
        self.__next_ns = ns_now() + self.__period_ns

    def sleep(self):
        target_ns = self.__next_ns
        now_ns = ns_now()
        remain_ns = target_ns - now_ns
        if remain_ns <= 0:
            needed_period = (now_ns - target_ns) // self.__period_ns + 1
            self.__next_ns += needed_period * self.__period_ns
            return

        spin_threshold = min(self.__spin_threshold_ns, self.__period_ns)
        coarse_sleep_ns = remain_ns - spin_threshold
        if coarse_sleep_ns > 0:
            time.sleep(coarse_sleep_ns / 1_000_000_000.0)

        while True:
            now_ns = ns_now()
            if now_ns >= target_ns:
                break
            if target_ns - now_ns > 50_000:
                time.sleep(0)

        self.__next_ns += self.__period_ns
