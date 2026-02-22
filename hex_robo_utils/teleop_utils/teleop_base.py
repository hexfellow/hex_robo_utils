#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import copy, threading
from abc import ABC, abstractmethod
from typing import Any


class HexTeleopBase(ABC):

    def __init__(self):
        self._work_thread = threading.Thread(
            target=self._teleop_listener,
            daemon=True,
        )
        self._work_event = threading.Event()
        self._value: dict[str, Any] | None = None
        self._value_event = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        self._work_event.set()
        self._value_event.clear()
        self._work_thread.start()

    def close(self):
        try:
            self._work_event.clear()
            if self._work_thread is not None:
                self._work_thread.join()
            self._value_event.clear()
        except Exception:
            pass

    def is_working(self) -> bool:
        return self._work_event.is_set()

    @abstractmethod
    def _teleop_listener(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_value(self) -> str | None:
        if not self._value_event.is_set():
            return None

        with self._lock:
            return copy.deepcopy(self._value)

    def clear_value(self):
        with self._lock:
            self._value_event.clear()

    def pop_value(self) -> str | None:
        if not self._value_event.is_set():
            return None

        with self._lock:
            value = copy.deepcopy(self._value)
            self._value_event.clear()
            return value
