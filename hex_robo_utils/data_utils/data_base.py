#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

from abc import ABC, abstractmethod
from typing import Any


class HexDataWriterBase(ABC):

    def __init__(self):
        pass

    def __del__(self):
        self.close()

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def append_data(self, data: dict[str, Any]):
        pass

    @abstractmethod
    def start_record(self, path: str, name: str):
        pass

    @abstractmethod
    def stop_record(self):
        pass
