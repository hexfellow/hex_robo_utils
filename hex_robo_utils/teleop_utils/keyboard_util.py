#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-02-22
################################################################

import sys, tty, termios, select
from .teleop_base import HexTeleopBase


class HexTeleopUtilKeyboard(HexTeleopBase):

    def __init__(self):
        super().__init__()

        self._value = {}

        self.__fd = None
        self.__old_attrs = None
        if not sys.stdin.isatty():
            print("stdin is not a tty")
        else:
            self.__fd = sys.stdin.fileno()
            self.__old_attrs = termios.tcgetattr(self.__fd)
            self.__old_attrs[3] |= (termios.ICANON | termios.ECHO)
            try:
                self.__old_attrs[6][termios.VMIN] = 1
                self.__old_attrs[6][termios.VTIME] = 0
            except Exception:
                pass
            tty.setcbreak(self.__fd)
            print("Keyboard teleop utility initialized")

    def close(self):
        super().close()
        if self.__fd is not None and self.__old_attrs is not None:
            termios.tcsetattr(self.__fd, termios.TCSADRAIN, self.__old_attrs)

    def _teleop_listener(self):
        try:
            while self.is_working():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self._value = {
                            "key": ch,
                        }
                        self._value_event.set()
        except Exception:
            return
