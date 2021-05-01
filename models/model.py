from __future__ import annotations
import ctypes
import typing as tp

class Model(tp.Protocol):
    def setup(self) -> None: ...
    def render(self) -> None: ...

