from __future__ import annotations
import ctypes
import typing as tp

class Model(tp.Protocol):
    def setup(self, resolution: tuple[int, int]) -> None: ...
    def render(self) -> None: ...

