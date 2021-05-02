from __future__ import annotations
import typing as tp



class Model(tp.Protocol):
    def setup(self, window: tp.Any) -> None: ...
    def render(self) -> None: ...

