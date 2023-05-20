from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class MinMax:
    min: Union[int, float]
    max: Union[int, float]


@dataclass(frozen=True)
class Constraints:
    layers: MinMax
    latency_sec: MinMax
    parameters: MinMax
