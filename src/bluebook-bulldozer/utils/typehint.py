from typing import Sequence, TypeVar, ParamSpec, Union, TypeAlias

R = TypeVar("R")
P = ParamSpec("P")
Number: TypeAlias = Union[int, float]
NumberSequence: TypeAlias = Sequence[Number]
