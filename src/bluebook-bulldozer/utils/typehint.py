from typing import ParamSpec, Sequence, TypeAlias, TypeVar, Union

R = TypeVar("R")
P = ParamSpec("P")
Number: TypeAlias = Union[int, float]
NumberSequence: TypeAlias = Sequence[Number]
