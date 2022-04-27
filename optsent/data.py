import typing

import numpy as np
import numpy.typing as npt
import pandas as pd

from optsent.abstract import Object


class Graph(Object):
    def __init__(self, size: int) -> None:
        super().__init__()
        if not isinstance(size, int):
            raise TypeError("size must be type `int.`")
        if not size > 0:
            raise ValueError("size must be >0.")
        self._dim = size
        self._matrix = np.zeros((size, size), dtype=np.float64)

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        return self._matrix

    def write_transition_weight(self, i: int, j: int, value: float) -> None:
        if not isinstance(i, int) and isinstance(j, int):
            raise TypeError("i and j must be `int` indices.")
        if i >= self._dim or j >= self._dim:
            raise ValueError(f"i and j must be in range [0, {self._dim})")
        if not issubclass(value.__class__, float):
            raise TypeError("value must be subtype of `float`")
        self._matrix[i, j] = np.float64(value)


class SentenceCollection(Object):
    def __init__(self, inputs: typing.Collection[str]) -> None:
        super().__init__()
        if not isinstance(inputs, pd.Series):
            try:
                inputs = pd.Series(inputs, dtype=str)
            except Exception as invalid_container:
                raise TypeError("Invalid inputs container type.") from invalid_container
        self._sentences = inputs
        self._graph = Graph(self.size)
        self.log(f"Built collection of {self.size} sentences.")

    @property
    def sentences(self) -> pd.Series:
        return self._sentences

    @property
    def size(self) -> int:
        return self.sentences.size

    @property
    def graph(self) -> Graph:
        return self._graph
