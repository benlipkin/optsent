import numpy as np
import numpy.typing as npt
import pandas as pd

from optsent.abstract import Object


class Graph(Object):
    def __init__(self, size: int) -> None:
        super().__init__()
        if not isinstance(size, int):
            raise TypeError("size must be type `int.`")
        if not size > 1:
            raise ValueError("size must be >1.")
        self._dim = size
        self._matrix = np.zeros((self.dim, self.dim), dtype=np.float64)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        return self._matrix

    def write_transition_weight(self, i: int, j: int, value: float) -> None:
        if not isinstance(i, int) and isinstance(j, int):
            raise TypeError("i and j must be `int` indices.")
        if i >= self.dim or j >= self.dim:
            raise ValueError(f"i and j must be in range [0, {self.dim})")
        if not issubclass(value.__class__, (float, int)):
            raise TypeError("value must be subtype of `float` or `int`")
        self._matrix[i, j] = np.float64(value)


class SentenceCollection(Object):
    def __init__(self, inputs: pd.Series) -> None:
        super().__init__()
        if not isinstance(inputs, pd.Series):
            raise TypeError("inputs must be type `pd.Series`")
        if not inputs.size > 1:
            raise ValueError("inputs must have at least 2 elements")
        if not all(isinstance(i, str) for i in inputs):
            raise TypeError("inputs must only contain elements of type `str`.")
        if not inputs.name == "Sentence":
            inputs.name = "Sentence"
        self._sentences = inputs
        self._graph = Graph(self.size)
        self.info(f"Built collection of {self.size} sentences.")

    @property
    def sentences(self) -> pd.Series:
        return self._sentences

    @property
    def size(self) -> int:
        return self.sentences.size

    @property
    def graph(self) -> Graph:
        return self._graph
