import abc
import re
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm

from optsent.abstract import Object
from optsent.data import SentenceCollection


class Optimizer(Object):
    def __init__(
        self,
        optimizer: str,
        constraint: str,
        cutoff: float,
        seqlen: int,
        maximize: bool,
    ):
        super().__init__()
        if not all(
            isinstance(arg, type)
            for arg, type in zip(
                (optimizer, constraint, cutoff, seqlen, maximize),
                (str, str, (int,float), int, bool),
            )
        ):
            raise TypeError("arguments must adhere to interface.")
        self._id = optimizer
        self._indices: typing.List[np.int64] = []
        self._values: typing.List[float] = []
        satisfied: typing.Callable = self._build_constraint(constraint)
        try:
            self._optimizer = self.supported_optimizers()[self._id](
                maximize, seqlen, satisfied, cutoff
            )
        except KeyError as invalid_optimizer:
            raise ValueError(
                f"optimizer_id must be in supported: {self.supported_optimizers().keys()}"
            ) from invalid_optimizer
        self.info(f"Defined {self._optimizer._name[1:]} optimizer.")

    @property
    def indices(self) -> typing.List[np.int64]:
        return self._indices

    @property
    def values(self) -> typing.List[float]:
        return self._values

    @classmethod
    def supported_optimizers(cls) -> typing.Dict[str, typing.Callable]:
        return {"greedy": _Greedy, "sampling": _Sampling}

    @classmethod
    def supported_constraints(cls) -> typing.Set[str]:
        return {"none", "repeats"}

    @staticmethod
    def _build_constraint(constraint) -> typing.Callable:
        if constraint == "none":

            def satisfied(text: pd.Series, vertex: int, target: int) -> bool:
                return True

        elif constraint == "repeats":

            def satisfied(text: pd.Series, vertex: int, target: int) -> bool:
                def tokenize(string: str) -> typing.List[str]:
                    return re.sub(r"[^A-Za-z0-9 ]+", "", string).lower().split()

                return tokenize(text[vertex])[-1] != tokenize(text[target])[0]

        else:
            raise ValueError("Unsupported constraint.")
        return satisfied

    def solve(self, sents: SentenceCollection) -> None:
        if not isinstance(sents, SentenceCollection):
            raise TypeError("Optimizer can only solve `SentenceCollection` objects.")
        self._indices, self._values = self._optimizer(sents)


class _LinearATSP(Object):
    def __init__(
        self, maximize: bool, seqlen: int, satisfied: typing.Callable, cutoff: float
    ):
        super().__init__()
        self._opt = np.max if maximize else np.min
        self._argopt = np.argmax if maximize else np.argmin
        self._sign = -1 if maximize else 1
        self._null = self._sign * np.inf
        self._seqlen = seqlen
        self._satisfied = satisfied
        self._cutoff = cutoff

    def _update_seqlen(self, sents: SentenceCollection) -> None:
        if self._seqlen > sents.size:
            self._seqlen = sents.size
        elif self._seqlen <= 0:
            self._seqlen = sents.size
        else:
            pass

    @abc.abstractmethod
    def _select_optimal_target(
        self, matrix: npt.NDArray[np.float64], vertex: np.int64
    ) -> np.int64:
        raise NotImplementedError()  # pragma: no cover

    def _select_random_target(
        self, matrix: npt.NDArray[np.float64], vertex: np.int64
    ) -> np.int64:
        self.warn(
            "Stuck at non-optimal transition. Sampling new states until constraints valid."
        )
        targets = matrix[vertex, :]
        mask = targets != self._null
        options = np.where(mask)[0]
        return np.random.choice(options)

    def _get_next_vertex(
        self,
        matrix: npt.NDArray[np.float64],
        vertex: np.int64,
        sents: SentenceCollection,
    ) -> np.int64:
        target = self._select_optimal_target(matrix, vertex)
        attempts = 5
        while not self._satisfied(sents.sentences, vertex, target):
            if not attempts:
                self.warn(
                    f"No valid transitions found. Relaxing constraints from state {vertex}."
                )
                break
            target = self._select_random_target(matrix, vertex)
            attempts -= 1
        return target

    def __call__(
        self, sents: SentenceCollection
    ) -> typing.Tuple[typing.List[np.int64], typing.List[float]]:
        if np.all(sents.graph.matrix == 0):
            raise RuntimeError("SentenceCollection graph is empty.")
        self._update_seqlen(sents)
        matrix = sents.graph.matrix.copy()
        matrix[np.isnan(matrix)] = self._null
        indices, values = [], []
        vertex = self._argopt(self._opt(matrix, axis=0))
        matrix[:, vertex] = self._null
        indices.append(vertex)
        values.append(np.nan)
        for _ in tqdm.trange(self._seqlen - 1):  # type:ignore
            target = self._get_next_vertex(matrix, vertex, sents)
            value = matrix[vertex, target]
            indices.append(target)
            values.append(value)
            matrix[:, target] = self._null
            vertex = target
        return indices, values


class _Greedy(_LinearATSP):
    def _select_optimal_target(
        self, matrix: npt.NDArray[np.float64], vertex: np.int64
    ) -> np.int64:
        return self._argopt(matrix[vertex, :])


class _Sampling(_LinearATSP):
    def _select_optimal_target(
        self, matrix: npt.NDArray[np.float64], vertex: np.int64
    ) -> np.int64:
        targets = matrix[vertex, :]
        mask = self._sign * targets < self._cutoff
        if not mask.sum() > 0:
            return self._select_random_target(matrix, vertex)
        options = np.where(mask)[0]
        return np.random.choice(options)
