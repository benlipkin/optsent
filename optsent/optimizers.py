import re
import typing

import numpy as np
import pandas as pd
import tqdm

from optsent.abstract import Object
from optsent.data import SentenceCollection


class Optimizer(Object):
    def __init__(self, solver: str, constraint: str, seqlen: int, maximize: bool):
        super().__init__()
        if not all(
            isinstance(arg, type)
            for arg, type in zip(
                (solver, constraint, seqlen, maximize), (str, str, int, bool)
            )
        ):
            raise TypeError("arguments must adhere to interface.")
        self._id = solver
        self._indices: typing.List[np.int64] = []
        self._values: typing.List[float] = []
        satisfied: typing.Callable = self._build_constraint(constraint)
        try:
            self._solver = self.supported_solvers()[self._id](
                maximize, seqlen, satisfied
            )
        except KeyError as invalid_solver:
            raise ValueError(
                f"solver_id must be in supported: {self.supported_solvers().keys()}"
            ) from invalid_solver
        self.info(f"Defined {self._solver._name[1:]} solver.")

    @property
    def indices(self) -> typing.List[np.int64]:
        return self._indices

    @property
    def values(self) -> typing.List[float]:
        return self._values

    @classmethod
    def supported_solvers(cls) -> typing.Dict[str, typing.Callable]:
        return {"greedy": _GreedyATSP}

    @classmethod
    def supported_constraints(cls) -> typing.Set[str]:
        return {"repeats"}

    @staticmethod
    def _build_constraint(constraint) -> typing.Callable:
        if constraint == "repeats":

            def satisfied(text: pd.Series, vertex: int, target: int) -> bool:
                return (
                    re.sub(r"[^A-Za-z0-9 ]+", "", text[vertex]).lower().split()[-1]
                    != re.sub(r"[^A-Za-z0-9 ]+", "", text[target]).lower().split()[0]
                )

            return satisfied
        raise ValueError("Unsupported constraint.")

    def solve(self, sents: SentenceCollection) -> None:
        if not isinstance(sents, SentenceCollection):
            raise TypeError("Optimizer can only solve `SentenceCollection` objects.")
        self._indices, self._values = self._solver(sents)


class _GreedyATSP(Object):
    def __init__(self, maximize: bool, seqlen: int, satisfied: typing.Callable):
        super().__init__()
        self._opt = np.max if maximize else np.min
        self._argopt = np.argmax if maximize else np.argmin
        self._null = -np.inf if maximize else np.inf
        self._seqlen = seqlen
        self._satisfied = satisfied

    def _update_seqlen(self, sents):
        if self._seqlen > sents.size:
            self._seqlen = sents.size
        elif self._seqlen <= 0:
            self._seqlen = sents.size
        else:
            pass

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
            target = self._argopt(matrix[vertex, :])
            while not self._satisfied(sents.sentences, vertex, target):
                matrix[:, target] = self._null
                target = self._argopt(matrix[vertex, :])
            value = matrix[vertex, target]
            if value != self._null:
                indices.append(target)
                values.append(value)
                matrix[:, target] = self._null
                vertex = target
            else:
                self.warn("Sample skipped due to constraint. Set will have n<seqlen.")
        return indices, values
