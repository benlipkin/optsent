import itertools
import pathlib
import typing

import joblib
import numpy as np
import pandas as pd
import tqdm

from optsent.abstract import Object, IModel, IObjective, IOptimizer
from optsent.args import ArgTool


class OptSent(Object):
    def __init__(
        self,
        inputs: str | pathlib.Path | typing.Collection[str],
        outdir: str | pathlib.Path = pathlib.Path(__file__).parents[1] / "outputs",
        model: str | IModel = "gpt2",
        objective: str | IObjective = "normlogp",
        optimizer: str | IOptimizer = "greedy",
        constraint: str = "none",
        cutoff: float = 0.0,
        seqlen: int = -1,
        maximize: bool = False,
        ncores: int = 1,
        export: bool = True,
    ) -> None:
        # pylint: disable=unused-argument
        super().__init__()
        kwargs = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        argtool = ArgTool()
        argtool.log_args(kwargs)
        self._unique_id = argtool.get_unique_id(kwargs)
        for arg, value in kwargs.items():
            argprep = getattr(argtool, f"prep_{arg}")
            setattr(self, f"_{arg}", argprep(value))
        self._optimizer = argtool.build_optimizer(kwargs)

    @property
    def unique_id(self):
        return self._unique_id

    def _build_graph(self) -> None:
        # refactor: loop O(n), compare each sent with batch of n sents, fill in whole column at once
        self.info("Building transition graph.")
        dim = self._inputs.size
        indices = tqdm.tqdm(
            itertools.product(range(dim), range(dim)), total=dim**2
        )  # type:ignore

        def write_weight(self, i, j):
            if i == j:
                value = np.nan
            else:
                sent1, sent2 = self._inputs.sentences[i], self._inputs.sentences[j]
                value = self._objective.evaluate(sent1, sent2, self._model)
            self._inputs.graph.write_transition_weight(i, j, value)

        with joblib.parallel_backend("threading", n_jobs=self._ncores):
            joblib.Parallel()(
                joblib.delayed(write_weight)(self, i, j) for i, j in indices
            )

    def _solve_optim(self) -> None:
        self.info("Solving sequence optimization.")
        self._optimizer.solve(self._inputs)

    def _make_output_table(self) -> pd.DataFrame:
        table = pd.DataFrame(self._inputs.sentences[self._optimizer.indices])
        table["TransitionObjective"] = self._optimizer.values
        return table

    def _save_input(self) -> None:
        if self._export:
            self.info("Caching input strings.")
            fname = self._outdir / self.unique_id / "INPUT.csv"
            table = self._inputs.sentences
            table.to_csv(fname, index_label="SentenceID")

    def _save_graph(self) -> None:
        if self._export:
            self.info("Caching transition graph.")
            fname = self._outdir / self.unique_id / "GRAPH.csv"
            table = pd.DataFrame(
                data=self._inputs.graph.matrix,
                index=self._inputs.sentences.index,
                columns=self._inputs.sentences.index,
            )
            table.to_csv(fname, index_label="SentenceID")

    def _save_optim(self) -> None:
        if self._export:
            self.info("Exporting optimal sequence.")
            fname = self._outdir / self.unique_id / "OPTIM.csv"
            table = self._make_output_table()
            table.to_csv(fname, index_label="SentenceID")

    def run(self) -> pd.DataFrame:
        if self._export:
            (self._outdir / self.unique_id).mkdir(parents=True, exist_ok=True)
        self._save_input()
        self._build_graph()
        self._save_graph()
        self._solve_optim()
        self._save_optim()
        return self._make_output_table()
