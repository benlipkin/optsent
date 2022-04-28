import hashlib
import pathlib
import typing

import pandas as pd

from optsent.abstract import Object, ModelInterface, ObjectiveInterface
from optsent.data import SentenceCollection
from optsent.models import Model
from optsent.objectives import Objective
from optsent.optimizers import Optimizer


class ArgTool(Object):
    def _log_arg(self, name: str, value: typing.Any) -> None:
        indent = " " * (12 - len(name))
        self.log(f"{name}{indent}{value}")

    def log_args(self, kwargs: typing.Dict[str, typing.Any]) -> None:
        for name, value in kwargs.items():
            self._log_arg(name, value)

    def get_unique_id(self, kwargs: typing.Dict[str, typing.Any]) -> str:
        md5 = lambda x: hashlib.md5(str(x).encode()).hexdigest()
        elements = ["max" if kwargs["maximize"] else "min"]
        inputs = kwargs["inputs"]
        if isinstance(inputs, (str, pathlib.Path)):
            elem = str(inputs).rsplit("/", maxsplit=1)[-1].split(".")[0]
        else:
            elem = f"CUSTOM{md5(inputs)}"
        elements.append(elem)
        for key in ["objective", "solver", "constraint", "model"]:
            value = kwargs[key]
            if isinstance(value, str):
                elem = value
            else:
                elem = f"CUSTOM{md5(value)}"
            elements.append(f"{key}={elem}")
        unique_id = "_".join(elements)
        self._log_arg("unique_id", unique_id)
        return unique_id

    @staticmethod
    def get_optimizer(kwargs: typing.Dict[str, typing.Any]) -> Optimizer:
        return Optimizer(
            solver=kwargs["solver"],
            constraint=kwargs["constraint"],
            seqlen=kwargs["seqlen"],
            maximize=kwargs["maximize"],
        )

    @staticmethod
    def prep_inputs(
        inputs: str | pathlib.Path | typing.Collection[str],
    ) -> SentenceCollection:
        if isinstance(inputs, str):
            inputs = pathlib.Path(inputs).resolve()
        if isinstance(inputs, pathlib.Path):
            if not inputs.is_file():
                raise FileNotFoundError(f"inputs file ({inputs}) does not exist.")
            inputs = pd.read_csv(inputs)
            if "Sentence" not in inputs.columns:
                raise ValueError(f"inputs file ({inputs}) must have `Sentence` column.")
            inputs = inputs["Sentence"]
        if not isinstance(inputs, typing.Collection):
            raise TypeError("Must supply valid inputs path or container.")
        return SentenceCollection(inputs)

    @staticmethod
    def prep_outdir(outdir: str | pathlib.Path) -> pathlib.Path:
        if isinstance(outdir, str):
            outdir = pathlib.Path(outdir).resolve()
        if not isinstance(outdir, pathlib.Path):
            raise TypeError("outdir must be valid path type.")
        return outdir

    @staticmethod
    def prep_model(model: str | ModelInterface) -> Model | ModelInterface:
        if isinstance(model, str):
            return Model(model)
        if not isinstance(model, ModelInterface):
            raise TypeError("model must implement `score` and `embed`.")
        return model

    @staticmethod
    def prep_objective(
        objective: str | ObjectiveInterface,
    ) -> Objective | ObjectiveInterface:
        if isinstance(objective, str):
            supported = Objective.supported_functions().keys()
            if objective not in supported:
                raise ValueError("objective must be one of {supported}.")
            return Objective(objective)
        if not isinstance(objective, ObjectiveInterface):
            raise TypeError("objective must implement `evaluate`.")
        return objective

    @staticmethod
    def prep_solver(solver: str) -> str:
        if not isinstance(solver, str):
            raise TypeError("solver only accepts type `str`.")
        supported = Optimizer.supported_solvers().keys()
        if solver not in supported:
            raise ValueError("solver must be one of {supported}.")
        return solver

    @staticmethod
    def prep_constraint(constraint: str) -> str:
        if not isinstance(constraint, str):
            raise TypeError("constraint only accepts type `str`.")
        supported = Optimizer.supported_constraints()
        if constraint not in supported:
            raise ValueError("constraint must be one of {supported}.")
        return constraint

    @staticmethod
    def prep_seqlen(seqlen: int) -> int:
        if not isinstance(seqlen, int):
            raise TypeError("seqlen only accepts type `int`.")
        if seqlen < -1 or seqlen == 0:
            raise ValueError("seqlen must be positive or -1 for all.")
        return seqlen

    @staticmethod
    def prep_maximize(maximize: bool) -> bool:
        if not isinstance(maximize, bool):
            raise TypeError("maximize only accepts type `bool`.")
        return maximize