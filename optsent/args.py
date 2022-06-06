import hashlib
import multiprocessing
import pathlib
import typing

import numpy as np
import pandas as pd

from optsent.abstract import Object, IModel, IObjective, IOptimizer
from optsent.data import SentenceCollection
from optsent.models import Model
from optsent.objectives import Objective
from optsent.optimizers import Optimizer


class ArgTool(Object):
    def _log_arg(self, name: str, value: typing.Any) -> None:
        indent = " " * (12 - len(name))
        self.info(f"{name}{indent}{value}")

    def log_args(self, kwargs: typing.Dict[str, typing.Any]) -> None:
        for name, value in kwargs.items():
            self._log_arg(name, value)

    def get_unique_id(self, kwargs: typing.Dict[str, typing.Any]) -> str:
        def md5(obj):
            return hashlib.md5(str(obj).encode()).hexdigest()

        elements = ["max" if kwargs["maximize"] else "min"]
        inputs = kwargs["inputs"]
        if isinstance(inputs, (str, pathlib.Path)):
            elem = str(inputs).rsplit("/", maxsplit=1)[-1].split(".")[0]
        else:
            elem = f"CUSTOM{md5(inputs)}"
        elements.append(elem)
        for key in ["objective", "optimizer", "constraint", "model"]:
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
    def prep_model(model: str | IModel) -> Model | IModel:
        if isinstance(model, str):
            return Model(model)
        if isinstance(model, type):
            raise TypeError("model must be an instance of a class, not a type.")
        if not isinstance(model, IModel):
            raise TypeError("custom model must implement `optsent.abstract.IModel`.")
        return model

    @staticmethod
    def prep_objective(
        objective: str | IObjective,
    ) -> Objective | IObjective:
        if isinstance(objective, str):
            supported = Objective.supported_functions().keys()
            if objective not in supported:
                raise ValueError(f"objective must be one of {supported}.")
            return Objective(objective)
        if isinstance(objective, type):
            raise TypeError("objective must be an instance of a class, not a type.")
        if not isinstance(objective, IObjective):
            raise TypeError(
                "custom objective must implement `optsent.abstract.IObjective`."
            )
        return objective

    @staticmethod
    def build_optimizer(kwargs: typing.Dict[str, typing.Any]) -> Optimizer | IOptimizer:
        optim = kwargs["optimizer"]
        if isinstance(optim, str):
            return Optimizer(
                optimizer=optim,
                constraint=kwargs["constraint"],
                seqlen=kwargs["seqlen"],
                maximize=kwargs["maximize"],
            )
        if isinstance(optim, type):
            raise TypeError("optimizer must be an instance of a class, not a type.")
        if not isinstance(optim, IOptimizer):
            raise TypeError(
                "custom optimizer must implement `optsent.abstract.IOptimizer`."
            )
        return optim

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
        if not isinstance(inputs, typing.Collection):
            raise TypeError("must supply valid inputs path or container.")
        if isinstance(inputs, pd.DataFrame):
            if "Sentence" not in inputs.columns:
                raise ValueError("inputs must have `Sentence` column if pd.DataFrame.")
            inputs = inputs["Sentence"]
        if isinstance(inputs, np.ndarray):
            if inputs.ndim > 1:
                inputs = inputs.squeeze()
                if inputs.ndim > 1:
                    raise ValueError("np.ndarray inputs must be 1D or singleton.")
        if not isinstance(inputs, pd.Series):
            try:
                inputs = pd.Series(inputs, dtype=str)
            except Exception as invalid_container:
                raise TypeError("invalid inputs container type.") from invalid_container
        if not inputs.size > 1:
            raise ValueError("inputs container must have at least 2 elements.")
        if not all(isinstance(i, str) for i in inputs):
            raise TypeError("inputs container must only contain strings.")
        return SentenceCollection(inputs)

    @staticmethod
    def prep_outdir(outdir: str | pathlib.Path) -> pathlib.Path:
        if isinstance(outdir, str):
            outdir = pathlib.Path(outdir).resolve()
        if not isinstance(outdir, pathlib.Path):
            raise TypeError("outdir must be valid path type.")
        return outdir

    @staticmethod
    def prep_optimizer(optimizer: str | IOptimizer) -> str | IOptimizer:
        if not isinstance(optimizer, str):
            return optimizer
        supported = Optimizer.supported_optimizers().keys()
        if optimizer not in supported:
            raise ValueError(f"optimizer must be one of {supported}.")
        return optimizer

    @staticmethod
    def prep_constraint(constraint: str) -> str:
        if not isinstance(constraint, str):
            raise TypeError("constraint only accepts type `str`.")
        supported = Optimizer.supported_constraints()
        if constraint not in supported:
            raise ValueError(f"constraint must be one of {supported}.")
        return constraint

    @staticmethod
    def prep_seqlen(seqlen: int) -> int:
        if not isinstance(seqlen, int):
            raise TypeError("seqlen only accepts type `int`.")
        if not (seqlen == -1 or seqlen > 1):
            raise ValueError("seqlen must be >1 or -1 for all.")
        return seqlen

    @staticmethod
    def prep_maximize(maximize: bool) -> bool:
        if not isinstance(maximize, bool):
            raise TypeError("maximize only accepts type `bool`.")
        return maximize

    @staticmethod
    def prep_ncores(ncores: int) -> int:
        if not isinstance(ncores, int):
            raise TypeError("ncores only accepts type `int`.")
        max_cores = multiprocessing.cpu_count()
        if not ((np.abs(ncores) <= max_cores) and (ncores not in [0, -max_cores])):
            raise ValueError(
                f"ncores must be != 0 and <= number of cores: {max_cores}."
            )
        return ncores

    @staticmethod
    def prep_export(export: bool) -> bool:
        if not isinstance(export, bool):
            raise TypeError("export only accepts type `bool`.")
        return export
