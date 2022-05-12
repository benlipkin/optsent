import abc
import logging
import pathlib
import typing

import numpy as np
import numpy.typing as npt


class Object(abc.ABC):
    def __init__(self) -> None:
        self._base = pathlib.Path(__file__).parents[1]
        self._name = self.__class__.__name__
        self._logger = logging.getLogger(self._name)

    def log(self, message: str) -> None:
        indent = " " * (20 - len(self._name))
        self._logger.info(f"{indent}{message}")

    def warn(self, message: str) -> None:
        indent = " " * (20 - len(self._name))
        self._logger.warning(f"{indent}{message}")

    def __setattr__(self, name: str, value: typing.Any) -> None:
        super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> typing.Any:
        return super().__getattribute__(name)


@typing.runtime_checkable
class ModelInterface(typing.Protocol):
    def score(self, sent: str) -> float:
        raise NotImplementedError()  # pragma: no cover

    def embed(self, sent: str) -> npt.NDArray[np.float32]:
        raise NotImplementedError()  # pragma: no cover


@typing.runtime_checkable
class ObjectiveInterface(typing.Protocol):
    def evaluate(self, sent1: str, sent2: str, model: ModelInterface) -> float:
        raise NotImplementedError()  # pragma: no cover
