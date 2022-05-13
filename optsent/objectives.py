import typing

import numpy as np

from optsent.abstract import Object, ModelInterface


class Objective(Object):
    def __init__(self, objective_id: str) -> None:
        super().__init__()
        if not isinstance(objective_id, str):
            raise TypeError("objective_id must be type `str`.")
        self._id = objective_id
        try:
            self._objective = self.supported_functions()[self._id]()
        except KeyError as invalid_objective:
            raise ValueError(
                f"objective_id must be in supported: {self.supported_functions().keys()}"
            ) from invalid_objective
        self.info(f"Defined {self._objective._name[1:]} objective.")

    @classmethod
    def supported_functions(cls) -> typing.Dict[str, typing.Callable]:
        return {
            "normlogp": _NormJointLogProb,
            "embsim": _EmbeddingSimilarity,
        }

    def evaluate(self, sent1: str, sent2: str, model: ModelInterface) -> float:
        if not all(
            isinstance(arg, type)
            for arg, type in zip((sent1, sent2, model), (str, str, ModelInterface))
        ):
            raise TypeError("arguments must adhere to interface to get evaluated.")
        return self._objective(sent1, sent2, model)


class _NormJointLogProb(Object):
    @staticmethod
    def __call__(sent1: str, sent2: str, model: ModelInterface) -> float:
        return model.score(sent1 + sent2) - (model.score(sent1) + model.score(sent2))


class _EmbeddingSimilarity(Object):
    @staticmethod
    def __call__(sent1: str, sent2: str, model: ModelInterface) -> float:
        emb1, emb2 = model.embed(sent1), model.embed(sent2)
        cos_sim = (emb1 @ emb2.T).item() / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        if np.abs(cos_sim) == 1.0:
            return np.sign(cos_sim) * np.Inf
        return np.arctanh(cos_sim)
