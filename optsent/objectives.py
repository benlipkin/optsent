import typing

import numpy as np

from optsent.abstract import Object, ModelInterface


class Objective(Object):
    def __init__(self, objective_id: str) -> None:
        super().__init__()
        self._id = objective_id
        self._objective = self.supported_functions()[self._id]()
        self.log(f"Defined {self._objective._name} objective.")

    @classmethod
    def supported_functions(cls) -> typing.Dict[str, typing.Callable]:
        return {
            "normlogp": NormJointLogProb,
            "embsim": EmbeddingSimilarity,
        }

    def evaluate(self, sent1: str, sent2: str, model: ModelInterface) -> float:
        if not any(
            isinstance(arg, type)
            for arg, type in zip((sent1, sent2, model), (str, str, ModelInterface))
        ):
            raise TypeError("arguments must adhere to interface to get evaluated.")
        return self._objective(sent1, sent2, model)


class NormJointLogProb(Object):
    @staticmethod
    def __call__(sent1: str, sent2: str, model: ModelInterface) -> float:
        return model.score(sent1 + sent2) - (model.score(sent1) + model.score(sent2))


class EmbeddingSimilarity(Object):
    @staticmethod
    def __call__(sent1: str, sent2: str, model: ModelInterface) -> float:
        emb1, emb2 = model.embed(sent1), model.embed(sent2)
        cos_sim = (emb1 @ emb2.T).item() / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return np.arctanh(cos_sim)
