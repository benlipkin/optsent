import pathlib

import numpy as np
import numpy.testing as npt

from test_abstract import check_raises

from optsent.abstract import IModel, IObjective, IOptimizer
from optsent.optsent import OptSent


def test_optsent_constructor():
    class ValidModel(IModel):
        pass

    class ValidObjective(IObjective):
        pass

    class ValidOptimizer(IOptimizer):
        pass

    def check_output(cls, args):
        base = "min_test_strings_objective=normlogp_optimizer=greedy_constraint=repeats_model=gpt2"
        if not all(isinstance(arg, str) for arg in args):
            assert "CUSTOM" in cls.unique_id
        else:
            assert cls.unique_id == base

    cls = OptSent
    fname = str(pathlib.Path(__file__).parent / "test_inputs" / "test_strings.txt")
    for arg in (
        (fname, "gpt2", "normlogp", "greedy"),
        (["abc", "123"], "gpt2", "normlogp", "greedy"),
        (fname, ValidModel(), "normlogp", "greedy"),
        (fname, "gpt2", ValidObjective(), "greedy"),
        (fname, "gpt2", "normlogp", ValidOptimizer()),
    ):
        check_output(cls(arg[0], model=arg[1], objective=arg[2], optimizer=arg[3]), arg)
    check_raises(cls, (), TypeError)


def test_optsent_runner():
    def check_output(table):
        assert table.shape[0] == 3
        assert table.shape[1] == 2

    cls = OptSent
    fname = pathlib.Path(__file__).parent / "test_inputs" / "test_strings.txt"
    for arg in ({"inputs": fname, "export": True}, {"inputs": fname, "export": False}):
        check_output(cls(**arg).run())


def test_optsent_custom():
    class MockCustomModel(IModel):
        @staticmethod
        def score(sent):
            return float(len(sent))

        @staticmethod
        def embed(sent):
            return np.array([0.0, float(len(sent)), 0.0])

    class MockCustomObjective(IObjective):
        @staticmethod
        def evaluate(sent1, sent2, model):
            return model.score(sent2) - model.score(sent1)

    class MockCustomOptimizer(IOptimizer):
        def __init__(self):
            self._indices = []
            self._values = []

        @property
        def indices(self):
            return self._indices

        @property
        def values(self):
            return self._values

        def solve(self, coll):
            weights = np.nanmin(coll.graph.matrix, axis=1)
            self._indices = np.argsort(weights)
            self._values = weights[self._indices]

    def check_output(table):
        npt.assert_array_equal(table.index, [2, 1, 0])
        npt.assert_array_equal(table.Sentence.values, ["abc", "ab", "a"])
        npt.assert_array_equal(table.TransitionObjective.values, [-2, -1, 1])

    inputs = ["a", "ab", "abc"]
    optsent = OptSent(
        inputs,
        model=MockCustomModel(),
        objective=MockCustomObjective(),
        optimizer=MockCustomOptimizer(),
    )
    check_output(optsent.run())
