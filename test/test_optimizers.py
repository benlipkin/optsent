import itertools
import pathlib

import numpy as np

from test_abstract import check_raises

from optsent.args import ArgTool
from optsent.optimizers import Optimizer


class MockCustomObjective:
    pass  # need to test this


def get_default_coll():
    fname = pathlib.Path(__file__).parent / "test_inputs" / "test_strings.txt"
    coll = ArgTool().prep_inputs(fname)
    return coll


def fill_graph(coll):
    for val, (i, j) in enumerate(itertools.product(range(coll.size), range(coll.size))):
        coll.graph.write_transition_weight(i, j, val)


def test_optimizer_constructor():
    def check_output(optim):
        assert hasattr(optim, "solve")

    cls = Optimizer
    check_output(cls("greedy", "repeats", 100, True))
    for arg in ((1, "", 1, True), ("", 1, 1, True), ("", "", "", True), ("", "", 1, 1)):
        check_raises(cls, arg, TypeError)
    for arg in (("fake", "repeats", 100, True), ("greedy", "fake", 100, True)):
        check_raises(cls, arg, ValueError)


def test_optimizer_greedy_base():
    def check_side_effect(cls):
        assert cls.indices == [0, 1, 2]
        assert cls.values == [np.nan, 1, 5]

    cls = Optimizer("greedy", "repeats", -1, False)
    func = cls.solve
    coll = get_default_coll()
    fill_graph(coll)
    func(coll)
    check_side_effect(cls)
    check_raises(func, ["abc", "123"], TypeError)
    check_raises(func, ArgTool().prep_inputs(["abc", "123"]), RuntimeError)


def test_optimizer_greedy_seqlen():
    def check_side_effect(cls, seqlen, coll):
        if seqlen <= 0 or seqlen > coll.size:
            assert len(cls.indices) == coll.size
        else:
            assert len(cls.indices) == seqlen

    for arg in (-1, 0, 1, 2, 3, 10):
        cls = Optimizer("greedy", "repeats", arg, False)
        coll = get_default_coll()
        fill_graph(coll)
        cls.solve(coll)
        check_side_effect(cls, arg, coll)


def test_optimizer_greedy_constraint():
    def check_side_effect(cls):
        assert len(cls.indices) == 2
        assert cls.indices == [0, 2]

    cls = Optimizer("greedy", "repeats", -1, False)
    test = ["a", "a", "b"]
    coll = ArgTool().prep_inputs(test)
    fill_graph(coll)
    cls.solve(coll)
    check_side_effect(cls)


def test_optimizer_greedy_maximize():
    def check_side_effect(cls):
        assert cls.indices == [2, 1, 0]
        assert cls.values == [np.nan, 7, 3]

    cls = Optimizer("greedy", "repeats", -1, True)
    coll = get_default_coll()
    fill_graph(coll)
    cls.solve(coll)
    check_side_effect(cls)
