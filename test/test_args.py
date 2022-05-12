import pathlib

import numpy as np
import pandas as pd

from test_abstract import check_raises, check_interface

from optsent.abstract import ModelInterface, ObjectiveInterface
from optsent.args import ArgTool
from optsent.optimizers import Optimizer


def test_input_prep():
    def check_output(coll, sents):
        pd.testing.assert_series_equal(coll.sentences, sents)

    func = ArgTool().prep_inputs
    fname = pathlib.Path(__file__).parent / "test_inputs" / "test_strings.txt"
    table = pd.read_csv(fname)
    for arg in (str(fname), fname.resolve(), list(table.Sentence)):
        check_output(func(arg), table.Sentence)
    for arg in (table, table.values, table.Sentence, table.Sentence.values):
        check_output(func(arg), table.Sentence)
    for arg in (123, set(table.Sentence), [table, table], pd.Series([123, 456])):
        check_raises(func, arg, TypeError)
    for arg in ([], pd.Series(["abc"], list(table))):
        check_raises(func, arg, ValueError)
    for arg in (pd.DataFrame({"a": ["b", "c"]}), np.array([["a", "b"], ["c", "d"]])):
        check_raises(func, arg, ValueError)
    check_raises(func, "fake_file", FileNotFoundError)


def test_outdir_prep():
    def check_output(path, dirname):
        assert path.resolve() == dirname.resolve()

    func = ArgTool().prep_outdir
    dirname = pathlib.Path(__file__).parents[1] / "outputs"
    for arg in (dirname, str(dirname), dirname.resolve()):
        check_output(func(arg), dirname)
    check_raises(func, 123, TypeError)


def test_model_prep():
    class ValidModel:
        def score(self, sent):
            raise NotImplementedError()  # pragma: no cover

        def embed(self, sent):
            raise NotImplementedError()  # pragma: no cover

    class InvalidModel:
        pass

    func = ArgTool().prep_model
    for arg in ("gpt2", ValidModel()):
        check_interface(func(arg), ModelInterface)
    for arg in (123, ValidModel, InvalidModel()):
        check_raises(func, arg, TypeError)
    for arg in ("fake", "distilbert"):
        check_raises(func, arg, ValueError)


def test_objective_prep():
    class ValidObjective:
        def evaluate(self, sent1, sent2, model):
            raise NotImplementedError()  # pragma: no cover

    class InvalidObjective:
        pass

    func = ArgTool().prep_objective
    for arg in ("normlogp", "embsim", ValidObjective()):
        check_interface(func(arg), ObjectiveInterface)
    for arg in (123, ValidObjective, InvalidObjective()):
        check_raises(func, arg, TypeError)
    check_raises(func, "fake", ValueError)


def test_solver_prep():
    def check_output(solver):
        assert solver in Optimizer.supported_solvers()

    func = ArgTool().prep_solver
    check_output(func("greedy"))
    check_raises(func, 123, TypeError)
    check_raises(func, "fake", ValueError)


def test_constraint_prep():
    def check_output(constraint):
        assert constraint in Optimizer.supported_constraints()

    func = ArgTool().prep_constraint
    check_output(func("repeats"))
    check_raises(func, 123, TypeError)
    check_raises(func, "fake", ValueError)


def test_seqlen_prep():
    def check_output(seqlen):
        assert seqlen > 1 or seqlen == -1

    func = ArgTool().prep_seqlen
    for arg in (-1, 100):
        check_output(func(arg))
    for arg in ("10", 4.4):
        check_raises(func, arg, TypeError)
    for arg in (-2, 0, 1):
        check_raises(func, arg, ValueError)


def test_flag_prep():
    def check_output(value, arg):
        assert value == arg

    for func in (ArgTool().prep_maximize, ArgTool().prep_export):
        for arg in (True, False):
            check_output(func(arg), arg)
        check_raises(func, "True", TypeError)