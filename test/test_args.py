import multiprocessing
import pathlib

import numpy as np
import pandas as pd

from test_abstract import check_raises, check_interface

from optsent.abstract import IModel, IObjective, IOptimizer
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
    class ValidModel(IModel):
        pass

    class InvalidModel:
        pass

    func = ArgTool().prep_model
    for arg in ("gpt2", ValidModel()):
        check_interface(func(arg), IModel)
    for arg in (123, ValidModel, InvalidModel()):
        check_raises(func, arg, TypeError)
    for arg in ("fake", "distilbert"):
        check_raises(func, arg, ValueError)


def test_objective_prep():
    class ValidObjective(IObjective):
        pass

    class InvalidObjective:
        pass

    func = ArgTool().prep_objective
    for arg in ("normlogp", "embsim", ValidObjective()):
        check_interface(func(arg), IObjective)
    for arg in (123, ValidObjective, InvalidObjective()):
        check_raises(func, arg, TypeError)
    check_raises(func, "fake", ValueError)


def test_optimizer_build():
    class ValidOptimizer(IOptimizer):
        pass

    class InvalidOptimizer:
        pass

    func = ArgTool().build_optimizer
    for arg in (
        {
            "optimizer": "greedy",
            "constraint": "repeats",
            "seqlen": -1,
            "maximize": False,
        },
        {"optimizer": ValidOptimizer()},
    ):
        check_interface(func(arg), IOptimizer)
    for arg in (
        {"optimizer": 123},
        {"optimizer": ValidOptimizer},
        {"optimizer": InvalidOptimizer()},
    ):
        check_raises(func, arg, TypeError)


def test_optimizer_prep():
    def check_output(optimizer):
        assert optimizer in Optimizer.supported_optimizers()

    func = ArgTool().prep_optimizer
    check_output(func("greedy"))
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


def test_ncores_prep():
    def check_output(ncores, max_cores):
        assert np.abs(ncores) <= max_cores and ncores not in [0, -max_cores]

    func = ArgTool().prep_ncores
    max_cores = multiprocessing.cpu_count()
    for arg in (1, -1, max_cores, -max_cores + 1):
        check_output(func(arg), max_cores)
    for arg in ("1", 4.3):
        check_raises(func, arg, TypeError)
    for arg in (0, max_cores + 1, -max_cores):
        check_raises(func, arg, ValueError)


def test_flag_prep():
    def check_output(value, arg):
        assert value == arg

    for func in (ArgTool().prep_maximize, ArgTool().prep_export):
        for arg in (True, False):
            check_output(func(arg), arg)
        check_raises(func, "True", TypeError)
