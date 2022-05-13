import pathlib

from test_abstract import check_raises

from optsent.abstract import IModel, IObjective, IOptimizer
from optsent.optsent import OptSent


class MockCustomOptimizer:
    pass  # need to test this


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
    def check_output(seq):
        assert len(seq) == 3

    cls = OptSent
    fname = pathlib.Path(__file__).parent / "test_inputs" / "test_strings.txt"
    for arg in ({"inputs": fname, "export": True}, {"inputs": fname, "export": False}):
        check_output(cls(**arg).run())
