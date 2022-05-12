import pathlib

from test_abstract import check_raises

from optsent.optsent import OptSent


def test_optsent_constructor():
    class MockModel:
        def score(self, sent):
            raise NotImplementedError()  # pragma: no cover

        def embed(self, sent):
            raise NotImplementedError()  # pragma: no cover

    class MockObjective:
        def evaluate(self, sent1, sent2, model):
            raise NotImplementedError()  # pragma: no cover

    def check_output(cls, args):
        default = "min_test_strings_objective=normlogp_solver=greedy_constraint=repeats_model=gpt2"
        if not all(isinstance(arg, str) for arg in args):
            assert "CUSTOM" in cls.unique_id
        else:
            assert cls.unique_id == default

    cls = OptSent
    fname = str(pathlib.Path(__file__).parent / "test_inputs" / "test_strings.txt")
    for arg in (
        (fname, "gpt2", "normlogp"),
        (["abc", "123"], "gpt2", "normlogp"),
        (fname, MockModel(), "normlogp"),
        (fname, "gpt2", MockObjective()),
    ):
        check_output(cls(arg[0], model=arg[1], objective=arg[2]), arg)
    check_raises(cls, (), TypeError)


def test_optsent_runner():
    def check_output(seq):
        assert len(seq) == 3

    cls = OptSent
    fname = pathlib.Path(__file__).parent / "test_inputs" / "test_strings.txt"
    for arg in ({"inputs": fname, "export": True}, {"inputs": fname, "export": False}):
        check_output(cls(**arg).run())
