import numpy as np

from test_abstract import check_raises, check_interface

from optsent.abstract import IObjective
from optsent.models import Model
from optsent.objectives import Objective


def test_objective_constructor():
    cls = Objective
    for arg in cls.supported_functions():
        check_interface(cls(arg), IObjective)
    check_raises(cls, 123, TypeError)
    check_raises(cls, "fake", ValueError)


def test_objective_normlogp():
    def check_pair(obj1, obj2):
        assert obj1 > obj2

    model = Model("gpt2")
    func = Objective("normlogp").evaluate
    check_pair(func("Hello, ", "my name", model), func("Hello, ", "name my", model))
    for arg in (("a", "b", "c"), ("a", 1, model), (1, "a", model)):
        check_raises(func, arg, TypeError)


def test_objective_embsim():
    def check_same(obj):
        np.testing.assert_approx_equal(obj, np.Inf)

    def check_pair(obj1, obj2):
        assert obj1 > obj2

    model = Model("gpt2")
    func = Objective("embsim").evaluate
    check_same(func("Hello!", "Hello!", model))
    check_pair(
        func("this is a cat", "this is a dog", model),
        func("this is a cat", "last year's presidential election", model),
    )
    for arg in (("a", "b", "c"), ("a", 1, model), (1, "a", model)):
        check_raises(func, arg, TypeError)
