import pytest

from optsent.abstract import Object, IModel, IObjective, IOptimizer


def check_raises(call, arg, exception):
    with pytest.raises(exception):
        if isinstance(arg, tuple):
            call(*arg)
        else:
            call(arg)


def check_interface(cls, interface):
    assert isinstance(cls, interface)


def test_interfaces():
    check_raises(IModel, (), TypeError)
    check_raises(IObjective, (), TypeError)
    check_raises(IOptimizer, (), TypeError)


def test_logging():
    class MockObject(Object):
        pass

    for arg in ("info", "warn"):
        func = getattr(MockObject(), arg)
        check_raises(func, 123, TypeError)
