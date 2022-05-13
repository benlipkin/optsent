import pytest

from optsent.abstract import Object, ModelInterface, ObjectiveInterface


def check_raises(call, arg, exception):
    with pytest.raises(exception):
        if isinstance(arg, tuple):
            call(*arg)
        else:
            call(arg)


def check_interface(cls, interface):
    assert isinstance(cls, interface)


def test_interfaces():
    check_raises(ModelInterface, (), TypeError)
    check_raises(ObjectiveInterface, (), TypeError)


def test_logging():
    class MockObject(Object):
        pass

    for arg in ("info", "warn"):
        func = getattr(MockObject(), arg)
        check_raises(func, 123, TypeError)
