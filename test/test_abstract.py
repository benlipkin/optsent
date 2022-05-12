import pytest

from optsent.abstract import ModelInterface, ObjectiveInterface


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
