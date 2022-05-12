import string

import numpy as np

from test_abstract import check_raises, check_interface

from optsent.abstract import ModelInterface
from optsent.models import Model


def test_model_constructor():
    cls = Model
    check_interface(cls("gpt2"), ModelInterface)
    check_raises(cls, 123, TypeError)
    check_raises(cls, "distilbert", ValueError)


def test_model_score():
    def check_output(score):
        assert score < 0

    def check_same(score1, score2):
        np.testing.assert_approx_equal(score1, score2)

    def check_pair(score1, score2):
        assert score1 > score2

    func = Model("gpt2").score
    for arg in ("I went to the store", string.printable):
        check_output(func(arg))
    for arg in (123, []):
        check_raises(func, arg, TypeError)
    check_same(func("Same string."), func("Same string."))
    check_pair(func("Subset of a"), func("Subset of a superset."))


def test_model_embed():
    def check_output(emb):
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        assert emb.shape[1] == 768

    def check_same(emb1, emb2):
        corr = np.corrcoef(emb1, emb2)[0, 1]
        np.testing.assert_approx_equal(corr, 1)

    func = Model("gpt2").embed
    for arg in ("I went to the store", string.printable):
        check_output(func(arg))
    for arg in (123, []):
        check_raises(func, arg, TypeError)
    check_same(func("Same string."), func("Same string."))
