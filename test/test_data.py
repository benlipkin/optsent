import numpy as np
import pandas as pd

from test_abstract import check_raises

from optsent.data import Graph, SentenceCollection


def test_graph_contructor():
    def check_output(graph, size):
        assert graph.dim == size

    cls = Graph
    for arg in (3, len(range(10))):
        check_output(cls(arg), arg)
    for arg in ("10", 11.1):
        check_raises(cls, arg, TypeError)
    for arg in (-1, 0, 1):
        check_raises(cls, arg, ValueError)


def test_graph_writer():
    def check_side_effect(graph, arg):
        np.testing.assert_approx_equal(graph.matrix[arg[0], arg[1]], arg[2])

    dim = 10
    graph = Graph(dim)
    func = graph.write_transition_weight
    for arg in ((1, 2, 2.14), (7, 3, 2)):
        func(*arg)
        check_side_effect(graph, arg)
    for arg in ((3.14, 2, 3), (2, 3, [7]), ("1", 2, 3)):
        check_raises(func, arg, TypeError)
    for arg in ((11, 2, 2.14), (7, 10, 2)):
        check_raises(func, arg, ValueError)


def test_collection_constructor():
    def check_output(coll, sents):
        pd.testing.assert_series_equal(coll.sentences, sents)
        assert coll.graph.dim == sents.size

    cls = SentenceCollection
    for arg in (pd.Series(["abc", "123"]), pd.Series(["abc", "123", "def"])):
        check_output(cls(arg), arg)
    for arg in (["abc", "123"], pd.Series([123, 456])):
        check_raises(cls, arg, TypeError)
    check_raises(cls, pd.Series(["abc"]), ValueError)
