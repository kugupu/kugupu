from collections import namedtuple
import numpy as np
import pytest
import networkx as nx
from numpy.testing import assert_equal, assert_almost_equal

import kugupu as kgp

@pytest.fixture(params=[True, False])
def graph(request):
    graph_stuff = namedtuple('Graph',
                             'graph,weighted,adj,lap,res')

    # the graph is same for both,
    # just the expected results change
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
    graph.add_weighted_edges_from([
        (1, 2, 1.0), (1, 3, 1.0), (2, 4, 1.0), (3, 4, 1.0),
        (4, 5, 2.0), (4, 6, 2.0), (6, 7, 2.0), (5, 7, 2.0),
    ], weight='Hij'
    )
    if request.param:
        # expected results for weighted=True
        weighted=True
        adj = np.array([[0., 1., 1., 0., 0., 0., 0.],
                        [1., 0., 0., 1., 0., 0., 0.],
                        [1., 0., 0., 1., 0., 0., 0.],
                        [0., 1., 1., 0., 2., 2., 0.],
                        [0., 0., 0., 2., 0., 0., 2.],
                        [0., 0., 0., 2., 0., 0., 2.],
                        [0., 0., 0., 0., 2., 2., 0.]])
        lap = np.array([[ 2., -1., -1., -0., -0., -0., -0.],
                        [-1.,  2., -0., -1., -0., -0., -0.],
                        [-1., -0.,  2., -1., -0., -0., -0.],
                        [-0., -1., -1.,  6., -2., -2., -0.],
                        [-0., -0., -0., -2.,  4., -0., -2.],
                        [-0., -0., -0., -2., -0.,  4., -2.],
                        [-0., -0., -0., -0., -2., -2.,  4.]])
        res = np.array([[0.   , 0.75 , 0.75 , 1.   , 1.375, 1.375, 1.5  ],
                        [0.75 , 0.   , 1.   , 0.75 , 1.125, 1.125, 1.25 ],
                        [0.75 , 1.   , 0.   , 0.75 , 1.125, 1.125, 1.25 ],
                        [1.   , 0.75 , 0.75 , 0.   , 0.375, 0.375, 0.5  ],
                        [1.375, 1.125, 1.125, 0.375, 0.   , 0.5  , 0.375],
                        [1.375, 1.125, 1.125, 0.375, 0.5  , 0.   , 0.375],
                        [1.5  , 1.25 , 1.25 , 0.5  , 0.375, 0.375, 0.   ]])
    else:
        # expected results for weighted=False
        weighted=False
        adj = np.array([[0., 1., 1., 0., 0., 0., 0.],
                        [1., 0., 0., 1., 0., 0., 0.],
                        [1., 0., 0., 1., 0., 0., 0.],
                        [0., 1., 1., 0., 1., 1., 0.],
                        [0., 0., 0., 1., 0., 0., 1.],
                        [0., 0., 0., 1., 0., 0., 1.],
                        [0., 0., 0., 0., 1., 1., 0.]])
        lap = np.array([[ 2, -1, -1,  0,  0,  0,  0],
                        [-1,  2,  0, -1,  0,  0,  0],
                        [-1,  0,  2, -1,  0,  0,  0],
                        [ 0, -1, -1,  4, -1, -1,  0],
                        [ 0,  0,  0, -1,  2,  0, -1],
                        [ 0,  0,  0, -1,  0,  2, -1],
                        [ 0,  0,  0,  0, -1, -1,  2]])
        res = np.array([[0.  , 0.75, 0.75, 1.  , 1.75, 1.75, 2.  ],
                        [0.75, 0.  , 1.  , 0.75, 1.5 , 1.5 , 1.75],
                        [0.75, 1.  , 0.  , 0.75, 1.5 , 1.5 , 1.75],
                        [1.  , 0.75, 0.75, 0.  , 0.75, 0.75, 1.  ],
                        [1.75, 1.5 , 1.5 , 0.75, 0.  , 1.  , 0.75],
                        [1.75, 1.5 , 1.5 , 0.75, 1.  , 0.  , 0.75],
                        [2.  , 1.75, 1.75, 1.  , 0.75, 0.75, 0.  ]])

    return graph_stuff(graph, weighted, adj, lap, res)


@pytest.fixture()
def square_graph():
    #  /2\
    # 1   3
    #  \4/
    g = nx.Graph()

    g.add_nodes_from([1, 2, 3, 4])
    g.add_weighted_edges_from([
        (1, 2, 1.0), (2, 3, 1.0),
        (1, 4, 1.0), (4, 3, 1.0),
        ], weight='Hij',
    )

    return g

@pytest.fixture
def square_adjacency():
    return np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ])

@pytest.fixture
def square_laplacian():
    return np.array([
        [2, -1, 0, -1],
        [-1, 2, -1, 0],
        [0, -1, 2, -1],
        [-1, 0, -1, 2],
    ])

@pytest.fixture
def square_resistance_distance():
    return np.array([
        [0, 0.75, 1, 0.75],
        [0.75, 0, 0.75, 1],
        [1, 0.75, 0, 0.75],
        [0.75, 1, 0.75, 0],
    ])


def test_square_adjacency(square_graph, square_adjacency):
    assert_equal(kgp.networks.adjacency_matrix(square_graph),
                 square_adjacency)

def test_square_laplacian(square_graph, square_laplacian):
    assert_equal(kgp.networks.laplacian_matrix(square_graph),
                 square_laplacian)

def test_square_resistance_distance(square_graph, square_resistance_distance):
    assert_almost_equal(
        kgp.networks.resistance_distance_matrix(square_graph),
        square_resistance_distance)


def test_adjacency(graph):
    assert_almost_equal(
        graph.adj,
        kgp.networks.adjacency_matrix(graph.graph, weighted=graph.weighted))

def test_laplacian(graph):
    assert_almost_equal(
        graph.lap,
        kgp.networks.laplacian_matrix(graph.graph, weighted=graph.weighted))

def test_resistance_distance(graph):
    assert_almost_equal(
        graph.res,
        kgp.networks.resistance_distance_matrix(graph.graph, weighted=graph.weighted))
