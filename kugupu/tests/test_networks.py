import numpy as np
import pytest
import networkx as nx
from numpy.testing import assert_equal, assert_almost_equal

import kugupu as kgp


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
