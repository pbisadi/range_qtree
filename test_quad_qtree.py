import string

import pandas
import pytest
import numpy.random as random
from quad_tree import QTree, Node


@pytest.fixture(scope="module")
def norm_data_frame():
    row_count = 100
    return pandas.DataFrame({
        'x': [random.normal(0, 2) for _ in range(row_count)],
        'y': [random.normal(0, 2) for _ in range(row_count)],
        'name': [string.ascii_uppercase[random.randint(0, 26)] for _ in range(row_count)]
    })


@pytest.fixture(scope="module")
def uniform_data_frame():
    uniform_df = pandas.DataFrame(columns=['x', 'y', 'name'])
    i = 0
    range_size = 4
    for x in range(-range_size, range_size):
        for y in range(-range_size, range_size):
            uniform_df.loc[i] = [x + .5, y + .5, string.ascii_uppercase[random.randint(0, 26)]]
            i += 1
    return uniform_df


def test_data_frame(norm_data_frame):
    """Test initialization of a QTree"""
    qt = QTree(norm_data_frame, ['x', 'y'], 10)


def test_max_depth_on_uniform_dist(uniform_data_frame):
    """Test the structure of QTree when the points are uniformly distributed"""
    qt = QTree(uniform_data_frame, ['x', 'y'], 1)
    all_leaves = qt.root.all_leaf_nodes()
    # pyplot.scatter(*list(zip(*c)))
    assert len(all_leaves) == 16


def test_reject_qtree_on_categorical_columns(norm_data_frame):
    assert False
