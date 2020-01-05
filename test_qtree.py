import string

import pandas
import pytest
import numpy.random as random
from qtree import QTree, Point, Node


@pytest.fixture(scope="module")
def data_frame():
    row_count = 200
    return pandas.DataFrame({
        'x': [random.normal(0, 2) for _ in range(row_count)],
        'y': [random.normal(0, 2) for _ in range(row_count)],
        'name' : [string.ascii_uppercase[random.randint(0, 26)] for _ in range(row_count)]
    })


def test_qtree_initialize():
    qt = QTree([Point(random.normal(0, 2), random.normal(0, 2)) for x in range(200)], 10)


def test_data_frame(data_frame):
    qt = QTree(data_frame, ['x', 'y'], 10)


def test_reject_qtree_on_categorical_columns(data_frame):
    assert False