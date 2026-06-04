#!/usr/bin/env python
#
# Shared pytest fixtures, auto-discovered by pytest across the tests/ package.

import numpy as np
import pytest

from xileh import xData


@pytest.fixture
def get_test_data():
    return xData(
        data=np.eye(5),
        header={'name': 'test_data',
                'description': 'Some data description'},
        meta={'mean': 5},
    )


@pytest.fixture
def get_nested_test_data():
    return xData(
        data=[
            xData(
                data=[
                    xData(data=np.eye(3), header={'name': 'test'}),
                    xData(data=[1, 23, 4], header={'name': 'somename'}),
                ],
                header={'name': 'first_level_child'},
            ),
            xData(
                data=[
                    xData(data=np.eye(4), header={'name': 'test2'}),
                    xData(data=[1, 23, 4], header={'name': 'somename2'}),
                ],
                header={'name': 'string_nest_c'},
            ),
            xData(
                data=np.ones(3),
                header={'name': 'search_target',
                        'discription': 'We will search for this'},
            ),
            xData(data=np.zeros(5), header={'name': 'not the target'}),
        ],
        name='outer_container',
        header={'description': 'A parent container without name'},
        meta={'some_meta': [1, 2, 3]},
    )
