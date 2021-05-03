#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 2021-03-26

import pytest

import numpy as np
from xileh.core.pipelinedata import xPData as PData


@pytest.fixture
def get_test_data():
    tdata = PData(
        data=np.eye(5),
        header={'description': 'Some data description'},
        meta={'mean': 5}
    )
    return tdata


@pytest.fixture
def get_nested_test_data():
    tdata = PData(
        data=[
            PData(
                data=[
                    PData(data=np.eye(3)),
                    PData(data=[1, 23, 4],
                          header={'name': 'somename'})
                ],
                header={'name': '1st_level_child'}
            ),
            PData(
                data=np.ones(3),
                header={'name': 'search_target',
                        'discription': 'We will search for this'}
            ),
            PData(data=np.zeros(5), header={'name': 'not the target'})
        ],
        header={'description': 'A parent container without name'},
    )
    return tdata


def test_init_meta():
    with pytest.raises(ValueError):
        tdata = PData(
            data=np.eye(5),
            header={'description': 'Some data description'},
            meta={'mean': np.arange(3)}
        )


def test_meta(get_test_data):
    td = get_test_data

    # getters
    assert td.meta['mean'] == 5
    assert td.check_meta('mean') == 5
    assert td.check_meta('sd', missing=1) == 1

    # setters
    with pytest.raises(ValueError):
        td.update_meta({'somemeta': np.arange(3)})

    td.update_meta({'testsum': td.data.sum(axis=1)})
    assert all(td.meta['testsum'] == np.ones(5))


def test_header(get_test_data):
    td = get_test_data
    assert td.header['description'] == 'Some data description'
    assert td.check_header('description') == 'Some data description'
    assert td.check_header('other', missing='abc') == 'abc'


def test_search_by_name(get_nested_test_data):
    td = get_nested_test_data

    assert td.get_by_name('search_target') == td.data[1]
