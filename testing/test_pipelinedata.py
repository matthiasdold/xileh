#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 2021-03-26

import pytest
import tempfile
import numpy as np

from pathlib import Path

from xileh.core.pipelinedata import xPData, from_container
from xileh.core.pipelinedata import CheckedList
from xileh.core.pipelinedata import from_dict



@pytest.fixture
def get_test_data():
    tdata = xPData(
        data=np.eye(5),
        header={'name': 'test_data',
                'description': 'Some data description'},
        meta={'mean': 5}
    )
    return tdata


@pytest.fixture
def get_nested_test_data():
    tdata = xPData(
        data=[
            xPData(
                data=[
                    xPData(data=np.eye(3), header={'name': 'test'}),
                    xPData(data=[1, 23, 4],
                           header={'name': 'somename'})
                ],
                header={'name': '1st_level_child'},
            ),
            xPData(
                data=np.ones(3),
                header={'name': 'search_target',
                        'discription': 'We will search for this'}
            ),
            xPData(data=np.zeros(5), header={'name': 'not the target'})
        ],
        header={'name': 'outer_container',
                'description': 'A parent container without name'},
        meta={'some_meta': [1, 2, 3]}
    )
    return tdata


def test_init_meta():
    with pytest.raises(ValueError):
        tdata = xPData(
            data=np.eye(5),
            header={'name': 'test', 'description': 'Some data description'},
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

    # header needs to at least contain name
    with pytest.raises(AssertionError):
        xPData(None, header={})


def test_get_by_name(get_nested_test_data):
    td = get_nested_test_data

    assert td.get_by_name('search_target') == td.data[1]
    assert isinstance(td.get_by_name('new', create_if_missing=True), xPData)
    assert 'new' in td.get_container_names()

    assert td.get_by_name('somename',
                          find_parent=True).name == '1st_level_child'

    td.get_by_name('new')   # -> will cache the result
    td.delete_by_name('new')
    assert 'new' not in td.get_containers()

    # test data also the cache is cleared
    assert td.get_by_name('new') is None, "Deletion of did not clear cache"


def test_creation_via_new(get_nested_test_data):
    get_nested_test_data.add(np.zeros(10),
                             name="my_new_test",
                             header={'description': 'some_description'},
                             meta={'plus_one': np.ones(10)}
                             )

    assert 'my_new_test' in get_nested_test_data.get_container_names()

    trg = get_nested_test_data['my_new_test']
    assert all(trg.data == np.zeros(10))
    assert trg.header['description'] == 'some_description'
    assert all(trg.meta['plus_one'] == np.ones(10))


def test_checked_list():
    tdata = xPData(None, header={'name': 'test'})

    assert CheckedList([1, 2, 3], tdata) == [1, 2, 3]

    chk_list = CheckedList([], tdata)
    chk_list.append('a')
    assert chk_list == ['a']

    with pytest.raises(AssertionError):
        chk_list.append(xPData([1, 2, 4], header={'name': 'test'}))


def test_checked_list_container(get_nested_test_data):
    """ Test if the checked list is used if a container
        is initialized with a list
    """
    td = get_nested_test_data
    assert isinstance(td.data, CheckedList)

    with pytest.raises(AssertionError):
        td.data.append(xPData([1], header={'name': 'search_target'}))


def test_name_attribute():
    td = xPData(name='test')

    assert td.name == 'test'
    assert td.header['name'] == 'test'

    # setter needs to overwrite attribute
    td.header = {'name': 'another name'}
    assert td.name == 'another name'

    # setting with header and name
    td = xPData(header={'description': 'aaaa', 'somekey': 'somevalue'},
                name='test')

    with pytest.raises(AssertionError):
        td = xPData(header={'name': 'testing'}, name='testing')
        td = xPData()   # should yield an error as no name is provided


def test_create_if_missing(get_test_data):
    td = get_test_data

    with pytest.raises(ValueError):
        td.get_by_name('test_data_2', create_if_missing=True)

    td.data = []
    td2 = td.get_by_name('test_data2', create_if_missing=True)
    assert isinstance(td2, xPData)
    assert len(td.get_container_names())
    assert td2.name == 'test_data2'


def test__to_dict(get_nested_test_data):
    td = get_nested_test_data           # just for brevity

    dtd = td._to_dict()

    # we only have one outer container
    assert len(list(dtd.keys())) == 2, "Dict of xPData should have 2 outer key"
    assert list(dtd.keys())[0] == td.name, "Name missmatch"
    assert list(dtd.keys())[1] == 'type' and dtd['type'] == 'xPData', \
        "Missing a type==xPData key:value pair"

    # lists conserved
    assert (len(dtd[td.name]['data']) == len(td.data)), "Data list missmatch"

    # deepest level children
    assert (list(dtd[td.name]['data'][0].values())[0]['data'][0]
            == td.get_by_name('1st_level_child').data[0]._to_dict())


def test_from_dict(get_nested_test_data):
    t = from_dict(get_nested_test_data._to_dict())

    assert t.get_containers() == get_nested_test_data.get_containers(), \
        "Inconsistent containers in from_dict(_to_dict()) cycle"

    assert t.get_by_name('outer_container').meta['some_meta'] == [1, 2, 3], \
        "Meta information got lost during read _from_dict"


def test_accessor(get_nested_test_data):
    t = get_nested_test_data

    assert t.get_by_name('somename') == t['somename'], "Dict accessor failed"

    new = t.get_by_name('somenewlongname', create_if_missing=True)
    new.data = 1241254
    assert t.get_by_name('somenewlongname') == t['somenewlongname'],\
        "Dict accessor failed after new container insertion"

    t['somename'].data = 'newvalue'
    assert t.get_by_name('somename').data == t['somename'].data,\
        "Dict accessor failed after modifying via dict key"

    nv = t.get_by_name('somename')
    nv.data = 'anothernewvalue'
    assert t.get_by_name('somename').data == t['somename'].data,\
        "Dict accessor failed after modifying via get_by_name"

    t.delete_by_name('somename')
    assert t['somename'] is None, "Deletion failed with dict accessor"


def test_assignment(get_nested_test_data):
    pdata = get_nested_test_data

    new_pdata = xPData([1, 2, 3, 'test'], name="newname",
                       meta={'some_meta': [1, 1, 1, 1]})

    pdata['test'] = new_pdata

    assert pdata['test'].data == new_pdata.data, "Assignment failed"
    assert pdata['test'].meta == new_pdata.meta, "Assignment failed"


def test_save_and_load(get_nested_test_data):
    with tempfile.TemporaryDirectory() as tmp:
        fname = Path(tmp, 'fname')
        get_nested_test_data.save(fname)

        assert fname.exists(), "Folder was not created"

        loaded = from_container(fname)

        for cn in loaded.get_container_names():
            assert loaded[cn].data == get_nested_test_data[cn].data, \
                f"Missmatch for {cn}"
