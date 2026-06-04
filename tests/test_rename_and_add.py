#!/usr/bin/env python
#
# Tests for the xPData -> xData rename (with backwards-compat alias) and the
# stricter .add validation.

import pytest

import xileh
from xileh import xData, xPData


def test_xpdata_is_alias_of_xdata():
    assert xData is xPData
    assert xileh.xData is xileh.xPData


def test_alias_constructs_identical_objects():
    a = xData(data=42, name="a")
    b = xPData(data=42, name="a")
    assert type(a) is type(b)
    assert a.data == b.data == 42


def test_repr_uses_new_name():
    d = xData(data=1, name="root")
    assert repr(d).startswith("xData object")


def test_add_requires_non_empty_name():
    root = xData(data=[], name="root")
    with pytest.raises(ValueError):
        root.add([1, 2, 3], name="")


def test_add_with_name_creates_child():
    root = xData(data=[], name="root")
    root.add([1, 2, 3], name="child")
    assert root.get_by_name("child").data == [1, 2, 3]


@pytest.mark.parametrize("header,meta", [(None, None), ({}, {}), (None, {})])
def test_default_header_meta_not_shared(header, meta):
    a = xData(data=1, name="a", header=header, meta=meta)
    a.header["only_on_a"] = True
    b = xData(data=2, name="b", header=header, meta=meta)
    assert "only_on_a" not in b.header
