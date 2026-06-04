#!/usr/bin/env python
#
# Tests for the scipy signal-filter wrappers in xileh.lib.signal.filter.

import numpy as np
import pytest
from scipy import signal

from xileh import xData
from xileh.lib.signal.filter import bandfilter, vectorized_filter


def _make_data(ntimes=200, nchan=3):
    """A small multi-channel signal container plus a Butterworth config."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((ntimes, nchan))
    b, a = signal.butter(2, 0.2)
    zi = np.zeros((max(len(a), len(b)) - 1, nchan))

    return xData(
        data=[
            xData(x, header={"name": "raw"}),
            xData({"a": a, "b": b, "zi": zi}, header={"name": "fcfg"}),
        ],
        name="root",
    )


def test_bandfilter_append_creates_suffixed_container():
    pdata = _make_data()
    out = bandfilter(pdata, src_container="raw",
                     lfilter_cfg_container="fcfg", mode="append")
    filtered = out.get_by_name("raw_bandfiltered")
    assert filtered is not None
    assert filtered.data.shape == out.get_by_name("raw").data.shape


def test_bandfilter_inplace_modifies_source():
    pdata = _make_data()
    original = pdata.get_by_name("raw").data.copy()
    out = bandfilter(pdata, src_container="raw",
                     lfilter_cfg_container="fcfg", mode="inplace")
    assert not np.allclose(out.get_by_name("raw").data, original)


def test_bandfilter_missing_config_raises():
    pdata = _make_data()
    with pytest.raises(ValueError):
        bandfilter(pdata, src_container="raw",
                   lfilter_cfg_container="does_not_exist")


def test_vectorized_filter_shapes_and_state():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((100, 2))
    b, a = signal.butter(2, 0.3)
    zi = np.zeros((max(len(a), len(b)) - 1, 2))

    fdata, zi_new = vectorized_filter(x, b, a, zi)
    assert fdata.shape == x.shape
    assert zi_new.shape == zi.shape
