from compare_utils import _compare_container

from xileh import xData
from xileh.utils.datahandler.saving import save_to_folder
from xileh.utils.datahandler.loading import (
    load_container,
    get_loader,
    load_pandas,
    load_pandas_series,
    load_polars,
    load_polars_series,
    load_numpy,
)

import os
import pathlib
from pathlib import Path

import numpy as np
import tempfile

import pytest


def _make_dataframe(backend):
    """Build a (DataFrame, Series) pair for the requested backend."""
    if backend == "pandas":
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        return df, df["a"]
    import polars as pl
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    return df, df["a"]


def test_save_load_cycle_with_dict(get_nested_test_data):
    d = get_nested_test_data._to_dict()

    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp, "test_container")

        save_to_folder(d, fname=fpath)
        ld = load_container(fpath)

    _compare_container(d, ld)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_save_load_cycle_with_dataframe(backend):
    pytest.importorskip(backend)
    df, series = _make_dataframe(backend)

    container = xData(
        data=[
            xData(data=df, header={"name": "frame"}),
            xData(data=series, header={"name": "series"}),
            xData(data=np.eye(3), header={"name": "arr"}),
        ],
        name="df_container",
    )
    d = container._to_dict()

    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp, "test_container")
        save_to_folder(d, fname=fpath)
        ld = load_container(fpath)

    _compare_container(d, ld)


def test_get_loader():
    # check loader is working for individual file types
    assert get_loader("<type pathlib.Path>") == pathlib.Path
    assert get_loader("<type pathlib.PosixPath>") == pathlib.Path
    assert get_loader("<type pathlib.WindowsPath>") == pathlib.Path
    assert get_loader("<type pandas.DataFrame>") == load_pandas
    assert get_loader("<type pandas.Series>") == load_pandas_series
    assert get_loader("<type polars.DataFrame>") == load_polars
    assert get_loader("<type polars.Series>") == load_polars_series
    assert get_loader("<type numpy.ndarray>") == load_numpy


def test_relative_paths_for_extra_data(get_nested_test_data):

    d = get_nested_test_data._to_dict()

    with tempfile.TemporaryDirectory() as tmp1:
        with tempfile.TemporaryDirectory() as tmp2:
            fpath1 = Path(tmp1, "test_container")
            fpath2 = Path(tmp2, "test_container")

            save_to_folder(d, fname=fpath1)

            os.system(f"mv {fpath1} {fpath2}")

            ld = load_container(fpath2)

    _compare_container(d, ld)
