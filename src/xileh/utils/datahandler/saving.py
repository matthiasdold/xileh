#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220201
#
# functionality for saving an xileh container


import shutil

from pathlib import Path
from functools import wraps

import numpy as np
import tomli_w


def _to_native(obj):
    """Recursively cast numpy scalars/arrays to native Python types.

    ``tomli_w`` (unlike the legacy ``toml`` package's ``TomlNumpyEncoder``)
    cannot serialise numpy scalar types such as ``numpy.float64``, so values
    are converted to their Python equivalents before dumping.

    Parameters
    ----------
    obj : object
        Arbitrary nested structure of dicts, lists, numpy or native values.

    Returns
    -------
    object
        The same structure with numpy types replaced by native equivalents.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def prepare_save(func):
    """Prepare for any saver function by e.g. creating the fname"""

    @wraps(func)
    def prepare_wrapped(*args, **kwargs):
        Path(kwargs["fname"], "extra").mkdir(exist_ok=True, parents=True)

        # add a file prefix to fname which otherwise just points to the
        # container
        # increment the data_{n} count --> explicit names will be in the toml
        data_files = list(Path(kwargs["fname"], "extra").glob("data_*_*"))

        if data_files != []:
            nmax = max([int(p.stem.split("_")[1]) for p in data_files])
        else:
            nmax = 0

        kwargs["fname"] = Path(kwargs["fname"]).joinpath("extra", f"data_{nmax + 1}_")

        return func(*args, **kwargs)

    return prepare_wrapped


# =============================================================================
# Non serializable savers
# =============================================================================


# wrapping with prepare save ensures some common preprocessing (and identifies saver functions)     # noqa
@prepare_save
def save_serializable(data, fname=Path()):
    # get rid of the extra folder which is needed for all other loaders
    fname = fname.parents[1]

    # numpy scalars are cast to native types before dumping, as TOML (and
    # yaml) cannot serialise e.g. a numpy.float64 directly

    with open(fname.joinpath("container.toml"), "wb") as f:
        tomli_w.dump(_to_native(data), f)
    # option = (
    #     orjson.OPT_SERIALIZE_NUMPY |            # should be able to work with numpy.float etc.      # noqa
    #     orjson.OPT_NAIVE_UTC |
    #     orjson.OPT_OMIT_MICROSECONDS |
    #     orjson.OPT_INDENT_2
    # )
    # with open(fname.joinpath('container.toml'), 'wb') as f:
    #     f.write(orjson.dumps(data, option=option))


@prepare_save
def pandas_saver(data, fname=Path()):
    """Store pandas DataFrame/Series objects as parquet.

    A ``Series`` has no ``to_parquet``; it is stored as a single-column frame
    and restored to a ``Series`` on load (see ``load_pandas``).
    """
    import pandas as pd

    fname = fname.parent.joinpath(fname.stem + "pandas.parquet")
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    frame.to_parquet(fname)

    return {"extra_fname": str(fname), "type": str(type(data))}


@prepare_save
def polars_saver(data, fname=Path()):
    """Store polars DataFrame/Series objects as parquet.

    A ``Series`` is stored as a single-column frame and restored on load
    (see ``load_polars``).
    """
    import polars as pl

    fname = fname.parent.joinpath(fname.stem + "polars.parquet")
    frame = data.to_frame() if isinstance(data, pl.Series) else data
    frame.write_parquet(fname)

    return {"extra_fname": str(fname), "type": str(type(data))}


@prepare_save
def numpy_saver(data, fname=Path()):
    """Store numpy objects"""

    # complete the prefix and store
    fname = fname.parent.joinpath(fname.stem + "numpy.npy")
    np.save(fname, data)
    return {"extra_fname": str(fname), "type": str(type(data))}


@prepare_save
def transform_paths(data, fname=Path()):
    """Cast paths to string"""
    fname = fname.parent.joinpath(fname.stem + "ica.fif")
    return {"transformed_data": str(data), "type": str(type(data))}


@prepare_save
def transform_named_int(data, fname=Path()):
    """Cast paths to string for saving in yaml"""
    fname = fname.parent.joinpath(fname.stem + "ica.fif")
    return {"transformed_data": str(data), "type": str(type(data))}


# Note: types work as dict keys as well - nice
# pandas and polars are optional backends (``pip install xileh[pandas]`` /
# ``xileh[polars]``); their savers are only registered when the backend is
# importable. The numpy/Path savers are always available as numpy is a core
# dependency.
non_serializeable_types = {
    np.ndarray: numpy_saver,
    Path: transform_paths,
}

try:
    import pandas as pd

    non_serializeable_types[pd.core.frame.DataFrame] = pandas_saver
    non_serializeable_types[pd.core.series.Series] = pandas_saver
except ImportError:
    pass

try:
    import polars as pl

    non_serializeable_types[pl.DataFrame] = polars_saver
    non_serializeable_types[pl.Series] = polars_saver
except ImportError:
    pass


def get_saver(data):
    """Get the correct saver for a given data type"""
    # NOTE this is done instead of directly looking up in the dict,
    # as this avoids needing to include every subclass since check is done
    # via isinstance

    for k, v in non_serializeable_types.items():
        if isinstance(data, k):
            return v

    raise NotImplementedError(f"No loader implemented for type={type(data)}")


# =============================================================================
# Saving
# =============================================================================


def save_to_folder(data: dict, fname: str | Path = "", overwrite: bool = False):
    """Save data in the dictionary to a given folder"""
    # --> if there would be an overwrite and it is not specified, ask
    save = True

    fname = Path(fname).resolve()
    if fname.exists():
        if overwrite:
            pass
        else:
            q = ""
            while q not in ["y", "n"]:
                q = input(
                    f"There is already a container at {fname}\n Do you"
                    " want to overwrite [y/n]? "
                )
            if q == "y":
                overwrite = True
                print(f"Removing for overwrite: {fname}")
                shutil.rmtree(Path(fname))
            else:
                save = False

    if save:
        serializable_data = save_dict_with_non_serializables(data, fname)
        save_serializable(serializable_data, fname=fname)


def save_dict_with_non_serializables(data: dict, fname: Path):
    """
    Save the non serializeable data with its according saver functions.

    Parameters
    ----------
    data : dict
        Dictionary containing data to save, possibly non serializeable data.
        Note that the dictionary might of arbitrary depth, containing nestings
        made of dicts, lists or tuples
    fname : Path
        path to folder to store the data at

    Returns
    -------
    serializable_data : dict
        A dict which is a copy of `data` but only containing serializeable data

    """

    serializable_data = {}

    # deepest level reached if data is storeable and saver_layout only
    # contains a function

    for k, v in data.items():
        if isinstance(v, (tuple, list)):
            serializable_data[k] = save_non_serializables_in_iterable(v, fname)
        elif isinstance(v, dict):
            serializable_data[k] = save_dict_with_non_serializables(v, fname)
        else:
            serializable_data[k] = save_non_serializable(v, fname)

    return serializable_data


def save_non_serializables_in_iterable(iter, fname):
    """
    This assumes that all elements in the iterable are either serializeable
    but non dictionary or are non_serializeable_types
    """

    # Any non_serializeable_types is also unhashable as of now
    if isinstance(iter, set):
        return iter
    else:
        ret = [
            save_non_serializable(v, fname)
            if not isinstance(v, (tuple, list))
            else save_non_serializables_in_iterable(v, fname)
            for v in iter
        ]

        if isinstance(iter, tuple):
            ret = tuple(ret)
        return ret


def save_non_serializable(v, fname):
    if isinstance(v, tuple(non_serializeable_types.keys())):
        meta = get_saver(v)(v, fname=fname)

        # make path local
        if "extra_fname" in meta.keys():
            pth = Path(meta["extra_fname"])
            meta["extra_fname"] = str(
                Path(".").joinpath(pth.parent.stem, pth.stem + pth.suffix)
            )

        return meta

    elif isinstance(v, dict):
        return save_dict_with_non_serializables(v, fname)

    else:
        return v


if __name__ == "__main__":
    from xileh import xData
    import numpy as np

    pdata = xData(
        [xData(np.ones(1000), name="some_numpy"), xData("test", name="some_text")],
        name="test_container",
    )

    data = pdata._to_dict()

    save_to_folder(data, fname="test_container")
