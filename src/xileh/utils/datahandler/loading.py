#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220203
#
# The loading functionality


import tomllib
import yaml
# import orjson

# Note that during saving all types are saved from the root module with full
# name -> so no pd or np aliases for recreation
import pathlib
import pandas
import numpy

from pathlib import Path
from warnings import warn

# =============================================================================
# The loaders
# =============================================================================


def load_pandas(fname):
    return pandas.read_hdf(fname)


def load_numpy(fname):
    return numpy.load(fname)


loaders_dict = {
    pandas.DataFrame: load_pandas,
    pandas.Series: load_pandas,
    numpy.ndarray: load_numpy,
    pathlib.Path: pathlib.Path,
}


def get_loader(datatype):

    tp = datatype.split(' ')[1].replace('>', '').replace("'", "")

    # check for an appropriate ancestor
    loader = [v for k, v in loaders_dict.items() if issubclass(eval(tp), k)]

    # due to inheritance paths might return twice the same loader, get only unique          # noqa
    loader = list(set(loader))

    if loader == []:
        raise NotImplementedError(f"No loader implemented for {tp=}")
    elif len(loader) > 1:
        warn("Encountered an ambigous implementation of loaders for:"
             f" {tp=} - {loader}\n--> will continue with first: {loader[0]}")

    return loader[0]

# =============================================================================
# Script
# =============================================================================


def load_extra_data_in_dict(d: dict, froot: Path) -> dict:
    """ Go over all key val pairs an load of extra data is present """
    # if in lowest layer of extra data
    if d.keys() == {'extra_fname', 'type'}:
        # if in lowest layer
        pth = d['extra_fname']

        # pdb.set_trace()
        if Path(pth).exists():        # an absolute path was saved (legacy)
            return get_loader(d['type'])(pth)
        else:
            return get_loader(d['type'])(froot.joinpath(pth))

    elif d.keys() == {'transformed_data', 'type'}:
        return get_loader(d['type'])(d['transformed_data'])

    # else check how to iterate further
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = load_extra_data_in_dict(v, froot=froot)
        elif isinstance(v, (tuple, list)):
            d[k] = load_extra_data_in_iterable(v, froot=froot)
        else:
            d[k] = v

    return d


def load_extra_data_in_iterable(iter, froot: Path):
    """
    Load all extra data entities which might be in an iterable
    """

    # Any non_serializeable_types is also unhashable as of now
    if isinstance(iter, set):
        return iter
    else:
        ret = [load_extra_data(v, froot=froot)
               if not isinstance(v, (tuple, list))
               else load_extra_data_in_iterable(v, froot=froot)
               for v in iter]

        if isinstance(iter, tuple):
            ret = tuple(ret)
        return ret


def load_extra_data(v, froot: Path):
    """
    Analog to the saving calls, but here the final load happens in _in_dict
    """
    if isinstance(v, dict):
        return load_extra_data_in_dict(v, froot=froot)
    else:
        return v


def load_container(fname, serializeable_only=False):
    """ Load the container at the path = fname """

    fname = Path(fname).resolve()

    # check for a container.toml or container.yaml
    try:
        d = tomllib.load(open(fname.joinpath('container.toml'), 'rb'))
        # d = orjson.loads(open(fname.joinpath('container.toml'), 'rb').read())
    except FileNotFoundError:
        d = yaml.safe_load(open(fname.joinpath('container.yaml'), 'r'))

    # do also load the data stored in extras
    if not serializeable_only:
        d = load_extra_data_in_dict(d, froot=fname)

    return d


if __name__ == '__main__':
    d = load_container('test_container')
