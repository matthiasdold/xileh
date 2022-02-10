#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220203
#
# The loading functionality


import yaml

# Note that during saving all types are saved from the root module with full
# name -> so no pd or np aliases for recreation
import pathlib
import pandas
import numpy

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
    pathlib.PosixPath: pathlib.Path,
    pathlib.WindowsPath: pathlib.Path,
}


def get_loader(datatype):
    tp = datatype.split(' ')[1].replace('>', '').replace("'", "")
    # check for an appropriate ancestor
    loader = [v for k, v in loaders_dict.items() if issubclass(eval(tp), k)]

    if loader == []:
        raise NotImplementedError(f"No loader implemented for {tp=}")
    elif len(loader) > 1:
        raise ValueError("Encountered an abigous implementation of loards"
                         " please report this to the package maintainers")

    return loader[0]


# =============================================================================
# Script
# =============================================================================

def load_extra_data_in_dict(d):
    """ Go over all key val pairs an load of extra data is present """
    # if in lowest layer of extra data
    if d.keys() == {'extra_fname', 'type'}:
        # if in lowest layer
        return get_loader(d['type'])(d['extra_fname'])
    elif d.keys() == {'transformed_data', 'type'}:
        return get_loader(d['type'])(d['transformed_data'])

    # else check how to iterate further
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = load_extra_data_in_dict(v)
        elif isinstance(v, (tuple, list)):
            d[k] = load_extra_data_in_iterable(v)
        else:
            d[k] = v

    return d


def load_extra_data_in_iterable(iter):
    """
    Load all extra data entities which might be in an iterable
    """

    # Any non_serializeable_types is also unhashable as of now
    if isinstance(iter, set):
        return iter
    else:
        ret = [load_extra_data(v)
               if not isinstance(v, (tuple, list))
               else load_extra_data_in_iterable(v)
               for v in iter]

        if isinstance(iter, tuple):
            ret = tuple(ret)
        return ret


def load_extra_data(v):
    """
    Analog to the saving calls, but here the final load happens in _in_dict
    """
    if isinstance(v, dict):
        return load_extra_data_in_dict(v)
    else:
        return v


def load_container(fname, serializeable_only=False):
    """ Load the container at the path = fname """

    fname = pathlib.Path(fname).resolve()
    d = yaml.safe_load(open(fname.joinpath('container.yaml'), 'r'))

    # do also load the data stored in extras
    if not serializeable_only:
        d = load_extra_data_in_dict(d)

    return d
