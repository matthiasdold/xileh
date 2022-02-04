#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220203
#
# The loading functionality


import yaml

from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# The loaders
# =============================================================================


def load_pandas(fname):
    return pd.read_hdf(fname)


def load_numpy(fname):
    return np.load(fname)


loaders = {
    str(pd.DataFrame): load_pandas,
    str(pd.Series): load_pandas,
    str(np.ndarray): load_numpy
}


# =============================================================================
# Script
# =============================================================================

def load_extra_data_in_dict(d):
    """ Go over all key val pairs an load of extra data is present """

    for k, v in d.items():
        if isinstance(v, dict):
            # If is leaf node, we will have those two keys
            if 'extra_fname' in v.keys() and 'type' in v.keys():
                d[k] = loaders[v['type']](v['extra_fname'])
            else:
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

    fname = Path(fname).resolve()
    d = yaml.safe_load(open(fname.joinpath('container.yaml'), 'r'))

    # do also load the data stored in extras
    if not serializeable_only:
        d = load_extra_data_in_dict(d)

    return d

