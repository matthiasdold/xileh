#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220203
#
# The loading functionality

import mne
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


def transform_str_to_path(pathstr):
    return pathlib.Path(pathstr)


def mne_load_raw(fname):
    return mne.io.read_raw(fname)


def mne_load_ica(fname):
    return mne.preprocessing.read_ica(fname)


def mne_load_epo(fname):
    return mne.read_epochs(fname)


def mne_transform_named_int(ni_str):
    intval, name = ni_str.split(' ')
    return mne.utils._bunch.NamedInt(name.replace('(', '').replace(')', ''),
                                     intval)


loaders_dict = {
    pandas.DataFrame: load_pandas,
    pandas.Series: load_pandas,
    numpy.ndarray: load_numpy,
    pathlib.Path: transform_str_to_path,
    pathlib.PosixPath: transform_str_to_path,
    pathlib.WindowsPath: transform_str_to_path,
    mne.io.BaseRaw: mne_load_raw,
    mne.io.RawArray: mne_load_raw,
    mne.preprocessing.ICA: mne_load_ica,
    mne.BaseEpochs: mne_load_epo,
    mne.utils._bunch.NamedInt: mne_transform_named_int
}

mne_type_lookup = {
    'mne.epochs.Epochs': mne_load_epo,
    'mne.io.fiff.raw.Raw': mne_load_raw
}


def get_loader(datatype):

    tp = datatype.split(' ')[1].replace('>', '').replace("'", "")

    # --> just why?
    # In [46]: isinstance(mne.io.fiff.raw.Raw, mne.io.fiff.raw.Raw)
    # Out[46]: False
    if tp in mne_type_lookup.keys():
        return mne_type_lookup[tp]

    try:
        try:
            return loaders_dict[eval(tp)]
        except KeyError:
            raise NotImplementedError(f"No loader implemented for {datatype=}")
    except Exception as e:
        print(e)


def get_loader(datatype):
    tp = datatype.split(' ')[1].replace('>', '').replace("'", "")

    try:
        return loaders_dict[eval(tp)]
    except KeyError:
        raise NotImplementedError(f"No loader implemented for {datatype=}")


# =============================================================================
# Script
# =============================================================================

def load_extra_data_in_dict(d):
    """ Go over all key val pairs an load of extra data is present """

    for k, v in d.items():
        if isinstance(v, dict):
            # If is leaf node, we will have those two keys
            if 'extra_fname' in v.keys() and 'type' in v.keys():
                d[k] = get_loader(v['type'])(v['extra_fname'])
            elif 'transformed_data' in v.keys() and 'type' in v.keys():
                d[k] = get_loader(v['type'])(v['transformed_data'])
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

    fname = pathlib.Path(fname).resolve()
    d = yaml.safe_load(open(fname.joinpath('container.yaml'), 'r'))

    # do also load the data stored in extras
    if not serializeable_only:
        d = load_extra_data_in_dict(d)

    return d
