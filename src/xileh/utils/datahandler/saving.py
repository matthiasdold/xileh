#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220201
#
# functionality for saving an xileh container
#
# [ ] TODO -> think about how to best extract some specific dependencies
#             like mne which might not be needed for every user

import types
import mne
import toml

from pathlib import Path
from functools import wraps

import pandas as pd
import numpy as np


def prepare_save(func):
    """ Prepare for any saver function by e.g. creating the fname """

    @wraps(func)
    def prepare_wrapped(*args, **kwargs):
        Path(kwargs['fname']).mkdir(exist_ok=True, parents=True)

        # add a file prefix to fname which otherwise just points to the
        # container
        # increment the data_{n} count --> explicit names will be in the toml
        data_files = list(
            Path(kwargs['fname']).joinpath('extra').glob('data_*_'))

        if data_files != []:
            nmax = max([int(p.stem.split('_')[1]) for p in data_files])

        kwargs['fname'] = Path(kwargs['fname']).joinpath('extra',
                                                         f'data_{nmax + 1}_')

        func(*args, **kwargs)

    return prepare_wrapped

# =============================================================================
# Non serializable savers
# =============================================================================


# wrapping with prepare save ensures some common preprocessing (and identifies saver functions)     # noqa
@prepare_save
def pandas_saver(data, fname=Path()):
    """ Store pandas data objects """

    print("Savign pandas")

    # complete the prefix and store
    fname = fname.parent.joinpath(fname.stem + 'pandas.hdf')

    data.to_hdf(fname, 'group1')

    return {'fname': fname, 'type': str(type(data))}


@prepare_save
def numpy_saver(data, fname=Path()):
    """ Store numpy objects """


    print("Savign numpy ")

    # complete the prefix and store
    fname = fname.parent.joinpath(fname.stem + 'numpy.npy')

    np.save(fname, data)

    return {'fname': fname, 'type': str(type(data))}


# Note: types work as dict keys as well - nice
non_serializeable_types = {
    pd.core.frame.DataFrame: pandas_saver,
    pd.core.series.Series: pandas_saver,
    np.ndarray: numpy_saver,
}


# =============================================================================
# Data layout
# =============================================================================


def get_savers_layout(data):
    """ Iterate over a data object and get a mapping layout with saver funcs

    Parameters
    ----------
    data : dict
        key value pairs to store in a container folder


    Returns
    -------
    saver_layout : dict
        key value pairs for data entities names and their savers

    """

    saver_layout = {}
    for k, v in data.items():
        saver_layout[k] = get_saver(v)

    return saver_layout


def get_saver(value):
    """ For a given value/object, get the appropriate saver"""

    if isinstance(value, dict):
        ret = get_savers_layout(value)
    elif isinstance(value, tuple(non_serializeable_types)):
        ret = non_serializeable_types[type(value)]
    elif isinstance(value, list) or isinstance(value, set):
        ret = [get_saver(d) for d in value]
    else:
        ret = value

    return ret


# =============================================================================
# Saving
# =============================================================================


def save_to_file(data, fname='', overwrite=False):
    """ Save data in the dictionary to a given folder """
    # --> if there would be an overwrite and it is not specified, ask
    save = True
    if not overwrite:
        if Path(fname).exists():
            q = ''
            while q not in ['y', 'n']:
                q = input(f"There is already a container at {fname}\n Do you"
                          " want to overwrite [y/n] ?")
            if q == 'y':
                overwrite = True
            else:
                save = False

    if save:
        saver_layout = get_savers_layout(data)

        # deal with everything that needs a saver function
        saver_layout = save_non_serializable(data, saver_layout, fname=fname)

        # the rest should be full serializeable and the layout should be ready
        fname = fname.joinpath('container.toml')
        if overwrite:
            toml.dump(saver_layout, open(fname, 'w'))


# TODO THINK over this again with a clear mind
def save_non_serializable(data, saver_layout, fname=Path()):
    """
    Save the non serializeable data with its according saver functions.
    Use the saver_layout dict to store where what was stored
    """

    for k, v in data.items():
        saver = saver_layout[k]
        print(saver)
        print(k)

        # not yet deep enough
        if isinstance(v, dict):
            print(f"Dict for {v}")
            saver_layout[k] = save_non_serializable(v, saver, fname=fname)

        elif isinstance(v, list) or isinstance(v, set):
            # TODO --> make this work for
            saver_layout[k] = process_list_for_saving_non_serializable(
                v, saver, fname=fname)
        # the saver shoud just be a function -> call it
        elif isinstance(v, tuple(non_serializeable_types)):
            print("Saving non-serializeable")
            saver_layout[k] = saver(v, fname=fname)

    return saver_layout


def process_list_for_saving_non_serializable(data, saver, fname=Path()):
    """ Note that saver will always be of same type as data if it is not a
    function
    """
    return [
        saver(e) if isinstance(saver, types.FunctionType) else
        save_non_serializable(e, saver, fname=fname) if isinstance(e, dict) else
        process_list_for_saving_non_serializable(e, saver)
        if isinstance(e, list) or isinstance(e, set)
        else e for e in data]





