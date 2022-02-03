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

import shutil
import yaml

from pathlib import Path
from functools import wraps

import pandas as pd
import numpy as np


def prepare_save(func):
    """ Prepare for any saver function by e.g. creating the fname """

    @wraps(func)
    def prepare_wrapped(*args, **kwargs):
        Path(kwargs['fname'], 'extra').mkdir(exist_ok=True, parents=True)

        # add a file prefix to fname which otherwise just points to the
        # container
        # increment the data_{n} count --> explicit names will be in the toml
        data_files = list(
            Path(kwargs['fname'], 'extra').glob('data_*_*'))

        if data_files != []:
            nmax = max([int(p.stem.split('_')[1]) for p in data_files])
        else:
            nmax = 0

        kwargs['fname'] = Path(kwargs['fname']).joinpath('extra',
                                                         f'data_{nmax + 1}_')

        return func(*args, **kwargs)

    return prepare_wrapped

# =============================================================================
# Non serializable savers
# =============================================================================


# wrapping with prepare save ensures some common preprocessing (and identifies saver functions)     # noqa
@prepare_save
def pandas_saver(data, fname=Path()):
    """ Store pandas data objects """

    # complete the prefix and store
    fname = fname.parent.joinpath(fname.stem + 'pandas.hdf')

    data.to_hdf(fname, 'group1')

    return {'extra_fname': str(fname), 'type': str(type(data))}


@prepare_save
def numpy_saver(data, fname=Path()):
    """ Store numpy objects """

    # complete the prefix and store
    fname = fname.parent.joinpath(fname.stem + 'numpy.npy')

    np.save(fname, data)

    return {'extra_fname': str(fname), 'type': str(type(data))}


# Note: types work as dict keys as well - nice
non_serializeable_types = {
    pd.core.frame.DataFrame: pandas_saver,
    pd.core.series.Series: pandas_saver,
    np.ndarray: numpy_saver,
}


# =============================================================================
# Saving
# =============================================================================


def save_to_file(data, fname='', overwrite=False):
    """ Save data in the dictionary to a given folder """
    # --> if there would be an overwrite and it is not specified, ask
    save = True

    fname = Path(fname).resolve()
    if overwrite:
        if fname.exists():
            q = ''
            while q not in ['y', 'n']:
                q = input(f"There is already a container at {fname}\n Do you"
                          " want to overwrite [y/n]? ")
            if q == 'y':
                overwrite = True
                print(f"Removing for overwrite: {fname}")
                shutil.rmtree(Path(fname))
            else:
                save = False

    if save:
        serializable_data = save_dict_with_non_serializables(data, fname)

        # the rest should be full serializeable and the layout should be ready
        if overwrite:
            yaml.safe_dump(serializable_data,
                           open(fname.joinpath('container.yaml'), 'w'))


def save_dict_with_non_serializables(data, fname):
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
        ret = [save_non_serializable(v, fname)
               if not isinstance(v, (tuple, list))
               else save_non_serializables_in_iterable(v, fname)
               for v in iter]

        if isinstance(iter, tuple):
            ret = tuple(ret)
        return ret


def save_non_serializable(v, fname):
    if isinstance(v, tuple(non_serializeable_types.keys())):
        return non_serializeable_types[type(v)](v, fname=fname)
    elif isinstance(v, dict):
        return save_dict_with_non_serializables(v, fname)
    else:
        return v

