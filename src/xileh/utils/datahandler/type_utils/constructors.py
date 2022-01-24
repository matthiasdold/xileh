from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd

from mne import Info

from mne.utils._bunch import NamedInt, NamedFloat
from mne.io._digitization import DigPoint
from mne.transforms import Transform

from xileh.utils.datahandler.mne_utils.read import _construct_info_from_dict
from xileh.utils.datahandler.mne_utils.read import (read_epochs_fif,
                                                    read_raw_fif, read_ica_fif)


def _construct_tuple_from_dict(values):
    # just needed this so can be constructed with a kwarg
    return tuple(values)


def _read_npy(path, res_dir=None):
    path = Path(res_dir).joinpath(path) if res_dir is not None else path
    return np.load(path)


def _read_fif(path, info=None, res_dir=None):
    """Helper func for aligning kwarg inputs"""

    suffix_reader_map = {
        'raw.fif': read_raw_fif,
        'epo.fif': read_epochs_fif,
        'ica.fif': read_ica_fif,
    }

    path = Path(res_dir).joinpath(path) if res_dir is not None else path
    sfx = (path.stem + path.suffix)[-7:]

    assert sfx in suffix_reader_map.keys(), f"Unknown suffix '{sfx}' - " \
        f"implemented are {list(suffix_reader_map.keys())}"

    return suffix_reader_map[sfx](path, info=info)


def _read_pathlib(path):
    """Helper func for aligning kwarg inputs"""
    return Path(path)


def _construct_namedtuple(name, fields, values):
    fields = tuple(fields)
    nt = namedtuple(name, fields)
    return nt(**values)


def _construct_transform(values):
    return Transform(**values)


# Looked into creating something dynamic and not hardcoded but it gets
# unncessarily complicated
TYPE_CONSTRUCTORS = {
    # packages
    str(type(np.array([]))): np.array,
    str(type(pd.Series([], dtype='object'))): pd.Series,
    str(type(pd.DataFrame([]))): pd.DataFrame,

    # mne
    str(type(NamedFloat(name="", val=0))): NamedFloat,
    str(type(NamedInt(name="", val=0))): NamedInt,
    str(type(Info())): _construct_info_from_dict,
    str(type(DigPoint())): DigPoint,
    'TRANSFORM': _construct_transform,

    # stuff in extra folder
    'NPY': _read_npy,
    'FIF': _read_fif,

    # built in
    str(type(tuple())): _construct_tuple_from_dict,
    'PATHLIB': _read_pathlib,
    'NAMEDTUPLE': _construct_namedtuple,
}
