
from warnings import warn
from functools import partial
from pathlib import Path

import srsly
import numpy as np
import pandas as pd
from mne.utils._bunch import NamedInt, NamedFloat
from mne.io._digitization import DigPoint
from mne.transforms import Transform
from mne import Info
import mne
from xileh.utils.datahandler.mne_utils.prepare import prepare_epochs_array
from xileh.utils.datahandler.type_utils.named_tuples import isinstance_namedtuple       # noqa
from xileh.utils.datahandler.type_utils.utils import contains_tuple
from xileh.utils.datahandler.mne_utils.prepare import _save_mne_to_fif
from xileh.utils.datahandler.saver.utils import (_get_large_files_dir,
                                                 _create_extra_file_name)


from datetime import datetime


def is_serializable(obj, key=None, mode='yaml'):
    if mode == 'yaml':
        fn = srsly.is_yaml_serializable
    elif mode == 'json':
        fn = srsly.is_json_serializable
        # basically is_json_serializable wraps around ujson.dumps
        # --> we want to avoid nested try/except statements (segfaults)
        # fn = srsly.ujson.dumps
    elif callable(mode):
        fn = mode
    else:
        raise ValueError(
            'mode must be "json" or "yaml", or some callable func')
    return _is_serializable(fn=fn, obj=obj, key=key)


def serialize(obj, fname, mode='yaml', **kwargs):

    # if check here to see if TYPE_CHANGE is for mne or npy?

    if mode == 'yaml':
        fn = srsly.write_yaml
    elif mode == 'json':
        fn = srsly.write_json
    elif mode == 'msg':
        fn = srsly.write_msgpack
    elif callable(mode):
        fn = mode
    else:
        raise ValueError(
            'mode must be "json" or "yaml" or "msg", or some callable func')
    fn(f'{fname}.{mode}', obj, **kwargs)


def _is_serializable(fn, obj, key=None):

    if not _check_special_cases(obj):
        return False

    try:
        res = fn(obj)

        # The ujson.dumps would return a string if working
        if fn == srsly.ujson.dumps and isinstance(res, str):
            return True
        return res
    except Exception as e:
        if key is not None:
            warn(f'{key} raised: {e}')
        else:
            warn(f'{obj} raised: {e}')
        return False


def _check_special_cases(obj):
    if isinstance(obj, (NamedFloat, NamedInt)):
        return False
    elif contains_tuple(obj):
        return False
    else:
        return True


def _prepare_dict_for_serialization(values, mode=None, key=None, fname=None):

    if is_serializable(values, key=key, mode=mode):
        output = values
    else:
        output = {}
        for k, value in values.items():
            value = _prepare_container_for_serialization(
                value, key=k, mode=mode, fname=fname)
            assert is_serializable(
                value, key=key, mode=mode), f'{key=}, {value=}, {mode=}'
            output[k] = value

    assert is_serializable(output, key=key, mode=mode)
    return output


def _prepare_list_for_serialization(value, fname=None, key=None, mode=None):
    ls = []
    for val in value:
        val = _prepare_container_for_serialization(
            val, fname=fname, key=key, mode=mode)
        ls.append(val)
    return ls


def _prepare_tuple_for_serialization(value, fname=None, key=None, mode=None):
    tmp = dict(TYPE_CHANGE=str(type(tuple())))
    ls = []
    for val in value:
        val = _prepare_container_for_serialization(
            val, fname=fname, key=key, mode=mode)
        ls.append(val)
    tmp['values'] = ls
    return tmp


def _prepare_array_for_serialization(value, fname=None, key=None, mode=None,
                                     array_threshold=1000):
    array_threshold = int(array_threshold)

    if np.isnan(value).any():
        raise NotImplementedError(
            f"Can't handle nan values yet: {key=}, {value=}, {mode=}")
    else:
        if value.size <= array_threshold:
            print(f'Array: {key} of size {value.size} is below threshold of'
                  f' {array_threshold} but contains nans'
                  f' so it will be saved as an npy file.')

        p = _get_large_files_dir(fname)

        # create file name for array
        arr_name = _create_extra_file_name(p, key, '.npy')

        np.save(arr_name, value)
        # return dict(path=str(arr_name.absolute()), TYPE_CHANGE='NPY')
        return dict(path=str(arr_name.relative_to(fname)), TYPE_CHANGE='NPY')


def _prepare_pandas_for_serialization(value, key=None, mode=None, fname=None):
    tmp = dict(data=value.values.tolist(), TYPE_CHANGE=str(type(value)))
    return _prepare_dict_for_serialization(tmp, key=key, mode=mode,
                                           fname=fname)


def _prepare_namedtuple_for_serialization(value, key=None, mode=None,
                                          fname=None):
    # Named tuples should always have the _asdict method
    tmp = {'TYPE_CHANGE': 'NAMEDTUPLE',
           'name': value.__class__.__name__,
           'fields': value._fields,
           'values': value._asdict()}

    return _prepare_container_for_serialization(tmp, key=key, mode=mode,
                                                fname=fname)


def _prepare_pathlib_for_serialization(value, key=None, mode=None, fname=None):
    return {'path': str(value), 'TYPE_CHANGE': 'PATHLIB'}


def _prepare_container_for_serialization(value, fname, key=None, mode=None,
                                         array_threshold=1000):

    # specific types
    if isinstance(value, Transform):
        prep_fn = _prepare_mne_transform_for_serialization

    # check if is any mne fif type -> raw, epo, ica
    elif (isinstance(value, mne.EpochsArray)
          or isinstance(value, mne.io.BaseRaw)
          or isinstance(value, mne.preprocessing.ICA)
          or isinstance(value, mne.epochs.Epochs)
          or isinstance(value, mne.epochs.EpochsFIF)):
        prep_fn = _save_mne_to_fif

    elif isinstance(value, Path):
        prep_fn = _prepare_pathlib_for_serialization
    elif isinstance(value, np.ndarray):
        prep_fn = partial(_prepare_array_for_serialization,
                          array_threshold=array_threshold)
    elif isinstance(value, (pd.Series, pd.DataFrame)):
        prep_fn = _prepare_pandas_for_serialization
    elif isinstance(value, Info):
        prep_fn = _prepare_mne_info_for_serialization
    elif isinstance(value, DigPoint):
        prep_fn = _prepare_mne_dig_point_for_serialization
    elif isinstance(value, (NamedFloat, NamedInt, np.integer)):
        prep_fn = _prepare_digits_for_serialization
    elif isinstance_namedtuple(value):
        prep_fn = _prepare_namedtuple_for_serialization
    elif isinstance(value, list):
        prep_fn = _prepare_list_for_serialization
    elif isinstance(value, tuple):
        prep_fn = _prepare_tuple_for_serialization
    elif isinstance(value, dict):
        prep_fn = _prepare_dict_for_serialization

    # lets do non specific ones now
    elif is_serializable(value, key=key, mode=mode):
        # skip prep below, just return value
        prep_fn = lambda x, **kwargs: x                                             # noqa
    else:
        raise TypeError(f'{key}: {value=} is not a supported type')
    # assert is_serializable(prep_fn(value, key=key, mode=mode), key=key, mode=mode),\ # noqa
    #     f'{key=}, {value=}, {mode=}'
    return prep_fn(value, fname=fname, key=key, mode=mode)


def _prepare_digits_for_serialization(value, key=None, mode=None, fname=None):

    if isinstance(value, np.integer):
        value = int(value)
    elif isinstance(value, (NamedFloat, NamedInt)):
        value = _prepare_named_digits_for_serialization(
            value, key=key, mode=mode, fname=fname)
    else:
        raise TypeError('Not a known digit')
    return value


def prepare_for_serialization(value, mode, fname, key='outer',
                              array_threshold=10000):
    return _prepare_container_for_serialization(
        value, fname=fname, key=key, mode=mode,
        array_threshold=array_threshold)

# MNE


def _prepare_named_digits_for_serialization(value, key=None, mode=None,
                                            fname=None):
    assert isinstance(value, (NamedInt, NamedFloat))
    return {
        'TYPE_CHANGE': str(type(value)),
        'name': value._name,
        'val': float(value)
    }


def _prepare_mne_info_for_serialization(value, key=None, mode=None,
                                        fname=None):
    tmp = dict(value) | {'TYPE_CHANGE': str(type(value))}

    # convert the meas_date
    if ('meas_date' in tmp.keys()
            and isinstance(tmp['meas_date'], datetime)):
        tmp['meas_date'] = tmp['meas_date'].strftime('%Y%m%d%H%M%S')

    return _prepare_dict_for_serialization(tmp, key=key, mode=mode,
                                           fname=fname)


def _prepare_mne_dig_point_for_serialization(value, key=None, mode=None,
                                             fname=None):
    tmp = value | {'TYPE_CHANGE': str(type(value))}
    return _prepare_dict_for_serialization(tmp, key=key, mode=mode,
                                           fname=fname)


def _prepare_epochs_array_for_serialization(epochs_array, mode=None, key=None,
                                            fname=None):
    epochs_array = prepare_epochs_array(epochs_array)

    # save

    p = _get_large_files_dir(fname)

    # this might raise that mne warning about naming
    name = _create_extra_file_name(p, f'{key}_epo', 'fif')

    info = _prepare_dict_for_serialization(
        epochs_array.info, mode=mode, key=key, fname=fname)
    epochs_array.save(name)

    # return dict(path=str(name.absolute()), info=info, TYPE_CHANGE='FIF')
    return dict(path=str(name.relative_to(fname)), info=info,
                TYPE_CHANGE='FIF')


def _prepare_mne_transform_for_serialization(value, mode=None, key=None,
                                             fname=None):
    d = {
        'fro': value['from'],
        'to': value['to'],
        'trans': None if (value['trans'] == np.identity(len(value['trans']))).all() else value['trans']         # noqa
    }
    out = dict(values=d, TYPE_CHANGE='TRANSFORM')
    return _prepare_dict_for_serialization(out, mode=mode, key=key,
                                           fname=fname)
