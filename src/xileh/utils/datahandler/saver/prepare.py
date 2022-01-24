
from xileh.utils.datahandler.type_utils.constructors import TYPE_CONSTRUCTORS
from xileh.utils.datahandler.type_utils.utils import (
    is_unprepared_type_dict, is_outermost_unprepared_type_dict)

from datahandler.saver.serialize import prepare_for_serialization


def prepare(res, fname, mode=None, array_threshold=10000):
    output = {}
    for i, (key, value) in enumerate(res.items()):

        value = prepare_for_serialization(
            value, fname=fname, mode=mode, key='initial',
            array_threshold=array_threshold)

        output[key] = value

    return output

# TODO: separate prepare and unprepare functions

# NOTE: We always pass the res_dir along which is the path to the container
# folder which is being loaded, this is used to create absolute paths for
# npy and fif files, as they are stored with relative paths in the json


def unprepare(loaded, mode=None, res_dir=None):
    """This undoes everything done in prepare, reconstruct data types etc"""
    return _unprepare_container(loaded, key=loaded, mode=mode, res_dir=res_dir)


def _unprepare_container(value, key=None, mode=None, res_dir=None):
    if is_unprepared_type_dict(value):
        value = _unprepare_changed_type_dict(
            value, key=key, mode=mode, res_dir=res_dir)
    elif isinstance(value, dict):
        value = _unprepare_dict(value, key=key, mode=mode, res_dir=res_dir)
    elif isinstance(value, list):
        value = _unprepare_list(value, key=key, mode=mode, res_dir=res_dir)

    return value


def _unprepare_dict(value, key=None, mode=None, res_dir=None):
    d = dict()
    for k, v in value.items():
        d[k] = _unprepare_container(v, key=k, mode=mode, res_dir=res_dir)
    return d


def _unprepare_list(value, key=None, mode=None, res_dir=None):
    ls = []
    for x in value:
        x = _unprepare_container(x, key=key, mode=mode, res_dir=res_dir)
        ls.append(x)
    return ls


def _unprepare_changed_type_dict(value, key=None, mode=None, res_dir=None):
    value = value.copy()
    type_to_change_to = value.pop('TYPE_CHANGE')

    # we are at a leaf node
    if is_outermost_unprepared_type_dict(value):
        # also pass the dir of the container which is loaded as npy and fif
        # are stored with relative paths
        if type_to_change_to in ['NPY', 'FIF']:
            value['res_dir'] = res_dir
        return TYPE_CONSTRUCTORS[type_to_change_to](**value)

    # look further
    else:
        dd = dict()
        for k, v in value.items():
            if is_unprepared_type_dict(v):
                dd[k] = _unprepare_changed_type_dict(v, key=k, mode=mode,
                                                     res_dir=res_dir)
            else:
                dd[k] = _unprepare_container(v, key=k, mode=mode,
                                             res_dir=res_dir)

        if type_to_change_to in ['NPY', 'FIF']:
            dd['res_dir'] = res_dir

        return TYPE_CONSTRUCTORS[type_to_change_to](**dd)
