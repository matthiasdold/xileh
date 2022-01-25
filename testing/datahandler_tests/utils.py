from unittest import TestCase
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne import EpochsArray
from datetime import datetime

from xileh.utils.datahandler.type_utils.named_tuples import (
    isinstance_namedtuple)


class NotEqual(Exception):
    # TODO: is this needed, if so should it be somewhere else?
    pass


def _compare_container(obj_1, obj_2, key=""):
    """For handling comparison of nested unknown data types. Raises NotEqual
    when obj_1 & obj_2 do
    not match somehow. If the type cannot be handled, then TypeError is raised.
    :param obj_1: obj, thing you want to compare
    :param obj_2: obj, thing you want to compare
    :param key: str, for printing error messages in recursion
    """
    if isinstance(obj_1, (int, float, np.integer)) and isinstance(
            obj_2, (int, float, np.integer)):
        # must be called before same type checking bc np.float64 != float
        if not np.allclose(obj_1, obj_2) and not (np.isnan(obj_1) and np.isnan(
                obj_2)):
            # pdb.set_trace()
            raise NotEqual(f'Floats not equal! {key=}, {obj_1=}, {obj_2=}')

    elif isinstance_namedtuple(obj_1) and isinstance_namedtuple(obj_2):
        print("Comparing elements of named lists going deeper")
        _compare_container(obj_1._asdict(), obj_2._asdict(), key=key)

    elif type(obj_1) is not type(obj_2):
        # after here we know the objs are of same type
        # handle bug involving importlib.util type mismatch

        # make an exception for Epochs vs EpochsArray... for all we use
        # them, they can be considered equivalent
        if (isinstance(obj_1, mne.BaseEpochs)
                and isinstance(obj_2, mne.BaseEpochs)):
            assert np.allclose(obj_1.get_data(), obj_2.get_data()), "Data of "\
                f"epochs original {obj_1=} and loaded {obj_2=} does not agree"
        else:
            if str(type(obj_1)) != str(type(obj_2)):
                raise NotEqual(
                    f'Type mismatch {key=}, {type(obj_1)=} & {type(obj_2)=}')
            try:
                TestCase().assertCountEqual(obj_1.__dir__(), obj_2.__dir__())
            except AssertionError as e:
                raise NotEqual(
                    f'__dir__ mismatch {key=}, {type(obj_1)=} & {type(obj_2)=}'
                    f'\n{e}')

    elif isinstance(obj_1, np.ndarray):
        if np.isnan(obj_1).all() and np.isnan(obj_2).all():
            pass
        elif np.isnan(obj_1).any() or np.isnan(obj_2).any():
            print("Comparing arrays with nan -> casting to list and"
                  " going deeper")
            _compare_container(obj_1.tolist(), obj_2.tolist(), key=key)
        else:
            if not np.allclose(obj_1, obj_2):
                raise NotEqual(f'arrays not equal {key=}')

    elif isinstance(obj_1, str):

        # catch tempfile path comparisons
        if obj_1.startswith('/tmp/'):
            obj_1 = Path(obj_1).name
            obj_2 = Path(obj_2).name

        if obj_1 != obj_2:
            raise NotEqual(
                f'Unequal {key=}, {type(obj_1)=}, {obj_1=}, {obj_2=}')

    elif isinstance(obj_1, (bool, slice, Path)) or obj_1 is None:
        # catch things that can be easily compared
        if obj_1 != obj_2:
            raise NotEqual(
                f'Unequal {key=}, {type(obj_1)=}, {obj_1=}, {obj_2=}')

    elif isinstance(obj_1, (list, tuple)):
        print("List or tuple -> zipping and comparing one by one")
        print(f"obj1: {obj_1}")
        print(f"obj2: {obj_2}")
        for x, y in zip(obj_1, obj_2):
            _compare_container(x, y, key=key)

    elif isinstance(obj_1, dict):
        for k, v in obj_1.items():
            if k in obj_2:
                print("Dict -> comparing by k: v pairs")
                _compare_container(v, obj_2[k], key=k)
            else:
                raise NotEqual(f'KeyError with dict:{key=}, {k=}')

    elif isinstance(obj_1, EpochsArray):

        obj_1_dict = obj_1.__dict__.copy()
        obj_2_dict = obj_2.__dict__.copy()

        del obj_1_dict['_raw_times']
        del obj_2_dict['_raw_times']
        del obj_1_dict['_times_readonly']
        del obj_2_dict['_times_readonly']

        _compare_container(obj_1_dict, obj_2_dict)

    elif isinstance(obj_1, (pd.DataFrame, pd.Series)):
        _compare_container(obj_1.values, obj_2.values)

    elif isinstance(obj_1, datetime):
        # use utctimetuple which compares down to seconds -> precise enough
        assert obj_1.utctimetuple() == obj_2.utctimetuple(), "Time tuples do "\
            f"not agree got {obj_1=} vs {obj_2=}"

    else:
        raise TypeError(
            f'Type not accounted for! {key=}, {type(obj_1)=}, {obj_1=}, '
            f'{obj_2=}')
