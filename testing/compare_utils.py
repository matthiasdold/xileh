from unittest import TestCase
from pathlib import Path

import numpy as np
import pandas as pd

from datetime import datetime


class NotEqual(Exception):
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

    elif type(obj_1) is not type(obj_2):
        # after here we know the objs are of same type
        # handle bug involving importlib.util type mismatch

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
        for k, v1 in obj_1.items():

            # NOTE: this exception is created on purpose as toml would have
            # problems with nest data structures including empty dicts
            # --> e.g. a meta = {} in the nested sample data
            # yaml was able to deal with this but falls short as it cannot
            # handle numpy.float64 etc. ....
            if isinstance(v1, dict) and v1 == {}:
                pass
            else:
                try:
                    v2 = obj_2[k]
                    print("Dict -> comparing by k: v pairs")
                    _compare_container(v1, v2, key=k)
                except KeyError:
                    raise NotEqual(f'KeyError with dict:{key=}, {k=}')

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
