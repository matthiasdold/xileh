from collections import namedtuple
from pathlib import Path

import mne
from mne.io.constants import FIFF
from mne.io._digitization import DigPoint
import numpy as np
import pandas as pd
from numpy.random import random  # will get called a lot
from numpy.random import rand
# from datetime import datetime


def create_basic_mixed_type_dict(small_arr_size=50, large_arr_size=10000,
                                 add_nested_dict=True):
    d = {
        'string': 'some string',
        'none': None,
        'some_small_array': random(small_arr_size),
        'some_large_array': random(large_arr_size),
        'some_tuple': (1, 2, 3, '4', '5', '6'),
        'some_list': random(10).tolist(),
        'some_float': 1.1,
        'some_int': 1,
        'some_bool': True,
        'some_path': Path('path', 'to', 'somewhere')

    }

    if add_nested_dict:
        d['nested_dict'] = create_basic_mixed_type_dict(add_nested_dict=False)

    return d


def create_named_tuple():
    nt = namedtuple('NamedTuple', ['one', 'two', 'three'])
    return nt(random(10), rand(), 'some string')


def create_pandas():
    out = {
        'pandas series': pd.Series(random(100)),
        'pandas dataframe': pd.DataFrame(data=random([10, 3]),
                                         columns=['one', 'two', 'three'])
    }
    return out


def create_mne_epochs(n_chans=10, n_samples=10000, n_epochs=3):
    info = mne.create_info(
        ch_names=[f'ch_{i}' for i in range(n_chans)], sfreq=100)
    epo = mne.EpochsArray(data=random([n_epochs, n_chans, n_samples]),
                          info=info)
    return epo


def create_dig_point():
    ident = np.random.choice([1, 2, 3, 4])
    return DigPoint(kind=1, ident=ident, coord_frame=FIFF.FIFFV_COORD_HEAD,
                    r=random(3))


def create_mne_info():
    # NOTE: This currently leads to a few deprecation warnings for mne 0.24
    # --> The mne approach is that info should not be user editable
    # besides: 'bads' and 'description'
    #
    # https://github.com/mne-tools/mne-python/issues/7818
    #
    # This complicates generating a dummy info object for testing :(
    # TODO: Think of a workaround

    # vaguely mimicing legacy data
    file_id = {
        'version': 100,
        'machid': random(2).astype(int),
        'secs': 0,
        'usecs': 10000
    }
    ch_names = ['F3', 'F4', 'C3', 'P3', 'P4']
    chs = [{
        'scanno': i+2,
        'logno': i+2,
        'kind': 2,
        'range': 1.0,
        'cal': 0.1000000000032,
        'coil_type': 1,
        'loc': random(12),
        'unit': 107,
        'unit_mul': 0,
        'ch_name': ch,
        'sfreq': 200,
        'coord_frame': FIFF.FIFFV_COORD_HEAD,
        'custom_ref_applied': False

    } for i, ch in enumerate(ch_names)]
    args = dict(
        dig=[create_dig_point() for x in range(5)],
        # file_id=file_id,
        # meas_id=file_id,
        # # (1579082795, 482000),  --> # setting as tuple this creates some difficulties with opening up the value        # noqa
        # highpass=1,
        # lowpass=90,
    )
    info = mne.create_info(ch_names, sfreq=200, ch_types='eeg')
    for k, v in args.items():
        info[k] = v

    for i, ch in enumerate(info['chs']):
        # unit_mul is named int after being read
        info['chs'][i]['unit_mul'] = int(ch['unit_mul'])

        # set scanno and logno
        info['chs'][i]['scanno'] = chs[i]['scanno']
        info['chs'][i]['logno'] = chs[i]['logno']
        info['chs'][i]['cal'] = chs[i]['cal']
        info['chs'][i]['loc'] = chs[i]['loc']
    return info


def create_fake_data(add_nested=False):
    d = {
        'named_tuple': create_named_tuple(),
        'basic_types': create_basic_mixed_type_dict(),
        'epochs_array': create_mne_epochs(),
        # 'info_data': create_mne_info(),       # see TODO above
        **create_pandas()
    }
    if add_nested:
        d['nested'] = create_fake_data()
    return d


if __name__ == "__main__":
    res = create_fake_data(add_nested=True)
