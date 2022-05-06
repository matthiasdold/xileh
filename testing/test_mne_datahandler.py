# Testing the datahandler with mne data objects

import tempfile
import numpy as np

from pathlib import Path

from xileh.core.pipelinedata import xPData, from_container

from testing.compare_utils import _compare_container
from testing.mne_test_common import get_mne_test_data

from xileh.utils.datahandler.loading import (get_loader,
                                             load_pandas,
                                             load_numpy,
                                             mne_load_epo,
                                             mne_load_ica,
                                             mne_load_raw,
                                             mne_transform_named_int
                                             )
import pathlib
import pytest


@pytest.fixture
def mne_nested_test_data():
    mne_td = get_mne_test_data()

    tdata = xPData(
        data=[
            xPData(
                data=[
                    xPData(data=np.eye(3), header={'name': 'test'}),
                    xPData(mne_td['epos'], name='epos')
                ],
                header={'name': 'level1child'},
            ),
            xPData(mne_td['raw'], name='raw'),
            xPData(data=np.zeros(5), header={'name': 'not the target'})
        ],
        header={'name': 'outer_container',
                'description': 'A parent container without name'},
        meta={'some_meta': [1, 2, 3]}
    )
    return tdata


def test_save_load_cycle_with_dict(mne_nested_test_data):

    td = mne_nested_test_data
    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp, 'test_container')

        td.save(fpath)
        ld = from_container(fpath)

        # make sure data is loaded here already as tmp will cease to exist
        # later after, so no loading in _compare_container can take place
        ld.raw.data.load_data()
        ld.level1child.epos.data.load_data()

    _compare_container(td._to_dict(), ld._to_dict())


def test_get_loader():
    # check loader is working for individual file types
    assert get_loader("<type pathlib.Path>") == pathlib.Path
    assert get_loader("<type pathlib.PosixPath>") == pathlib.Path
    assert get_loader("<type pathlib.WindowsPath>") == pathlib.Path
    assert get_loader("<type pandas.DataFrame>") == load_pandas
    assert get_loader("<type pandas.Series>") == load_pandas
    assert get_loader("<type numpy.ndarray>") == load_numpy
    assert get_loader("<class mne.io.array.array.RawArray>") == mne_load_raw
    assert get_loader("<class mne.io.BaseRaw>") == mne_load_raw
    assert get_loader("<class mne.BaseEpochs>") == mne_load_epo
    assert get_loader("<class mne.preprocessing.ICA>") == mne_load_ica
