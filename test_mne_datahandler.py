# same as test_datahandler but mne specific
from testing.test_pipelinedata import get_nested_test_data
from testing.compare_utils import _compare_container
from testing.mne_test_common import get_mne_test_data

from xileh.utils.datahandler.saving import save_to_file
from xileh.utils.datahandler.loading import (load_container,
                                             get_loader,
                                             load_pandas,
                                             load_numpy,
                                             )

import pathlib
from pathlib import Path
import numpy as np

import tempfile

@pytest.fixture
def get_nested_test_data():
    mne_data = get_mne_test_data()
    tdata = xPData(
        data=[
            xPData(
                data=[
                    xPData(data=np.eye(3), header={'name': 'test'}),
                    xPData(data=[1, 23, 4],
                           header={'name': 'somename'}),
                    xPData(mne_data['epos'], name='epos')
                ],
                header={'name': '1st_level_child'},
            ),
            xPData(
                data=[
                    xPData(data=np.eye(3), header={'name': 'test2'}),
                    xPData(data=[1, 23, 4],
                           header={'name': 'somename2'})
                ],
                header={'name': 'string_nest_c'},
            ),
            xPData(
                data=np.ones(3),
                header={'name': 'search_target',
                        'discription': 'We will search for this'}
            ),
            xPData(mne_data['raw'], name='raw'),
            xPData(data=np.zeros(5), header={'name': 'not the target'})
        ],
        header={'name': 'outer_container',
                'description': 'A parent container without name'},
        meta={'some_meta': [1, 2, 3]}
    )
    return tdata


def test_save_load_cycle_with_dict(get_nested_test_data):
    d = get_nested_test_data._to_dict()

    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp, 'test_container')

        save_to_file(d, fname=fpath)
        ld = load_container(fpath)

    _compare_container(d, ld)


def test_get_loader():
    # check loader is working for individual file types
    assert get_loader("<type pathlib.Path>") == pathlib.Path
    assert get_loader("<type pathlib.PosixPath>") == pathlib.Path
    assert get_loader("<type numpy.ndarray>") == load_numpy
