from pathlib import Path
import numpy as np
import tempfile
import pytest

from xileh.utils.datahandler.saver import load, save
from xileh.utils.datahandler.saver.prepare import prepare

from testing.datahandler_tests.create import create_fake_data
from testing.datahandler_tests.utils import _compare_container


@pytest.fixture  # TODO: does this need to be a fixture..?
def data():
    # This is covering the following data types:
    # named_tuples
    # basic types (None, str, int, float, tuple, list, bool, path)
    # mne.Epochs
    # pandas.Series and pandas.DataFrames
    data = create_fake_data(add_nested=True)
    return data


@pytest.mark.parametrize('mode,array_threshold',
                         [('json', '10'),
                          ('json', '1000000')
                          ])
def test_save_load(data, mode, array_threshold):
    """
    For testing prepare function against save followed by load.
    Passing this test means that save
    then load does not make any additional changes from what prepare did."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = Path(tmp, 'fname')
        fname_ = Path(tmp, 'fname_')

        data_prepared = prepare(
            data, fname=fname_, mode=mode, array_threshold=array_threshold)

        save(data, fname=fname, mode=mode, array_threshold=array_threshold)
        data_loaded = load(fname, unprepare_output=False)

    _compare_container(data_prepared, data_loaded)


@pytest.mark.parametrize('mode,array_threshold',
                         [('json', '10'),
                          ('json', '1000000')
                          ])
def test_full_save_load(data, mode, array_threshold):
    with tempfile.TemporaryDirectory() as tmp:
        fname = Path(tmp, 'fname')
        save(data, fname=fname, mode=mode, array_threshold=array_threshold)
        data_loaded = load(fname, unprepare_output=True)

    _compare_container(data, data_loaded)


@pytest.mark.parametrize('mode,array_threshold',
                         [('json', '10'),
                          ('json', '1000000')
                          ])
def test_fname_mapping(data, mode, array_threshold):
    data = data

    fname_mapping = dict(
        cfg='a',
        epochs_spoc='b',
        ev_array='c',
        info_data='d',
        labels='e',
        dataults='f',
        stim_block='g'
    )

    with tempfile.TemporaryDirectory() as tmp:
        fname = Path(tmp, 'fname')
        save(data, fname, mode=mode, fname_mapping=fname_mapping,
             array_threshold=array_threshold)
        data_loaded = load(fname, unprepare_output=True, undo_fname_map=True)

    _compare_container(data, data_loaded, key='Initial')


@pytest.mark.parametrize('threshold,mode',
                         [(10, 'json'),
                          (1000, 'json'),
                          (100000, 'json'),
                          (1000000000, 'json')])
def test_array_thresholds(threshold, mode):
    res = {str(s): np.ones(10 ** s) for s in range(0, 8)}

    with tempfile.TemporaryDirectory() as tmp:
        fname = Path(tmp, 'fname')
        save(fname=fname, res=res, mode=mode, array_threshold=threshold)

        for s in res.keys():
            if int(s) > threshold:
                p = Path(fname, 'extra', f'{s}.npy')
                assert p.exists(), f'Could not locate {p}'
            else:
                p = fname.joinpath(f'{s}.{mode}')
                assert p.exists(), f'Could not locate {p}'


@pytest.mark.parametrize('mode,array_threshold',
                         [('json', '10'),
                          ])
def test_prepare_unprepare_moved_dir(data, mode, array_threshold):
    """
    For testing prepare & unprepare functions. res == unprepare(prepare(res)).
    """

    with tempfile.TemporaryDirectory() as tmp:
        with tempfile.TemporaryDirectory() as tmp2:
            fname = Path(tmp, 'fname')
            fname2 = Path(tmp2, 'fname')
            save(data, fname=fname, mode=mode, array_threshold=array_threshold)

            # move folders
            fname.rename(fname2)

            print(f"{fname=}")
            print(f"{fname2=}")

            # loac again
            data_loaded = load(fname2, unprepare_output=True)

    _compare_container(data, data_loaded, key='Initial')
