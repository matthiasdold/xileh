import pytest
import tempfile
from pathlib import Path

from xileh.utils.datahandler.saver.prepare import prepare, unprepare

from testing.datahandler_tests.create import create_fake_data
from testing.datahandler_tests.utils import _compare_container


@pytest.fixture
def data():
    data = create_fake_data(add_nested=True)
    return data


@pytest.mark.parametrize('mode,array_threshold',
                         [('json', '10'),
                          ('json', '1000000')
                          ])
def test_prepare_unprepare(data, mode, array_threshold):
    """
    For testing prepare & unprepare functions. res == unprepare(prepare(res)).
    """
    res = data

    with tempfile.TemporaryDirectory() as tmp:
        fname = Path(tmp, 'fname')
        res_prepared = prepare(res, fname=fname, mode=mode,
                               array_threshold=array_threshold)
        res_unprepared = unprepare(res_prepared, mode=mode, res_dir=fname)

    _compare_container(res, res_unprepared, key='Initial')


if __name__ == "__main__":

    tdata = create_fake_data(add_nested=False)
    mode = 'json'
    array_threshold = 10
    tmp = tempfile.TemporaryDirectory()
    fname = Path(tmp.name, 'fname')

    prepared = prepare(tdata, fname=fname, mode=mode,
                       array_threshold=array_threshold)
    unprepared = unprepare(prepared, mode='json')

    for k in [e for e in tdata.keys()]:
        print(f"******* {k} *+++************")
        _compare_container(tdata[k], unprepared[k])
