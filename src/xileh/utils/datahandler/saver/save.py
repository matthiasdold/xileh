from pathlib import Path
import shutil

import srsly

from xileh.utils.datahandler.saver.serialize import is_serializable, serialize
from xileh.utils.datahandler.saver.prepare import prepare


def _save(res, fname='test_dir', mode='json', fname_mapping=None,
          array_threshold=10000):

    fname = Path(fname)

    # TODO: Think of overwrite parameter
    if fname.exists():
        raise FileExistsError(f"Dir at {fname} already exists. No automatic"
                              " overwrite implemented yet. Please delete befor"
                              " saving."
                              )
    else:
        fname.mkdir()

    res = prepare(res, fname, mode=mode, array_threshold=array_threshold)

    if fname_mapping is None:
        fname_mapping = dict()

    for key, value in res.items():

        if key in fname_mapping:
            key = fname_mapping[key]

        if is_serializable(value, mode=mode):
            serialize(value, fname.joinpath(key), mode=mode)
            continue

        srsly.write_msgpack(fname.joinpath(f'{key}.msg'), value)

    serialize(fname_mapping, fname.joinpath('fname_mapping'), mode='json')


def save(res, fname='test_dir', mode='json', fname_mapping=None,
         array_threshold=10000):
    try:
        _save(res, fname=fname, mode=mode, fname_mapping=fname_mapping,
              array_threshold=array_threshold)
    except Exception as e:
        shutil.rmtree(fname)
        raise e
