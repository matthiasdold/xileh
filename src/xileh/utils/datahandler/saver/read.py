from pathlib import Path

import srsly

from xileh.utils.datahandler.mne_utils.read import read_epochs_fif


# DW:
# I think that unless a package is found that plays nicely with Windows
# (looking at you, python magic) and is able to recognise json/yaml
# (filetype D: ) the only solution for detecting
# filetypes is to record extensions and the associated read function.
# This will be painfully rigid/static but we are fortunate in being able to
# control the saved filetypes & extensions.


EXTS = {
    'json': {
        'exts': ['json', 'jsn', 'j'],
        'fn': srsly.read_json
    },
    'yaml': {
        'exts': ['yaml', 'yml'],
        'fn': srsly.read_yaml
    },
    'msgpack': {
        'exts': ['msg', 'msgpack'],
        'fn': srsly.read_msgpack
    },
    'mne': {
        'exts': ['fif'],
        'fn': read_epochs_fif
    },
}


def _find_type_from_extension(fname):
    fname = Path(fname)
    for filetype in EXTS.keys():
        # TODO: removing the '.' here, is that ok?
        if fname.suffix[1:] in EXTS[filetype]['exts']:
            return filetype
    raise ValueError(f'Format of {fname} not supported')


def _get_reader_from_type(filetype):
    return EXTS[filetype]['fn']


def get_reader(fname):
    filetype = _find_type_from_extension(fname)
    return _get_reader_from_type(filetype)
